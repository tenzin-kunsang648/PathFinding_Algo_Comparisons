import folium
import osmnx as ox
import networkx as nx
import heapq
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import logging
import sys


# Function to retrieve street network data from OpenStreetMap
def get_street_network(location):
    G = ox.graph_from_place(location, network_type="drive")
    return G

# Function to find the nearest network node
def nearest_node(graph, coordinates):
    return ox.distance.nearest_nodes(graph, coordinates[1], coordinates[0])

# Add weights for distance and time as parameters
def heuristic_cost_estimate(graph, current, goal, weight_distance, weight_time, max_speed):
    distance = ox.distance.great_circle(
        graph.nodes[current]['y'], graph.nodes[current]['x'],
        graph.nodes[goal]['y'], graph.nodes[goal]['x']
    )
    distance_miles = distance * 0.62137 # Convert earth_radius to miles
    travel_time = (distance_miles / max_speed) * 60 # in minutes
    # print('#### travel time ', travel_time)
    # print('#### distance_miles ', distance_miles)

    return (weight_distance * distance_miles) + (weight_time * travel_time)


# A* algorithm implementation
def astar(graph, start, end, weight_distance, weight_time):
    # Priority queue for open nodes
    open_set = []
    # Set of visited nodes
    closed_set = set()
    # Dictionary to store the previous node in the optimal path for A*
    came_from_astar = {}
    # Dictionary to store the cost to reach each node
    g_score = {node: float('inf') for node in graph.nodes} #initialize the g_score of all the nodes (except the first node)to infinity
    g_score[start] = 0 #initialize the g_score of the start node to 0 
    # Dictionary to store the estimated total cost from start to goal
    f_score = {node: float('inf') for node in graph.nodes} #initialize the f_score of all the nodes (except the first node) to infinity
    # initialize the f_score of the start node to the heuristic estimate from start to end node
    # initialize speed to 45 mph for the initial node's f_score
    f_score[start] = heuristic_cost_estimate(graph, start, end, weight_distance, weight_time, 45) 

    # Push start node to the priority queue
    heapq.heappush(open_set, (f_score[start], start))

    while open_set:
        current = heapq.heappop(open_set)[1] #pops the lowest fscore

        if current == end:
            # Reconstruct the path
            path = []
            while current in graph.nodes:
                path.append(current)
                current = came_from_astar.get(current)
            return path[::-1]

        closed_set.add(current)

        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue

            edge_data = graph.get_edge_data(current, neighbor)[0]
            tentative_g_score = g_score[current] + edge_data['length']

            if tentative_g_score < g_score[neighbor]:
                came_from_astar[neighbor] = current
                g_score[neighbor] = tentative_g_score
                max_speed = edge_data.get('maxspeed', '45 mph')
                if isinstance(max_speed, list):
                    max_speed = max_speed[0]
                max_speed_int = int(max_speed.split()[0])
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(graph, neighbor, end, weight_distance, weight_time, max_speed_int)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# Dijkstra's algorithm implementation with time and distance weights
def dijkstra(graph, start, end, time_weight, distance_weight):
    # Priority queue for open nodes
    open_set = []
    # Set of visited nodes
    closed_set = set()
    # Dictionary to store the cost to reach each node
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0

    # Dictionary to store the previous node in the optimal path
    came_from_dijkstra = {}

    # Push start node to the priority queue
    heapq.heappush(open_set, (0, start))

    while open_set:
        current_distance, current_node = heapq.heappop(open_set)

        if current_node == end:
            # Reconstruct the path
            path = []
            while current_node in graph.nodes:
                path.append(current_node)
                current_node = came_from_dijkstra.get(current_node)
            return path[::-1]

        closed_set.add(current_node)

        for neighbor in graph.neighbors(current_node):
            if neighbor in closed_set:
                continue

            # Get edge data
            edge_data = graph.get_edge_data(current_node, neighbor)[0]

            # Get distance from edge data
            distance_cost = edge_data.get('distance', 1)
            
            # Get speed from edge data
            max_speed = edge_data.get('maxspeed', '45 mph')
            if isinstance(max_speed, list):
                max_speed = max_speed[0]
            max_speed_int = int(max_speed.split()[0])

            # Calculate time from distance and speed
            time_cost = distance_cost / max_speed_int

            # Calculate weighted cost
            weighted_cost = time_weight * time_cost + distance_weight * distance_cost

            tentative_g_score = g_score[current_node] + weighted_cost

            if tentative_g_score < g_score[neighbor]:
                came_from_dijkstra[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))

    return None

# Modify the run_astar and run_dijkstra functions to incorporate weights
def run_astar(graph, start_node, end_node, weight_distance, weight_time):
    start_time = time.time()
    route_astar = astar(graph, start_node, end_node, weight_distance, weight_time)
    end_time = time.time()
    runtime = end_time - start_time

    # can probably do away with these logics
    if route_astar:
        distance, traveltime, avgspeed = calculate_path_distance_time(graph, route_astar)
    else:
        distance = 0
        traveltime = 0

    return route_astar, runtime, distance, traveltime, avgspeed

def run_dijkstra(graph, start_node, end_node, weight_distance, weight_time):
    start_time = time.time()
    route_dijkstra = dijkstra(graph, start_node, end_node, weight_distance, weight_time)
    end_time = time.time()
    runtime = end_time - start_time
    if route_dijkstra:
        distance,traveltime, avgspeed   = calculate_path_distance_time(graph, route_dijkstra)
    else:
        distance = 0
        traveltime = 0

    return route_dijkstra, runtime, distance, traveltime, avgspeed

# Function to calculate the total distance of a path
def calculate_path_distance_time(graph, path):
    total_distance_meters = 0
    all_speed = 0
    # number of nodes in a shortest path is 1 more than the number of edges - the edges are all stored in the list path
    number_of_nodes = 1 + len(path) 

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        edge_data = graph.get_edge_data(node1, node2)[0]
        # print(f"\nNode 1 :" + str(node1) + " ||||| Node 2: " + str(node2) + "||||| edge data: " + str(edge_data))
        total_distance_meters += edge_data['length']
        
        # default to 45 mph if maxspeed is not provided
        speed_str = edge_data.get('maxspeed', '45 mph')
        # Check if speed_str is a list, and if so, take the first element
        if isinstance(speed_str, list):
            speed_str = speed_str[0]

        speed_int = int(speed_str.split()[0])
        # print('SPEED INT -> ', speed_int)
        # edge data's max speed is in mph
        all_speed += speed_int

    # Convert distance from meters to miles (1 mile = 1609.34 meters)
    total_distance_miles = total_distance_meters / 1609.34
    avg_speed = all_speed / number_of_nodes
    total_time = total_distance_miles / avg_speed * 60

    return total_distance_miles, total_time, avg_speed

# Function to compare algorithms
def compare_algorithms(graph, start_node, end_node, weight_distance, weight_time):
    astar_result = run_astar(graph, start_node, end_node, weight_distance, weight_time)
    dijkstra_result = run_dijkstra(graph, start_node, end_node, weight_distance, weight_time)

    print("A* results (Distance, Time, Avg Speed, Runtime):      ", astar_result[2], astar_result[3],astar_result[4], astar_result[1])
    print("Dijkstra results (Distance, Time, Avg Speed, Runtime):", dijkstra_result[2], dijkstra_result[3],astar_result[4],dijkstra_result[1])

# Function to execute pathfinding and comparison
def execute_pathfinding(location, start_address, end_address):
    street_graph = get_street_network(location)
    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    # define different weights for distance and time
    weights = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]  # distance only, time only, and hybrid

    for weight_distance, weight_time in weights:
        print(f"Comparing algorithms with weights - Distance: {weight_distance}, Time: {weight_time}")
        compare_algorithms(street_graph, start_node, end_node, weight_distance, weight_time)

# Run the main execution
if __name__ == "__main__":

    # get python version
    # major, minor, micro = sys.version_info[:3]
    # print(f"Your Python version is {major}.{minor}.{micro}")

    # start_address = "New York, NY"
    # end_address = "Washington, DC"
    # location = "New York, NY"

    start_address = "1300 17th Street North, Arlington, Virginia"
    end_address = "AMC, Arlington, Virginia"
    location = "Arlington, Virginia"

    execute_pathfinding(location, start_address, end_address)
