import folium
import osmnx as ox
import networkx as nx
import heapq
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import logging

# Function to retrieve street network data from OpenStreetMap
def get_street_network(location):
    G = ox.graph_from_place(location, network_type="drive")
    return G

# Function to find the nearest network node
def nearest_node(graph, coordinates):
    return ox.distance.nearest_nodes(graph, coordinates[1], coordinates[0])

# Add weights for distance and time as parameters
def heuristic_cost_estimate(graph, current, goal, weight_distance, weight_time, average_speed_mph=50):
    distance = ox.distance.great_circle(
        graph.nodes[current]['y'], graph.nodes[current]['x'],
        graph.nodes[goal]['y'], graph.nodes[goal]['x']
    )
    distance_miles = distance / 1609.34  # Convert meters to miles
    travel_time = (distance_miles / average_speed_mph) * 60 # in minutes
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
    f_score[start] = heuristic_cost_estimate(graph, start, end, weight_distance, weight_time) #initialize the f_score of the start node to the heuristic estimate from start to end node

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

            tentative_g_score = g_score[current] + graph.get_edge_data(current, neighbor).get('length', 1)

            if tentative_g_score < g_score[neighbor]:
                came_from_astar[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(graph, neighbor, end, weight_distance, weight_time)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Dijkstra's algorithm implementation
def dijkstra(graph, start, end):
    # Priority queue for open nodes
    open_set = []
    # Set of visited nodes
    closed_set = set()
    # Dictionary to store the cost to reach each node
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0

    # Dictionary to store the previous node in the optimal path for Dijkstra's algorithm
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

            tentative_g_score = g_score[current_node] + graph.get_edge_data(current_node, neighbor).get('length', 1)

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
        distance = calculate_path_distance(graph, route_astar)
        traveltime = estimate_travel_time(distance)
    else:
        distance = 0
        traveltime = 0

    return route_astar, runtime, distance, traveltime

def run_dijkstra(graph, start_node, end_node):
    start_time = time.time()
    route_dijkstra = dijkstra(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time
    if route_dijkstra:
        distance = calculate_path_distance(graph, route_dijkstra)
        traveltime = estimate_travel_time(distance)
    else:
        distance = 0
        traveltime = 0

    return route_dijkstra, runtime, distance, traveltime

# Function to calculate the total distance of a path
def calculate_path_distance(graph, path):
    total_distance_meters = 0
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        edge_data = graph.get_edge_data(node1, node2)
        # print(f"\nNode 1 :" + str(node1) + " ||||| Node 2: " + str(node2) + "||||| edge data: " + str(edge_data))
        total_distance_meters += edge_data.get('length', 1)
        # print('total distance so far in meters-> ' + str(total_distance_meters))

    # Convert distance from meters to miles (1 mile = 1609.34 meters)
    total_distance_miles = total_distance_meters / 1609.34
    return total_distance_miles

# Function to estimate travel time given distance and average speed (constant variable)
def estimate_travel_time(distance_miles, average_speed_mph=50):
    # Time = Distance / Speed, time in hours
    time_hours = distance_miles / average_speed_mph
    
    # Convert hours to minutes (1 hour = 60 minutes)
    return time_hours * 60

# Function to compare algorithms
def compare_algorithms(graph, start_node, end_node, weight_distance, weight_time):
    astar_result = run_astar(graph, start_node, end_node, weight_distance, weight_time)
    dijkstra_result = run_dijkstra(graph, start_node, end_node)

    print("A* results (Distance, Time, Runtime):      ", astar_result[2], astar_result[3], astar_result[1])
    print("Dijkstra results (Distance, Time, Runtime):", dijkstra_result[2], dijkstra_result[3], dijkstra_result[1])

# Function to execute pathfinding and comparison
def execute_pathfinding(location, start_address, end_address):
    street_graph = get_street_network(location)
    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    # define different weights for distance and time
    weights = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]  # Example weights: distance only, time only, and hybrid

    for weight_distance, weight_time in weights:
        print(f"Comparing algorithms with weights - Distance: {weight_distance}, Time: {weight_time}")
        compare_algorithms(street_graph, start_node, end_node, weight_distance, weight_time)

# Run the main execution
if __name__ == "__main__":
    # start_address = "New York, NY"
    # end_address = "Washington, DC"
    # location = "New York, NY"

    start_address = "1300 17th Street North, Arlington, Virginia"
    end_address = "AMC, Arlington, Virginia"
    location = "Arlington, Virginia"
    execute_pathfinding(location, start_address, end_address)
