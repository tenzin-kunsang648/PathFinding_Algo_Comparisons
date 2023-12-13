import folium
import osmnx as ox
import networkx as nx
import heapq
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import logging
from osmnx import utils_geo

# Function to retrieve street network data from OpenStreetMap
def get_street_network(location):
    G = ox.graph_from_place(location, network_type="drive")
    return G

# Function to find the nearest network node
def nearest_node(graph, coordinates):
    return ox.distance.nearest_nodes(graph, coordinates[1], coordinates[0])

# A* algorithm implementation
def astar(graph, start, end):
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
    f_score[start] = heuristic_cost_estimate(graph, start, end) #initialize the f_score of the start node to the heuristic estimate from start to end node

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

            tentative_g_score = g_score[current] + graph.get_edge_data(current, neighbor)[0]['length']

            if tentative_g_score < g_score[neighbor]:
                came_from_astar[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(graph, neighbor, end)
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
        current_node_gscore, current_node = heapq.heappop(open_set)
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

            # current_node_gscore = g_score[current_node]
            tentative_g_score = current_node_gscore + graph.get_edge_data(current_node, neighbor)[0]['length']

            if tentative_g_score < g_score[neighbor]:
                came_from_dijkstra[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))

    return None

# Function to estimate the cost from the current node to the goal
def heuristic_cost_estimate(graph, current, goal):
    return ox.distance.great_circle(graph.nodes[current]['y'], graph.nodes[current]['x'],
                                        graph.nodes[goal]['y'], graph.nodes[goal]['x'])

# Function to run A* algorithm and measure runtime
def run_astar(graph, start_node, end_node):
    start_time = time.time()
    route_astar = astar(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time

    if route_astar:
        distance, traveltime, avgspeed = calculate_path_distance_time(graph, route_astar)
    else:
        distance, traveltime = 0, 0

    return route_astar, runtime, distance, traveltime, avgspeed

# Function to run Dijkstra's algorithm and measure runtime
def run_dijkstra(graph, start_node, end_node):
    start_time = time.time()
    route_dijkstra = dijkstra(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time

    if route_dijkstra:
        distance, traveltime, avgspeed = calculate_path_distance_time(graph, route_dijkstra)
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
        
        speed_str = edge_data.get('maxspeed', '45 mph')
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

# Function to print all
def print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime, astar_traveltime, dijkstra_traveltime, astar_traveldistance, dijkstra_traveldistance, astar_avgspeed, dijkstra_avgspeed):
    # Print coordinates of A* path
    print("\nA* Path Coordinates:")
    print([(street_graph.nodes[node]['y'], street_graph.nodes[node]['x']) for node in astar_path])

    # Print coordinates of Dijkstra's path
    print("\nDijkstra's Path Coordinates:")
    print([(street_graph.nodes[node]['y'], street_graph.nodes[node]['x']) for node in dijkstra_path])

    # Print start and end coordinates
    print("\nStart Coordinates:", start_node)
    print("End Coordinates:", end_node)

    # Print A* algorithm runtime
    print(f"\nA* Algorithm Runtime: {astar_runtime:.4f} seconds")

    # Print Dijkstra's algorithm runtime
    print(f"\nDijkstra's Algorithm Runtime: {dijkstra_runtime:.4f} seconds")

    # Print A* algorithm travel distance
    print(f"\nA* Travel Distance: {astar_traveldistance:.4f} miles")

    # Print Dijkstra's travel distance
    print(f"\nDijkstra's Travel Distance: {dijkstra_traveldistance:.4f} miles")

    # Print A* algorithm travel time
    print(f"\nA* Travel Time: {astar_traveltime:.4f} minutes")

    # Print Dijkstra's travel time
    print(f"\nDijkstra's Travel Time: {dijkstra_traveltime:.4f} minutes")

    # Print A* algorithm travel speed
    print(f"\nA* Travel Speed: {astar_avgspeed:.4f} mph")

    # Print Dijkstra's travel speed
    print(f"\nDijkstra's Travel Speed: {dijkstra_avgspeed:.4f} mph")


def long_running_operation(callback=None):
    for i in range(100):
        # Some operation
        if callback:
            callback(i)

def progress(percent):
    print(f"Progress: {percent}%")

# Function to run the pathfinding process
def run_pathfinding(start_address, end_address, location):

    street_graph = get_street_network(location)

    # Convert the addresses to the nearest nodes in the network
    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    # Run A* algorithm and measure runtime
    astar_path, astar_runtime, astar_traveldistance, astar_traveltime, astar_avgspeed = run_astar(street_graph, start_node, end_node)

    # Run Dijkstra's algorithm and measure runtime
    dijkstra_path, dijkstra_runtime, dijkstra_traveldistance, dijkstra_traveltime, dijkstra_avgspeed = run_dijkstra(street_graph, start_node, end_node)

    # Print all
    print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime, astar_traveltime, dijkstra_traveltime, astar_traveldistance, dijkstra_traveldistance, astar_avgspeed, dijkstra_avgspeed)

    # Plot the street network using GeoPandas
    gdf_edges = ox.graph_to_gdfs(street_graph, nodes=False, edges=True)
    gdf_nodes = ox.graph_to_gdfs(street_graph, nodes=True, edges=False)

    # Plot the paths on the GeoPandas plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot street network edges
    gdf_edges.plot(ax=ax, color='black', linewidth=0.5)

    # Plot start and end nodes - green starting point, blue ending point
    gdf_nodes.loc[[start_node, end_node]].plot(ax=ax, color=['green', 'blue'], markersize=50)

    # Plot A* path
    astar_line = LineString([(street_graph.nodes[node]['x'], street_graph.nodes[node]['y']) for node in astar_path])
    gdf_astar_path = gpd.GeoDataFrame(geometry=[astar_line])
    gdf_astar_path.plot(ax=ax, color='red', linewidth=2, linestyle='dashed', label=f'A* Path (Runtime: {astar_runtime:.4f} seconds)')

    # Plot Dijkstra's path
    dijkstra_line = LineString([(street_graph.nodes[node]['x'], street_graph.nodes[node]['y']) for node in dijkstra_path])
    gdf_dijkstra_path = gpd.GeoDataFrame(geometry=[dijkstra_line])
    gdf_dijkstra_path.plot(ax=ax, color='blue', linewidth=2, linestyle='dashed', label=f"Dijkstra's Path (Runtime: {dijkstra_runtime:.4f} seconds)")

    # Set plot title and legend
    plt.title('Street Network with Paths \n Start: ' + start_address + '\n Destination:' + end_address)
    plt.legend()

    # Display the plot
    # plt.show()

    plt.savefig('/Users/kunsang/Desktop/5800algorithm/final/map_with_both_paths_lineMap.png')

# Main function
def main():
    start_address = "1300 17th Street North, Arlington, Virginia"
    end_address = "AMC, Arlington, Virginia"
    location = "Arlington, Virginia"

    # long_running_operation(progress)

    run_pathfinding(start_address, end_address, location)

# Run the main function
if __name__ == "__main__":
    main()