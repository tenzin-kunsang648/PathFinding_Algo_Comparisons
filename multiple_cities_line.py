import folium
import osmnx as ox
import networkx as nx
import heapq
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import logging
import pandas as pd

# Function to retrieve street network data from OpenStreetMap
def get_street_network():
    print('in get street network')
    # geographic bounds (North, South, East, West) of the East Coast
    north, south, east, west = 45.0, 24.0, -66.0, -81.0  # approximate values

    # Download the street network data
    G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    return G

# Function to find the nearest network node
def nearest_node(graph, coordinates):
    print('in nearest node')
    return ox.distance.nearest_nodes(graph, coordinates[1], coordinates[0])

# A* algorithm implementation
def astar(graph, start, end):
    print('in astar')
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

            tentative_g_score = g_score[current] + graph.get_edge_data(current, neighbor).get('length', 1)

            if tentative_g_score < g_score[neighbor]:
                came_from_astar[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate(graph, neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Dijkstra's algorithm implementation
def dijkstra(graph, start, end):
    print('in dijkstar')
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

# Function to estimate the cost from the current node to the goal
def heuristic_cost_estimate(graph, current, goal):
    return ox.distance.great_circle(graph.nodes[current]['y'], graph.nodes[current]['x'],
                                        graph.nodes[goal]['y'], graph.nodes[goal]['x'])

# Function to run A* algorithm and measure runtime
def run_astar(graph, start_node, end_node):
    print('in run astar')
    start_time = time.time()
    route_astar = astar(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time

    if route_astar:
        distance = calculate_path_distance(graph, route_astar)
        traveltime = estimate_travel_time(distance)
    else:
        distance = 0
        traveltime = 0

    return route_astar, runtime, distance, traveltime

# Function to run Dijkstra's algorithm and measure runtime
def run_dijkstra(graph, start_node, end_node):
    print('in run dijkstar')
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
    print('in calculate path distance ')

    total_distance_meters = 0
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        edge_data = graph.get_edge_data(node1, node2)
        print(f"\nNode 1 :" + str(node1) + " ||||| Node 2: " + str(node2) + "||||| edge data: " + str(edge_data))
        total_distance_meters += edge_data.get('length', 1)
        print('total distance so far in meters-> ' + str(total_distance_meters))

    # Convert distance from meters to miles (1 mile = 1609.34 meters)
    total_distance_miles = total_distance_meters / 1609.34
    return total_distance_miles

# Function to estimate travel time given distance and average speed (constant variable)
def estimate_travel_time(distance_miles, average_speed_mph=50):
    # Time = Distance / Speed, time in hours
    time_hours = distance_miles / average_speed_mph
    
    # Convert hours to minutes (1 hour = 60 minutes)
    return time_hours * 60

# Function to print all
def print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime, astar_traveltime, dijkstra_traveltime, astar_traveldistance, dijkstra_traveldistance):
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


# Function to run the pathfinding process
def run_pathfinding(start_address, end_address, count):
    print('In run pathfinding')
    street_graph = get_street_network()

    # Convert the addresses to the nearest nodes in the network
    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    print('before run astar search')
    # Run A* algorithm and measure runtime
    astar_path, astar_runtime, astar_traveldistance, astar_traveltime = run_astar(street_graph, start_node, end_node)
    print('after run astar search')

    print('before run dijkstar search')

    # Run Dijkstra's algorithm and measure runtime
    dijkstra_path, dijkstra_runtime, dijkstra_traveldistance, dijkstra_traveltime = run_dijkstra(street_graph, start_node, end_node)
    print('after run dijkstar search')

    # Print all
    print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime, astar_traveltime, dijkstra_traveltime, astar_traveldistance, dijkstra_traveldistance)

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

    file_path = '/Users/kunsang/Desktop/5800algorithm/final/multipleCities_lineMap_{count}.png'

    print('before saving fig')

    plt.savefig(file_path)

    return astar_traveldistance

def create_distance_matrix(cities):
    print('In create distance matrix')
    matrix = {}
    count = 0
    for city1 in cities:
        matrix[city1] = {}
        for city2 in cities:
            if city1 == city2:
                matrix[city1][city2] = 0
            elif city2 in matrix and city1 in matrix[city2]:  # Use symmetry to reduce calculations
                matrix[city1][city2] = matrix[city2][city1]
            else:
                distance = run_pathfinding(city1, city2, count)
                count += 1
                matrix[city1][city2] = distance
    return pd.DataFrame(matrix)

def long_running_operation(callback=None):
    for i in range(100):
        # Some operation
        if callback:
            callback(i)

def progress(percent):
    print(f"Progress: {percent}%")

# Main function
def main():
    print('In main')
    # cities = [
    #     "New York, NY",
    #     "Washington, DC",
    #     "Boston, MA",
    #     "Baltimore, MD",
    #     "Philadelphia, PA",
    #     "Atlanta, GA",
    #     "Miami, FL",
    #     "Charlotte, NC", 
    #     "Raleigh, NC",
    #     "Jacksonville, FL"
    # ]


    cities = [
            "New York, NY",
            "Washington, DC",
            "Boston, MA"
        ]
    
    long_running_operation(progress)

    distance_matrix = create_distance_matrix(cities)
    print(distance_matrix)

# Run the main function
if __name__ == "__main__":
    main()
