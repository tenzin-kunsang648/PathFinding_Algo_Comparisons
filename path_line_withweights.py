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
import pandas as pd
from collections import Counter

# Function to retrieve street network data from OpenStreetMap
def get_street_network(location):
    G = ox.graph_from_place(location, network_type="drive")
    return G

# Function to retrieve street network data from OpenStreetMap using graph_from_address and distance from it (in meters) instead of a particular location
def get_street_network_from_address(address):
    G = ox.graph_from_address(address, dist=100000, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, truncate_by_edge=False, return_coords=False, clean_periphery=None, custom_filter=None)
    return G

def get_street_network_from_bbox(north, south, east, west):
    """
    Get the street network within a bounding box.

    Parameters:
    north, south, east, west: coordinates defining the bounding box
    """
    # Download the street network data
    G = ox.graph_from_bbox(north, south, east, west)
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
                max_speed_int = int(float(max_speed.split()[0]))
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
            max_speed_int = int(float(max_speed.split()[0]))

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
        distance,traveltime, avgspeed = calculate_path_distance_time(graph, route_dijkstra)
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

def display_highway_types_table(graph, astar_path, dijkstra_path):
    """
    Display a table of the counts of different types of highways in both paths.

    Parameters:
    graph: The graph object from which edge data is extracted.
    astar_path: The list of nodes representing the optimal path from A* algorithm.
    dijkstra_path: The list of nodes representing the optimal path from Dijkstra's algorithm.
    """

    def get_highway_counts(path):
        highway_types = []

        # Iterate over the path to get consecutive pairs of nodes
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = graph.get_edge_data(u, v)

            # Extract highway type; it can be a list or a single value
            highway = edge_data[0].get('highway', 'unknown')
            if isinstance(highway, list):
                highway_types.append(highway)  # Add all types if it's a list
            else:
                highway_types.append(highway)  # Add the single type

        return Counter(highway_types)

    # Get highway counts for both paths
    astar_counts = get_highway_counts(astar_path)
    dijkstra_counts = get_highway_counts(dijkstra_path)

    # Create DataFrames from the counts
    df_astar = pd.DataFrame(astar_counts.items(), columns=['Highway Type', 'Count A*'])
    df_dijkstra = pd.DataFrame(dijkstra_counts.items(), columns=['Highway Type', 'Count Dijkstra'])

    # Merge the DataFrames on 'Highway Type'
    df = pd.merge(df_dijkstra, df_astar, on='Highway Type', how='outer').fillna(0)

    # Display the table
    print('Comparison of Different Types of Roads in A* and Dijkstra\'s Paths')
    print(df)

# Function to compare algorithms
def compare_algorithms(graph, start_node, end_node, weight_distance, weight_time):
    astar_result = run_astar(graph, start_node, end_node, weight_distance, weight_time)
    dijkstra_result = run_dijkstra(graph, start_node, end_node, weight_distance, weight_time)

    print("A* results (Distance, Time, Avg Speed, Runtime):      ", astar_result[2], astar_result[3],astar_result[4], astar_result[1])
    print("Dijkstra results (Distance, Time, Avg Speed, Runtime):", dijkstra_result[2], dijkstra_result[3],astar_result[4],dijkstra_result[1])

    return astar_result[0], dijkstra_result[0], astar_result[1], dijkstra_result[1]

# Function to execute pathfinding and comparison
def execute_pathfinding(location, start_address, end_address):

    # street_graph = get_street_network(location)
    # street_graph = get_street_network_from_address(location)
    street_graph = get_street_network_from_bbox(location[0], location[1], location[2], location[3])

    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    # define different weights for distance and time
    weights = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]  # distance only, time only, and hybrid

    for weight_distance, weight_time in weights:
        print(f"Comparing algorithms with weights - Distance: {weight_distance}, Time: {weight_time}")
        astar_path, dijkstra_path, astar_runtime, dijkstra_runtime = compare_algorithms(street_graph, start_node, end_node, weight_distance, weight_time)

        # Print road types for A* and Dijkstra's optimal paths
        # display_highway_types_table(street_graph, astar_path, dijkstra_path)

        # Plot the street network using GeoPandas
        gdf_edges = ox.graph_to_gdfs(street_graph, nodes=False, edges=True)
        gdf_nodes = ox.graph_to_gdfs(street_graph, nodes=True, edges=False)

        # Plot the paths on the GeoPandas plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set background color to black
        ax.set_facecolor('black')

        # Plot street network edges
        gdf_edges.plot(ax=ax, color='dimgray', linewidth=1)

        # Plot start and end nodes with distinct colors
        gdf_nodes.loc[[start_node]].plot(ax=ax, color='lime', markersize=100, zorder=3)  # start node
        gdf_nodes.loc[[end_node]].plot(ax=ax, color='red', markersize=100, zorder=3)  # end node

        # Plot A* path
        astar_line = LineString([(street_graph.nodes[node]['x'], street_graph.nodes[node]['y']) for node in astar_path])
        gdf_astar_path = gpd.GeoDataFrame(geometry=[astar_line])
        gdf_astar_path.plot(ax=ax, color='yellow', linewidth=4, linestyle='-', label=f'A* Path (Runtime: {astar_runtime:.4f} seconds)')

        # Plot Dijkstra's path
        dijkstra_line = LineString([(street_graph.nodes[node]['x'], street_graph.nodes[node]['y']) for node in dijkstra_path])
        gdf_dijkstra_path = gpd.GeoDataFrame(geometry=[dijkstra_line])
        gdf_dijkstra_path.plot(ax=ax, color='cyan', linewidth=2, linestyle='-', label=f"Dijkstra's Path (Runtime: {dijkstra_runtime:.4f} seconds)")

        # Set plot title, legend, and adjust the text color for visibility
        plt.title('Street Network with Weights: (dist_weight, time_weight) = (' + str(weight_distance) + ', ' + str(weight_time) + 
        ')\n Start: ' + start_address + '\n Destination:' + end_address, color='black')
        plt.legend(facecolor='black', edgecolor='white', labelcolor = 'white', framealpha=1, fontsize='medium')
        plt.xticks(color='black')
        plt.yticks(color='black')

        # Display the plot
        plt.show()

        # Save the plot
        # plt.savefig('/Users/kunsang/Desktop/5800algorithm/final/visualMaps/bothPaths_lineMap.png', facecolor=fig.get_facecolor())

# Run the main execution
if __name__ == "__main__":

    # get python version
    # major, minor, micro = sys.version_info[:3]
    # print(f"Your Python version is {major}.{minor}.{micro}")

    # start_address = "1300 17th Street North, Arlington, Virginia"
    # end_address = "AMC, Arlington, Virginia"
    # location = "Arlington, Virginia"

    # start_address = "252 First Ave Loop, New York, NY 10009"
    # end_address = "881 7th Ave, New York, NY 10019"
    # location = "Manhattan, New York, NY"

    # start_address = "99 Margaret Corbin Dr, New York, NY 10040"
    # end_address = "11 Wall St, New York, NY 10005"
    # location = "Manhattan, New York, NY"

    # start_address = "200 Santa Monica Pier, Santa Monica, CA 90401"
    # end_address = "6100 Sepulveda Blvd, Los Angeles, CA 91411"
    # location = "Los Angeles, CA"

    # start_address = "20601 Bohemian Ave, Monte Rio, CA 95462"
    # end_address = "18000 Old Winery Rd, Sonoma, CA 95476"
    # location = "California"
    # will be checking 100,000 miles from current location - no need for location variable 
    # need to use get_street_network_from_address in run_pathfinding instead of get_street_network
    # also comment out this line when running run_pathfinding(start_address, end_address, location)
    # execute_pathfinding(start_address, end_address, start_address) #100000 meters

    # start_address = "Groom, TX 79039"
    # end_address = "Estelline, TX 79233"
    # location = "Texas"
    # # will be checking 100,000 miles from current location - no need for location variable 
    # # need to use get_street_network_from_address in run_pathfinding instead of get_street_network
    # # also comment out this line when running run_pathfinding(start_address, end_address, location)
    # execute_pathfinding(start_address, end_address, start_address) #100000 meters

    # start_address = "1300 17th St N, Arlington, VA 22209"
    # end_address = "Adams Morgan, Washington, DC"
    # location = "DMV Area"
    # # execute_pathfinding(start_address, end_address, start_address) # 20000 meters 

    # start_address = "Beckley, West Virginia 25801"
    # end_address = "Coal City, West Virginia"
    # location = [37.78, 37.67, -81.18, -81.22] # north, south, east, west

    # start_address = "3030 Holmes Ave, Minneapolis, MN"
    # end_address = "119 N 4th St, Minneapolis, MN 55401"
    # location = [44.99, 44.93, -93.26, -93.30] # north, south, east, west

    start_address = "6008 Stallion Chase Ct, Fairfax, VA 22030"
    end_address = "1300 17th St N, Arlington, VA 22209"
    location = [38.9136462, 38.7979889, -77.0525402, -77.8065756] # north, south, east, west

    print("STARTING LOCATION : " + start_address)
    print("DESTINATION LOCATION : " + end_address)

    # execute_pathfinding(start_address, end_address, start_address) 
    execute_pathfinding(location, start_address, end_address)
