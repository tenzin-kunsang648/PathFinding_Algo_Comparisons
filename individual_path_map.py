import folium
import osmnx as ox
import networkx as nx
import heapq
import time
import geopandas as gpd

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
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    # Dictionary to store the estimated total cost from start to goal
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic_cost_estimate(graph, start, end)

    # Push start node to the priority queue
    heapq.heappush(open_set, (f_score[start], start))

    while open_set:
        current = heapq.heappop(open_set)[1]

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
    # Priority queue for open nodes
    open_set = []
    # Set of visited nodes
    closed_set = set()
    # Dictionary to store the previous node in the optimal path for Dijkstra's algorithm
    came_from_dijkstra = {}
    # Dictionary to store the cost to reach each node
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0

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
    start_time = time.time()
    route_astar = astar(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time
    return route_astar, runtime

# Function to run Dijkstra's algorithm and measure runtime
def run_dijkstra(graph, start_node, end_node):
    start_time = time.time()
    route_dijkstra = dijkstra(graph, start_node, end_node)
    end_time = time.time()
    runtime = end_time - start_time
    return route_dijkstra, runtime

# Function to print all
def print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime):
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


# Function to run the pathfinding process
def run_pathfinding(start_address, end_address, location):


    street_graph = get_street_network(location)

    # Convert the addresses to the nearest nodes in the network
    start_node = nearest_node(street_graph, ox.geocode(start_address))
    end_node = nearest_node(street_graph, ox.geocode(end_address))

    # Run A* algorithm and measure runtime
    astar_path, astar_runtime = run_astar(street_graph, start_node, end_node)

    # Run Dijkstra's algorithm and measure runtime
    dijkstra_path, dijkstra_runtime = run_dijkstra(street_graph, start_node, end_node)

    # Print all
    print_all(street_graph, astar_path, dijkstra_path, start_node, end_node, astar_runtime, dijkstra_runtime)

    # Create a Folium map centered around the starting point
    map_center = [street_graph.nodes[start_node]['y'], street_graph.nodes[start_node]['x']]
    mymap = folium.Map(location=map_center, zoom_start=15)

    # Plot the street network on the Folium map
    folium.TileLayer('cartodb positron').add_to(mymap)
    ox.plot_graph_folium(street_graph, folium_map=mymap, popup_attribute='name')

    # Plot the A* path on the Folium map
    folium.PolyLine(locations=[(street_graph.nodes[node]['y'], street_graph.nodes[node]['x']) for node in astar_path], 
                    color="red", weight=6, opacity=1, popup=f"A* Path (Runtime: {astar_runtime:.4f} seconds)").add_to(mymap)

    # Plot the Dijkstra's path on the Folium map
    folium.PolyLine(locations=[(street_graph.nodes[node]['y'], street_graph.nodes[node]['x']) for node in dijkstra_path], 
                    color="blue", weight=6, opacity=1, popup=f"Dijkstra's Path (Runtime: {dijkstra_runtime:.4f} seconds)").add_to(mymap)

    # Add markers for start and end points
    folium.Marker([street_graph.nodes[start_node]['y'], street_graph.nodes[start_node]['x']], popup='Start Point', icon=folium.Icon(color='green')).add_to(mymap)
    folium.Marker([street_graph.nodes[end_node]['y'], street_graph.nodes[end_node]['x']], popup='End Point', icon=folium.Icon(color='blue')).add_to(mymap)

    mymap.save('/Users/kunsang/Desktop/5800algorithm/final/map_with_both_paths_realMap.html')

    # Save the map as an HTML file
    # mymap.savefig('/Users/kunsang/Desktop/5800algorithm/final/map_with_both_paths_lineMap.png')

# Main function
def main():
    start_address = "1300 17th Street North, Arlington, Virginia"
    end_address = "AMC, Arlington, Virginia"
    location = "Arlington, Virginia"

    run_pathfinding(start_address, end_address, location)

# Run the main function
if __name__ == "__main__":
    main()
