import pandas as pd

# Assuming 'dist_matrix' is your distance matrix as a list of lists or a 2D array and 'cities' is the list of city names
dist_matrix = [
        [0, 225, 215, 190, 95, 745, 1305, 635, 650, 885],
        [225, 0, 410, 35, 225, 635, 1090, 370, 395, 665],
        [215,410, 0, 365, 215, 1005, 1260, 860, 890, 1200],
        [190, 35, 365, 0, 190, 650, 1110, 390, 415, 685],
        [95, 225, 215,190,0,655,1075,410,425,665],
        [745,635,1005,650,655,0,660,440,430,330],
        [1305,1090,1260,1110,1075,660,0,810,800,495],
        [635,370,860,390,410,440,810,0,30,655],
        [650,395,890,415,425,430,800,30,0,685],
        [885,665,1200,685,665,330,495,655,685,0]
    ]

cities = [
        "New York, NY",
        "Washington, DC",
        "Boston, MA",
        "Baltimore, MD",
        "Philadelphia, PA",
        "Atlanta, GA",
        "Miami, FL",
        "Charlotte, NC", 
        "Raleigh, NC",
        "Jacksonville, FL"
    ]

# Converting the matrix to a DataFrame
data = {
    'start_location': [],
    'end_location': [],
    'distance_miles': []
}

# Populate the DataFrame
for i, origin in enumerate(cities):
    for j, destination in enumerate(cities):
        if i < j:  # This ensures we don't duplicate distances since it's a symmetric matrix
            data['start_location'].append(origin)
            data['end_location'].append(destination)
            data['distance_miles'].append(dist_matrix[i][j])

data_df = pd.DataFrame(data)
file_path = '/Users/kunsang/Desktop/5800algorithm/final/distance_eastcoast.csv'
data_df.to_csv(file_path, index=False)
print(data_df)

