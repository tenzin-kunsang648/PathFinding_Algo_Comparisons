import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the image
weighted_averages = np.array([
    4.980422409, 3.422513888, 14.30405611, 17.50858323, 
    52.41384087, 92.42258514, 3.840211619, 12.22673953, 
    5.151258383, 23.40492355
])
unweighted_averages = np.array([
    4.5732, 3.0014, 12.2488, 15.604, 
    20.9671, 21.7544, 3.0986, 12.0381, 
    2.8798, 19.9674
])

# Indices for x-axis
indices = np.arange(len(weighted_averages))

# Plotting the line charts
plt.figure(figsize=(10, 5))

plt.plot(indices, weighted_averages, marker='o', color='blue', label='Weighted Averages')
plt.plot(indices, unweighted_averages, marker='o', color='red', label='Unweighted Averages')

plt.title('Weighted vs Unweighted Averages')
plt.xlabel('Location')
plt.ylabel('Distance (in miles)')
plt.legend()

plt.tight_layout()
plt.show()
