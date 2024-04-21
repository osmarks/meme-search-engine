import matplotlib.pyplot as plt
import json

# Data as a JSON string
data_json = '{"0.95":22,"0.75":21,"0.5":15,"0.98":23,"0.25":3,"0.05":0,"0.99":24,"0.1":2,"0.01":0,"0.02":0}'

# Parse the JSON string into a dictionary
data = json.loads(data_json)

# Extract the keys and values from the dictionary
keys = list(data.keys())
values = list(data.values())

# Convert the keys to floats
keys = [float(key) for key in keys]

# Sort the keys and values based on the keys
sorted_data = sorted(zip(keys, values))
keys, values = zip(*sorted_data)

plt.plot(keys, values)

# Set the x-axis tick labels
plt.xticks(keys, rotation=45)

# Add labels and title
plt.xlabel('Percentile')
plt.ylabel('Memes Kept')
plt.title('Final Model Evaluation')

# Display the plot
plt.tight_layout()
plt.show()
