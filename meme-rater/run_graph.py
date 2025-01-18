# claude-3

import json
import matplotlib.pyplot as plt
import sys

# Read data from log.jsonl
data = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Extract steps, loss, and val_loss
steps = [entry['step'] for entry in data if "loss" in entry]
loss = [entry['loss'] for entry in data if "loss" in entry]
val_loss_data = [entry['val_loss'] for entry in data if 'val_loss' in entry]
val_steps = [entry['step'] for entry in data if 'val_loss' in entry]

# Extract individual validation loss series
val_loss_series = {}
for val_loss in val_loss_data:
    for key, value in val_loss.items():
        if key not in val_loss_series:
            val_loss_series[key] = []
        val_loss_series[key].append(value)

# Calculate rolling average for loss
window_size = 50
rolling_avg = [sum(loss[i:i+window_size])/window_size for i in range(len(loss)-window_size+1)]
rolling_steps = steps[window_size-1:]

# Calculate rolling averages for validation loss series
val_rolling_avgs = {}
for key, series in val_loss_series.items():
    val_rolling_avgs[key] = [sum(series[i:i+window_size])/window_size for i in range(len(series)-window_size+1)]

print([(name, min(series)) for name, series in val_loss_series.items()])

# Create the plot
plt.figure(figsize=(10, 6))
#plt.plot(steps, loss, label='Loss')
plt.plot(rolling_steps, rolling_avg, label='Rolling Average (Loss)')

for key, series in val_loss_series.items():
    #plt.plot(val_steps, series, marker='o', linestyle='', label=f'Validation Loss ({key})')
    plt.plot(val_steps[window_size-1:], val_rolling_avgs[key], label=f'Rolling Average (Validation Loss {key})')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss and Validation Loss vs. Steps')
plt.legend()
plt.grid(True)
plt.show()
