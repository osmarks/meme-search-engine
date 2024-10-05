# claude-3

import json
import matplotlib.pyplot as plt
import sys

logs = sys.argv[1:]

def read_log(log):
    # Read data from log.jsonl
    data = []
    with open(log, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    print(log, data[0]) # config

    # Extract steps, loss, and val_loss
    steps = [entry['step'] for entry in data if "loss" in entry]
    loss = [entry['loss'] for entry in data if "loss" in entry]

    # Calculate rolling average for loss
    window_size = 50
    rolling_avg = [sum(loss[i:i+window_size])/window_size for i in range(len(loss)-window_size+1)]
    rolling_steps = steps[window_size-1:]

    return rolling_steps, rolling_avg

# Create the plot
plt.figure(figsize=(10, 6))
#plt.plot(steps, loss, label='Loss')
for i, log in enumerate(logs):
    rolling_steps, rolling_avg = read_log(log)
    plt.plot(rolling_steps, rolling_avg, label=f"{i}")

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
