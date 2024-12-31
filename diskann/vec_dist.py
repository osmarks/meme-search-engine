import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

A = 0.6
LOG_A = np.log(A)

def scale(xs):
    return np.sign(xs) * (np.log(np.abs(xs) + A) - LOG_A)

n_dims = 1152
n_used_dims = 32
data = np.frombuffer(open("embeddings.bin", "rb").read(), dtype=np.float16).reshape(-1, n_dims).astype(np.float32) # TODO

# Create histogram bins
n_bins = 256
s = __import__("math").sqrt(n_dims)
hist_range = (-1.2, 1.2)
histogram_data = np.zeros((n_used_dims, n_bins))

# Calculate histograms for each dimension
for dim in range(n_used_dims):
    dbd = data[:, dim]
    dbd = (dbd - np.mean(dbd)) / np.std(dbd)
    dbd = scale(dbd)
    hist, _ = np.histogram(dbd, bins=n_bins, range=hist_range, density=True)
    histogram_data[dim] = hist

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(histogram_data,
            cmap='viridis',
            xticklabels=np.linspace(hist_range[0], hist_range[1], n_bins),
            yticklabels=range(n_used_dims),
            cbar_kws={'label': 'Density'})

plt.xlabel('Value')
plt.ylabel('Dimension')
plt.title('Distribution Heatmap of First 16 Dimensions')

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()
