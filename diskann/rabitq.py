# https://arxiv.org/pdf/2405.12497

import numpy as np
import msgpack
import math
import tqdm

n_dims = 1152
output_dims = 64*8
scale = 1 / math.sqrt(n_dims)
dataset = np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
queryset = np.fromfile("query.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
mean = np.mean(dataset, axis=0)

centered_dataset = dataset - mean
norms = np.linalg.norm(centered_dataset, axis=1)
centered_dataset = centered_dataset / norms[:, np.newaxis]
print(centered_dataset)

sample = centered_dataset[:64]

def random_ortho(dim):
    h = np.random.randn(dim, dim)
    q, r = np.linalg.qr(h)
    return q

p = random_ortho(n_dims) # algorithm only uses the inverse of P, so just sample that directly
p = p[:output_dims, :]

def quantize(datavecs):
    xs = (p @ datavecs.T).T
    quantized = xs > 0
    dequantized = scale * (2 * quantized - 1)
    dots = np.sum(dequantized * xs, axis=1) # <o_bar, o>
    return quantized, dots

qsample, dots = quantize(sample)
print(qsample.sum(axis=1).mean())
#print(dots)
#print(dots.mean())

def approx_dot(quantized_samples, dots, query):
    mean_to_query = np.dot(mean, query)
    print(mean_to_query)
    dequantized = scale * (2 * quantized_samples - 1)
    query_transformed = p @ query
    o_bar_dot_q = np.sum(dequantized * query_transformed, axis=1)
    return norms[:sample.shape[0]] * o_bar_dot_q * dots + mean_to_query

print(norms)
approx_results = approx_dot(qsample, dots, queryset[0])
exact_results = sample @ queryset[0]

for x in zip(approx_results, exact_results):
    print(*x)

print(*[ f"{x:.2f}" for x in (approx_results - exact_results) / np.abs(exact_results).mean() ])

print(np.argsort(approx_results))
print(np.argsort(exact_results))

with open("rabitq.msgpack", "wb") as f:
    msgpack.pack({
        "mean": mean.flatten().tolist(),
        "transform": p.flatten().tolist(),
        "output_dims": output_dims,
        "n_dims": n_dims
    }, f)
