import numpy as np
import msgpack
import math
import torch
import faiss
import tqdm

n_dims = 1152
output_code_size = 64
output_code_bits = 8
output_codebook_size = 2**output_code_bits
n_dims_per_code = n_dims // output_code_size
dataset = np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
queryset = np.fromfile("query.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
device = "cpu"

def pq_assign(centroids, batch):
    quantized = torch.zeros_like(batch)

    # Assign to nearest centroid in each subspace
    for dmin in range(0, n_dims, n_dims_per_code):
        dmax = dmin + n_dims_per_code
        similarities = torch.matmul(batch[:, dmin:dmax], centroids[:, dmin:dmax].T)
        assignments = similarities.argmax(dim=1)
        quantized[:, dmin:dmax] = centroids[assignments, dmin:dmax]

    return quantized

with open("opq.msgpack", "rb") as f:
    data = msgpack.unpack(f)
    centroids = torch.tensor(data["centroids"], device=device).reshape(2**output_code_bits, n_dims)
    projection = torch.tensor(data["transform"], device=device).reshape(n_dims, n_dims)

vectors = torch.tensor(dataset, device=device)
queries = torch.tensor(queryset, device=device)

sample_size = 64
qsample = pq_assign(centroids, vectors[:sample_size] @ projection)
print(qsample)
print(vectors[:sample_size])
exact_results = vectors[:sample_size] @ queries[0]
approx_results = qsample @ (projection @ queries[0])
print(np.argsort(approx_results))
print(np.argsort(exact_results))
