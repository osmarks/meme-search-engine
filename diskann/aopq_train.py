import numpy as np
import msgpack
import math
import torch
from torch import autograd
import faiss
import tqdm

n_dims = 1152
output_code_size = 64
output_code_bits = 8
output_codebook_size = 2**output_code_bits
n_dims_per_code = n_dims // output_code_size
dataset = np.random.permutation(np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims)).astype(np.float32)
queryset = np.random.permutation(np.fromfile("query.bin", dtype=np.float16).reshape(-1, n_dims))[:100000].astype(np.float32)
device = "cuda"

def pq_assign(centroids, batch):
    quantized = torch.zeros_like(batch)

    # Assign to nearest centroid in each subspace
    for dmin in range(0, n_dims, n_dims_per_code):
        dmax = dmin + n_dims_per_code
        similarities = torch.matmul(batch[:, dmin:dmax], centroids[:, dmin:dmax].T)
        assignments = similarities.argmax(dim=1)
        quantized[:, dmin:dmax] = centroids[assignments, dmin:dmax]

    return quantized

# OOD-DiskANN (https://arxiv.org/abs/2211.12850) uses a more complicated scheme because it uses L2 norm
# We only care about inner product so our quantization error (wrt a query) is just abs(dot(query, centroid - vector))
# Directly optimize for this (wrt top queries; it might actually be better to use a random sample instead?)
def partition(vectors, centroids, projection, opt, queries, k, max_iter=100, batch_size=4096, query_batch_size=2048):
    n_vectors = len(vectors)
    #perm = torch.randperm(n_vectors, device=device)

    t = tqdm.trange(max_iter)
    for iter in t:
        total_loss = 0
        opt.zero_grad(set_to_none=True)

        # randomly sample queries (with replacement, probably fine)
        queries_for_iteration = queries[torch.randint(0, len(queries), (query_batch_size,), device=device)]

        for i in range(0, n_vectors, batch_size):
            loss = torch.tensor(0.0, device=device)
            batch = vectors[i:i+batch_size] @ projection
            quantized = pq_assign(centroids, batch)
            residuals = batch - quantized

            batch_error = queries_for_iteration @ residuals.T

            loss += torch.mean(torch.pow(batch_error, 2))

            total_loss += loss.detach().item()
            loss.backward()

        opt.step()

        t.set_description(f"loss: {total_loss:.4f}")

def random_ortho(dim):
    h = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(h)
    return q

# non-parametric OPQ algorithm (roughly)
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/opq_tr.pdf
projection = random_ortho(n_dims)
vectors = torch.tensor(dataset, device=device)
queries = torch.tensor(queryset, device=device)
perm = torch.randperm(len(vectors), device=device)
centroids = vectors[perm[:output_codebook_size]]
centroids.requires_grad = True
opt = torch.optim.Adam([centroids], lr=0.0005)
for i in range(30):
    # update centroids to minimize query-aware quantization loss
    partition(vectors, centroids, projection, opt, queries, output_codebook_size, max_iter=300)
    # compute new projection as R = VU^T from XY^T = USV^T (SVD)
    # where X is dataset vectors, Y is quantized dataset vectors
    with torch.no_grad():
        y = pq_assign(centroids, vectors)
        # paper uses D*N and not N*D in its descriptions for whatever reason (so we transpose when they don't)
        u, s, vt = torch.linalg.svd(vectors.T @ y)
        projection = vt.T @ u.T

    with open("opq.msgpack", "wb") as f:
        msgpack.pack({
            "centroids": centroids.detach().cpu().numpy().flatten().tolist(),
            "transform": projection.cpu().numpy().flatten().tolist(),
            "n_dims_per_code": n_dims_per_code,
            "n_dims": n_dims
        }, f)

print("done")
