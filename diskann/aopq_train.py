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
dataset = np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
queryset = np.fromfile("query.bin", dtype=np.float16).reshape(-1, n_dims)[:100000].astype(np.float32)
device = "cpu"

index = faiss.index_factory(n_dims, "HNSW32,SQfp16", faiss.METRIC_INNER_PRODUCT)
index.train(queryset)
index.add(queryset)
print("index ready")

T = 64

nearby_query_indices = torch.zeros((dataset.shape[0], T), dtype=torch.int32)

SEARCH_BATCH_SIZE = 1024

for i in range(0, len(dataset), SEARCH_BATCH_SIZE):
    res = index.search(dataset[i:i+SEARCH_BATCH_SIZE], T)
    nearby_query_indices[i:i+SEARCH_BATCH_SIZE] = torch.tensor(res[1])

print("query indices ready")

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
def partition(vectors, centroids, projection, opt, queries, nearby_query_indices, k, max_iter=100, batch_size=4096):
    n_vectors = len(vectors)
    perm = torch.randperm(n_vectors, device=device)

    t = tqdm.trange(max_iter)
    for iter in t:
        total_loss = 0
        opt.zero_grad(set_to_none=True)

        for i in range(0, n_vectors, batch_size):
            loss = torch.tensor(0.0, device=device)
            batch = vectors[i:i+batch_size] @ projection
            quantized = pq_assign(centroids, batch)
            residuals = batch - quantized

            # for each index in our set of nearby queries
            for j in range(0, nearby_query_indices.shape[1]):
                queries_for_batch_j = queries[nearby_query_indices[i:i+batch_size, j]]
                # minimize quantiation error in direction of query, i.e. mean abs(dot(query, centroid - vector))
                # PyTorch won't do batched dot products cleanly, to spite me. Do componentwise multiplication and reduce.
                sg_errs = (queries_for_batch_j * residuals).sum(dim=-1)
                loss += torch.mean(torch.abs(sg_errs))

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
opt = torch.optim.Adam([centroids], lr=0.001)
for i in range(30):
    # update centroids to minimize query-aware quantization loss
    partition(vectors, centroids, projection, opt, queries, nearby_query_indices, output_codebook_size, max_iter=8)
    # compute new projection as R = VU^T from XY^T = USV^T (SVD)
    # where X is dataset vectors, Y is quantized dataset vectors
    with torch.no_grad():
        y = pq_assign(centroids, vectors)
        # paper uses D*N and not N*D in its descriptions for whatever reason (so we transpose when they don't)
        u, s, vt = torch.linalg.svd(vectors.T @ y)
        projection = vt.T @ u.T

print("done")

with open("opq.msgpack", "wb") as f:
    msgpack.pack({
        "centroids": centroids.detach().cpu().numpy().flatten().tolist(),
        "transform": projection.cpu().numpy().flatten().tolist(),
        "n_dims_per_code": n_dims_per_code,
        "n_dims": n_dims
    }, f)
