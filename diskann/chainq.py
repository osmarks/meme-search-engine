import numpy as np
import msgpack
import math
import torch
from torch import autograd
import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/una-dinosauria/local-search-quantization/blob/master/src/encodings/encode_chain.jl#L2
# vectorized somewhat
def viterbi_encode(
    out_codes: torch.Tensor, # N x M (ints)
    vectors: torch.Tensor, # N x D
    codebooks: torch.Tensor # M x H x D
):
    N, D = vectors.shape
    M, H, D2 = codebooks.shape
    assert D == D2

    # M x H x N - ||x-c||^2 ignoring x.T @ x component
    unary_costs = -2 * (codebooks @ vectors.T) + (torch.linalg.norm(codebooks, dim=2) ** 2).unsqueeze(-1)
    binary_costs = torch.zeros(M - 1, H, H, dtype=torch.float, device=DEVICE)
    for i in range(M - 1):
        binary_costs[i] = 2 * codebooks[i] @ codebooks[i + 1].T
    print("binary costs", binary_costs)

    min_cost = torch.zeros(H, N, dtype=torch.float, device=DEVICE)
    min_idx = torch.zeros(M, H, N, dtype=torch.int, device=DEVICE)

    cost = torch.zeros(H, N, dtype=torch.float, device=DEVICE)

    # forward pass - propagate optimal costs and indices forward
    for step in tqdm.trange(M - 1):
        if step > 0:
            unary_costs[step] += min_cost

        ucost = unary_costs[step]

        # for all possible costs at this step
        for j in range(H):
            bcost = binary_costs[step, j].unsqueeze(-1) # independent of N
            cost = ucost + bcost

            min_values, min_indices = torch.min(cost, dim=0)
            min_cost[j] = min_values
            min_idx[step, j] = min_indices

    unary_costs[-1] += min_cost

    # backward pass - propagate optimal indices backwards
    out_codes[:, -1] = torch.argmin(unary_costs[-1], dim=0)
    for i in range(M - 2, -1, -1):
        out_codes[:, i] = min_idx[i][out_codes[:, i + 1], range(N)]

def dims_for(dim, total_dims, m):
    dims_per_code = total_dims // m
    relevant_codebooks = [dim // dims_per_code]
    if relevant_codebooks[-1] < m - 1:
        relevant_codebooks.append(relevant_codebooks[-1] + 1)
    return relevant_codebooks

def update_codebooks(transformed_data, codes, h):
    n, d = transformed_data.shape
    n2, m = codes.shape
    assert n == n2

    new_codebook = torch.zeros(m, h, d, dtype=torch.float, device=DEVICE)
    for dim in tqdm.trange(d):
        relevant_codebooks = dims_for(dim, d, m)
        assignment_matrix = torch.zeros(n, len(relevant_codebooks), h, dtype=torch.float, device=DEVICE)
        indices = (
            torch.arange(n, dtype=torch.int, device=DEVICE).repeat(len(relevant_codebooks)),
            torch.arange(len(relevant_codebooks), dtype=torch.int, device=DEVICE).repeat_interleave(n),
            codes[:, relevant_codebooks].T.flatten()
        )
        assignment_matrix[indices] = 1
        #print(assignment_matrix, assignment_matrix.shape, transformed_data[:, dim], transformed_data[:, dim].shape)
        assignment_matrix = assignment_matrix.reshape(n, len(relevant_codebooks) * h)
        #print(assignment_matrix, assignment_matrix.shape, transformed_data[:, dim], transformed_data[:, dim].shape)
        #soln = torch.linalg.lstsq(assignment_matrix, transformed_data[:, dim])[0]

        reg = 1e-3 * torch.eye(len(relevant_codebooks) * h, device=DEVICE)
        A = assignment_matrix.T @ assignment_matrix + reg
        b = assignment_matrix.T @ transformed_data[:, dim]

        #print("matrix", A)

        usage = assignment_matrix.sum(dim=0)
        unused = usage < 1

        print(unused.sum().detach().item())
        soln = torch.linalg.solve(A, b)

        #print("solution", soln.reshape(len(relevant_codebooks), h))
        if unused.any():
            soln[unused] = torch.randn_like(soln[unused])

        new_codebook[relevant_codebooks, :, dim] = soln.reshape(len(relevant_codebooks), h)

        if torch.isnan(new_codebook[relevant_codebooks, :, dim]).any():
            print("oh no", dim, new_codebook, relevant_codebooks, new_codebook[relevant_codebooks, :, dim])
            print("--- dim ---", dim)
            print("- sum per column:", assignment_matrix.sum(dim=0))  # Check if any columns are all zero
            print("- rank:", torch.linalg.matrix_rank(assignment_matrix))
            print("- condition number:", torch.linalg.cond(assignment_matrix))
            raise SystemExit

    return new_codebook

BATCH = 8192

def train_chainq(vectors, m, h, transform, codebooks, n_iters):
    for i in range(n_iters):
        transformed_data = vectors @ transform.T
        codes = torch.zeros(vectors.shape[0], m, dtype=torch.int, device=DEVICE)
        for i in range(0, vectors.shape[0], BATCH):
            viterbi_encode(codes[i:i+BATCH], transformed_data[i:i+BATCH], codebooks)
        print("encoded")
        #codebooks = update_codebooks(transformed_data, codes, h)
        print("codebooks updated")

        quantized = torch.zeros_like(vectors, dtype=torch.float, device=DEVICE)
        for j in range(m):
            quantized[:] += codebooks[j, codes[:, j]]
        print("quantized")
        print((quantized - transformed_data).abs().mean(), transformed_data.abs().mean())

        print("comparing")
        res = transformed_data.T @ quantized

        print("running SVD...")
        u, s, vt = torch.linalg.svd(res)
        print("done.")
        transform = u @ vt
        print("regenerated transform")

    return codebooks, transform

with open("opq.msgpack", "rb") as f:
    data = msgpack.unpackb(f.read())

n_dims = 1152
dataset = torch.tensor(np.random.permutation(np.fromfile("embeddings.bin", dtype=np.float16).reshape(-1, n_dims).astype(np.float32))[:BATCH*1], device=DEVICE)

codebooks = torch.zeros(64, 256, n_dims, dtype=torch.float, device=DEVICE)

centroids = torch.tensor(np.array(data["centroids"]).astype(np.float32).reshape(256, n_dims), device=DEVICE)
for dim in range(n_dims):
    relevant_codebooks = dim // 64
    codebooks[relevant_codebooks, :, dim] = centroids[:, dim]

print(centroids)
#print("codebooks", codebooks.tolist())

codebooks, transform = train_chainq(dataset, 64, 256, torch.tensor(np.array(data["transform"]).astype(np.float32).reshape(n_dims, n_dims), device=DEVICE), codebooks, 100)

with open("chainq.msgpack", "wb") as f:
    msgpack.pack({
        "codebooks": codebooks.cpu().numpy().flatten().tolist(),
        "transform": transform.cpu().numpy().flatten().tolist(),
        "n_dims": n_dims,
        "n_dims_per_code": n_dims // 64
    }, f)
