import torch
import numpy as np

torch.set_float32_matmul_precision("high")

loaded_arrays = np.memmap("embeddings.bin", dtype=np.float16).reshape(-1, 1152)
loaded_arrays_permutation = np.random.permutation(len(loaded_arrays))

train_split = 0.8

def ckpt_path(steps):
    return f"ckpt/{steps}.pt", f"ckpt/{steps}.optim.pt"
