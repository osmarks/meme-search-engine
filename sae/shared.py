import torch
import pyarrow as pa

torch.set_float32_matmul_precision("high")

with pa.memory_map("../../sample_1m.arrow", "r") as source:
    loaded_arrays = pa.ipc.open_file(source).read_all()

train_split = 0.8

def ckpt_path(steps):
    return f"ckpt/{steps}.pt", f"ckpt/{steps}.optim.pt"