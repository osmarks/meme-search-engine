import torch.nn
import torch.nn.functional as F
import torch
import sqlite3
import random
import numpy
import json
import time
from tqdm import tqdm
from torch.func import functional_call, vmap, grad

from model import Config, BradleyTerry
import shared

steps = 855
batch_size = 128
num_pairs = batch_size * 1024
device = "cuda"

config = Config(
    d_emb=1152,
    n_hidden=1,
    n_ensemble=1,
    device=device,
    dtype=torch.bfloat16
)
model = BradleyTerry(config)
modelc, _ = shared.checkpoint_for(855)
model.load_state_dict(torch.load(modelc))
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)

files = shared.fetch_all_files()
importance = {}

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

# https://pytorch.org/tutorials/intermediate/per_sample_grads.html
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = F.binary_cross_entropy(predictions, targets)
    return loss

ft_compute_grad = grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 1, 1))

pairs = []
for _ in range(num_pairs):
    pairs.append(tuple(random.sample(files, 2)))

for bstart in tqdm(range(0, len(pairs), batch_size)):
    batch = pairs[bstart:bstart + batch_size]
    filenames = [ (f1, f2) for ((f1, e1), (f2, e2)) in batch ]
    embs = torch.stack([ torch.stack((torch.Tensor(e1).to(config.dtype), torch.Tensor(e2).to(config.dtype))) for ((f1, e1), (f2, e2)) in batch ])
    inputs = embs.unsqueeze(0).expand((config.n_ensemble, batch_size, 2, config.d_emb)).to(device)
    #win_probs = model(inputs)
    # TODO gradients
    # don't take variance: do backwards pass and compute gradient norm
    grads = ft_compute_sample_grad(params, buffers, inputs, torch.full((1, batch_size), 0.95).to(device))
    total_grad_norms = torch.zeros(batch_size).to(device)
    for k, v in grads.items():
        param_dims = tuple(range(1, len(v.shape)))
        total_grad_norms += torch.linalg.vector_norm(v, dim=param_dims)
    tgn = total_grad_norms.cpu().numpy()

    for filename, tg in zip(filenames, tgn):
        importance[filename] = float(tg)

top = sorted(importance.items(), key=lambda x: -x[1])
with open("top.json", "w") as f:
    json.dump(top[:256], f)