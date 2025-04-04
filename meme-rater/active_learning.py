import torch.nn
import torch.nn.functional as F
import torch
import sqlite3
import random
import numpy
import json
import time
from tqdm import tqdm
import sys

from model import Config, BradleyTerry
import shared

batch_size = 128
num_pairs = batch_size * 1024
device = "cuda"

config = Config(
    d_emb=1152,
    n_hidden=1,
    n_ensemble=16,
    device=device,
    dtype=torch.float32,
    output_channels=3,
    dropout=0.1
)
model = BradleyTerry(config)
modelc, _ = shared.checkpoint_for(int(sys.argv[1]))
model.load_state_dict(torch.load(modelc))
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)
model.eval()

files = shared.fetch_all_files()
variance = {}

pairs = []
for _ in range(num_pairs):
    pairs.append(tuple(random.sample(files, 2)))

model.eval()
with torch.inference_mode():
    for bstart in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[bstart:bstart + batch_size]
        filenames = [ (f1, f2) for ((f1, e1), (f2, e2)) in batch ]
        embs = torch.stack([ torch.stack((torch.Tensor(e1).to(config.dtype), torch.Tensor(e2).to(config.dtype))) for ((f1, e1), (f2, e2)) in batch ])
        inputs = embs.unsqueeze(0).expand((config.n_ensemble, batch_size, 2, config.d_emb)).to(device)
        win_probs = model(inputs)
        #print(win_probs, win_probs.shape)
        #print(win_probs.shape)
        batchvar = torch.var(win_probs, dim=0).max(-1).values
        #print(batchvar, batchvar.shape)
        for filename, var in zip(filenames, batchvar):
            variance[filename] = float(var)

top = sorted(variance.items(), key=lambda x: -x[1])
with open("top.json", "w") as f:
    json.dump(top[:50], f)
