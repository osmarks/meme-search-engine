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
from collections import defaultdict

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

files = shared.fetch_all_files()
results = {}

model.eval()
with torch.inference_mode():
    for bstart in tqdm(range(0, len(files), batch_size)):
        batch = files[bstart:bstart + batch_size]
        filenames = [ f1 for f1, e1 in batch ]
        embs = torch.stack([ torch.Tensor(e1).to(config.dtype) for f1, e1 in batch ])
        inputs = embs.unsqueeze(0).expand((config.n_ensemble, len(batch), config.d_emb)).to(device)
        scores = model.ensemble(inputs).median(dim=0).values.cpu().numpy()
        #print(batchvar, batchvar.shape)
        for filename, score in zip(filenames, scores):
            results[filename] = score

channel = int(sys.argv[2])
percentile = float(sys.argv[3])
output_pairs = int(sys.argv[4])
mean_scores = numpy.mean(numpy.stack([score for filename, score in results.items()]))
top = sorted(((filename, score) for filename, score in results.items()), key=lambda x: x[1][channel], reverse=True)
select_from = top[:int(len(top) * percentile)]

out = []
for _ in range(output_pairs):
    # dummy score for compatibility with existing code
    out.append(((random.choice(select_from)[0], random.choice(select_from)[0]), 0))

with open("top.json", "w") as f:
    json.dump(out, f)
