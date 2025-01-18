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
import msgpack

from model import Config, BradleyTerry
import shared

def fetch_files_with_timestamps():
    csr = shared.db.execute("SELECT filename, embedding, timestamp FROM files WHERE embedding IS NOT NULL")
    x = [ (row[0], numpy.frombuffer(row[1], dtype="float16").copy(), row[2]) for row in csr.fetchall() ]
    csr.close()
    return x

batch_size = 2048
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
model.eval()
modelc, _ = shared.checkpoint_for(int(sys.argv[1]))
model.load_state_dict(torch.load(modelc))
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)

for x in model.ensemble.models:
    x.output.bias.data.fill_(0) # hack to match behaviour of cut-down implementation

results = defaultdict(list)
model.eval()

files = fetch_files_with_timestamps()

with torch.inference_mode():
    for bstart in tqdm(range(0, len(files), batch_size)):
        batch = files[bstart:bstart + batch_size]
        timestamps = [ t1 for f1, e1, t1 in batch ]
        embs = torch.stack([ torch.Tensor(e1).to(config.dtype) for f1, e1, t1 in batch ])
        inputs = embs.unsqueeze(0).expand((config.n_ensemble, len(batch), config.d_emb)).to(device)
        scores = model.ensemble(inputs).mean(dim=0).cpu().numpy()
        for sr in scores:
            for i, s in enumerate(sr):
                results[i].append(s)
        # add an extra timestamp channel
        results[config.output_channels].extend(timestamps)

cdfs = []
# we want to encode scores in one byte, and 255/0xFF is reserved for "greater than maximum bucket"
cdf_bins = 255
for i, s in results.items():
    quantiles = numpy.linspace(0, 1, cdf_bins)
    cdf = numpy.quantile(numpy.array(s), quantiles)
    print(cdf)
    cdfs.append(cdf.tolist())

with open("cdfs.msgpack", "wb") as f:
    msgpack.pack(cdfs, f)
