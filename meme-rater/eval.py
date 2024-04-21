import torch.nn
import torch.nn.functional as F
import torch
import sqlite3
import random
import numpy
import json
import time
from tqdm import tqdm
import torch

from model import Config, BradleyTerry
import shared

batch_size = 128
device = "cuda"

config = Config(
    d_emb=1152,
    n_hidden=1,
    n_ensemble=16,
    device=device,
    dtype=torch.float32,
    dropout=0.1
)
model = BradleyTerry(config).float()
modelc, _ = shared.checkpoint_for(1500)
model.load_state_dict(torch.load(modelc))
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)

files = shared.fetch_all_files()
ratings = {}

model.eval()
with torch.inference_mode():
    for bstart in tqdm(range(0, len(files), batch_size)):
        batch = files[bstart:bstart + batch_size]
        filenames = [ filename for filename, embedding in batch ]
        embs = torch.stack([ torch.Tensor(embedding) for filename, embedding in batch ])
        inputs = embs.unsqueeze(0).expand((config.n_ensemble, len(batch), config.d_emb)).to(device)
        scores = model.ensemble(inputs).float()
        mscores = torch.median(scores, dim=0).values
        for filename, mscore in zip(filenames, mscores):
            ratings[filename] = float(mscore)

ratings = sorted(ratings.items(), key=lambda x: x[1])

def percentile(p, n):
    base = round(p * len(ratings))
    return ratings[base:base + n]

N = 25
def render_memeset(p):
    filenames = percentile(p, N)
    return f"""
<div>
    <details><summary>Reveal Memeset</summary>{p}</details>
    {''.join(f'<div><img src="{"images/" + f}" width="30%"><br><input type=checkbox id="{"col-" + str(p) + "-" + str(i)}"></div>' for i, (f, s) in enumerate(filenames))}
</div>
"""

buf = """<!DOCTYPE html>"""
probs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.98, 0.99]
random.shuffle(probs)
for p in probs:
#for p in [0.3]:
    buf += render_memeset(p)

buf += """
<script>
    const computeCounts = () => {
        const counts = {}
        for (const x of document.querySelectorAll("input[type=checkbox]")) {
            const [_, percentile, index] = x.getAttribute("id").split("-")
            counts[percentile] ??= 0
            if (x.checked) counts[percentile] += 1
        }
        console.log(counts)
    }
</script>
"""

with open("eval.html", "w") as f:
    f.write(buf)