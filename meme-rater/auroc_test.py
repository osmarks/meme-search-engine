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
random.shuffle(ratings)

N = 150

buf = f"""<!DOCTYPE html>
<div>
{''.join(f'<div><img src="{"images/" + f}" width="30%"><br><input type=checkbox data-score="{s}"></div>' for i, (f, s) in enumerate(ratings[:N]))}
</div>
<script>
    const dump = () => {{
        const data = []
        for (const x of document.querySelectorAll("input[type=checkbox]")) {{
            data.push([parseFloat(x.getAttribute("data-score")), x.checked])
        }}
        console.log(JSON.stringify(data))
    }}
</script>
"""

with open("eval.html", "w") as f:
    f.write(buf)