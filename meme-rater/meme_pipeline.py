import subprocess
import torch
from tqdm import tqdm
import json
from pathlib import Path
import os
import asyncio
import aiohttp
import time

import shared
from model import Config, BradleyTerry

meme_search_backend = "http://localhost:1707/"
score_threshold = 1.7264162302017212

shared.db.executescript("""
CREATE TABLE IF NOT EXISTS last_crawl (time INTEGER);
CREATE TABLE IF NOT EXISTS library_queue (
    filename TEXT PRIMARY KEY,
    score REAL NOT NULL
);
""")
shared.db.commit()
csr = shared.db.execute("SELECT MAX(time) FROM last_crawl")
row = csr.fetchone()
last_crawl = row[0] or 0
csr.close()

with open("rater_mse_config.json", "r") as f:
    mse_config = json.load(f)
    basedir = Path(mse_config["files"])

print("crawling...")
crawl_start = time.time()
subprocess.run(["python", "crawler.py", str(last_crawl)]).check_returncode()
print("indexing...")
subprocess.run(["python", "../mse.py", "rater_mse_config.json"]).check_returncode()
print("evaluating...")

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

print(sorted(ratings.values())[round(len(ratings) * 0.95)])
print(f"{len(ratings)} memes in {crawl_start - last_crawl} seconds ({len(ratings) / (crawl_start - last_crawl) * 1e3}mHz)")

files = dict(files)

async def run_inserts():
    async with aiohttp.ClientSession():
        async def duplicate_exists(embedding):
            async with aiohttp.request("POST", meme_search_backend, json={
                "embeddings": [ list(float(x) for x in embedding) ], # sorry
                "top_k": 1
            }) as res:
                result = await res.json()
                closest = result[0]["score"]
                return closest > 0.99 # arbitrary threshold, TODO

        for filename, rating in ratings.items():
            if rating > score_threshold and not await duplicate_exists(files[filename]):
                shared.db.execute("INSERT OR REPLACE INTO library_queue VALUES (?, ?)", (filename, rating))
            else:
                os.unlink(basedir / filename)
    shared.db.execute("INSERT INTO last_crawl VALUES (?)", (crawl_start,))
    shared.db.commit()

asyncio.run(run_inserts())