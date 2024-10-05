import torch.nn
import torch.nn.functional as F
import torch
import numpy
import json
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy as np
import base64
import asyncio
import aiohttp
import aioitertools

from model import SAEConfig, SAE
from shared import train_split, loaded_arrays, ckpt_path

model_path, _ = ckpt_path(12991)

model = SAE(SAEConfig(
    d_emb=1152,
    d_hidden=65536,
    top_k=128,
    device="cuda",
    dtype=torch.float32,
    up_proj_bias=False
))

state_dict = torch.load(model_path)

batch_size = 1024
retrieve_batch_size = 512

with torch.inference_mode():
    model.load_state_dict(state_dict)
    model.eval()

    validation_set = loaded_arrays["embedding"][int(len(loaded_arrays["embedding"]) * train_split):]

    for batch_start in tqdm(range(0, len(validation_set), batch_size)):
        batch = numpy.stack([ numpy.frombuffer(embedding.as_py(), dtype=numpy.float16) for embedding in validation_set[batch_start:batch_start + 1024] ])
        batch = torch.Tensor(batch).to("cuda")
        reconstructions = model(batch).float()

    feature_frequencies = model.reset_counters()
    features = model.up_proj.weight.cpu().numpy()

meme_search_backend = "http://localhost:1707/"
memes_url = "https://i.osmarks.net/memes-or-something/"
meme_search_url = "https://mse.osmarks.net/?e="

def emb_url(embedding):
    return meme_search_url + base64.urlsafe_b64encode(embedding.astype(np.float16).tobytes()).decode("utf-8")

async def get_exemplars():
    async with aiohttp.ClientSession():
        for base in tqdm(range(0, len(features), retrieve_batch_size)):
            chunk = features[base:base + retrieve_batch_size]
            with open(f"feature_dumps/features{base}.html", "w") as f:
                f.write("""<!DOCTYPE html>
        <title>Embeddings SAE Features</title>
        <style>
        div img {
            width: 20%
        }
        </style>
        <body><h1>Embeddings SAE Features</h1>""")
                
                async def lookup(embedding):
                    async with aiohttp.request("POST", meme_search_backend, json={
                        "terms": [{ "embedding": list(float(x) for x in embedding) }], # sorry
                        "k": 10
                    }) as res:
                        return (await res.json())["matches"]

                exemplars = await aioitertools.asyncio.gather(*[ lookup(feature) for feature in chunk ])
                negative_exemplars = await aioitertools.asyncio.gather(*[ lookup(-feature) for feature in chunk ])

                for offset, (feature, frequency) in sorted(enumerate(zip(chunk, feature_frequencies[base:])), key=lambda x: -x[1][1]):
                    f.write(f"""
    <h2>Feature {offset + base}</h2>
    <h3>Frequency {frequency / len(validation_set)}</h3>
    <div>
    <h4><a href="{emb_url(feature)}">Max</a></h4>
    """)
                    for match in exemplars[offset]:
                        f.write(f'<img loading="lazy" src="{memes_url+match[1]}">')
                    f.write(f'<h4><a href="{emb_url(-feature)}">Min</a></h4>')
                    for match in negative_exemplars[offset]:
                        f.write(f'<img loading="lazy" src="{memes_url+match[1]}">')
                    f.write("</div>")

asyncio.run(get_exemplars())