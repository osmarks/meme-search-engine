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
from shared import train_split, loaded_arrays, ckpt_path, loaded_arrays_permutation

model_path, _ = ckpt_path(111885)

@dataclass
class TrainConfig:
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool

state_dict = torch.load(model_path)
model = SAE(state_dict["config"].model)

batch_size = 1024
retrieve_batch_size = 512

with torch.inference_mode():
    model.load_state_dict(state_dict["model"])
    model.eval()

    print("loading val set")
    train_set_size = int(len(loaded_arrays_permutation) * train_split)
    val_set_size = len(loaded_arrays_permutation) - train_set_size
    print("sliced. executing.")

    for batch_start in tqdm(range(train_set_size, train_set_size+val_set_size, batch_size)):
        batch = numpy.stack([ numpy.frombuffer(embedding, dtype=numpy.float16) for embedding in loaded_arrays[loaded_arrays_permutation[batch_start:batch_start + batch_size]] ])
        batch = torch.Tensor(batch).to("cuda")
        reconstructions = model(batch).float()

    feature_frequencies = model.reset_counters()
    features = model.down_proj.weight.cpu().numpy()

meme_search_backend = "http://localhost:5601/"
memes_url = ""
meme_search_url = "https://nooscope.osmarks.net/?page=advanced&e="

def emb_url(embedding):
    return meme_search_url + base64.urlsafe_b64encode(embedding.astype(np.float16).tobytes()).decode("utf-8")

async def get_exemplars():
    async with aiohttp.ClientSession():
        for base in tqdm(range(0, features.shape[1], retrieve_batch_size)):
            chunk = features[:, base:base + retrieve_batch_size].T
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
                        return (await res.json())["matches"][:10]

                exemplars = await aioitertools.asyncio.gather(*[ lookup(feature) for feature in chunk ])
                negative_exemplars = await aioitertools.asyncio.gather(*[ lookup(-feature) for feature in chunk ])

                for offset, (feature, frequency) in sorted(enumerate(zip(chunk, feature_frequencies[base:])), key=lambda x: -x[1][1]):
                    f.write(f"""
    <h2>Feature {offset + base}</h2>
    <h3>Frequency {frequency / val_set_size}</h3>
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
