import torch.nn
import torch.nn.functional as F
import torch
import sqlite3
import random
import numpy
import json
import time
from tqdm import tqdm
import math
from dataclasses import dataclass, asdict
import sys

from model import Config as ModelConfig, BradleyTerry
import shared

trains, validations = shared.fetch_ratings(sys.argv[1:])
for train, validation in zip(trains, validations):
    print(len(train), len(validation))

device = "cuda"

@dataclass
class TrainConfig:
    model: ModelConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool
    data_grouped_by_iter: bool

config = TrainConfig(
    model=ModelConfig(
        d_emb=1152,
        n_hidden=1,
        n_ensemble=16,
        device=device,
        dtype=torch.float32,
        dropout=0.0,
        output_channels=3
    ),
    lr=3e-4,
    weight_decay=0.0,
    batch_size=1,
    epochs=5,
    compile=False,
    data_grouped_by_iter=False
)

def exprange(min, max, n):
    lmin, lmax = math.log(min), math.log(max)
    step = (lmax - lmin) / (n - 1)
    return (math.exp(lmin + step * i) for i in range(n))

model = BradleyTerry(config.model)
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def train_step(model, batch, real):
    optimizer.zero_grad()
    # model batch
    win_probabilities = model(batch).float()
    loss = F.binary_cross_entropy(win_probabilities, real)
    loss.backward()
    optimizer.step()
    loss = loss.detach().cpu().item()
    return loss

if config.compile:
    print("compiling...")
    train_step = torch.compile(train_step)

def batch_from_inputs(inputs: list[list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]]):
    batch_input = torch.stack([
        torch.stack([ torch.stack((torch.Tensor(emb1).to(config.model.dtype), torch.Tensor(emb2).to(config.model.dtype))) for emb1, emb2, rating in input ])
        for input in inputs
    ]).to(device)
    target = torch.stack([ torch.Tensor(numpy.array([ rating for emb1, emb2, rating in input ])).to(config.model.dtype) for input in inputs ]).to(device)
    return batch_input, target

def evaluate(steps):
    print("evaluating...")
    model.eval()
    results = {"step": steps, "time": time.time(), "val_loss": {}}
    for vset, validation in enumerate(validations):
        with torch.no_grad():
            batch_input, target = batch_from_inputs([ validation[:128] for _ in range(config.model.n_ensemble) ])
            result = model(batch_input).float()
            val_loss = F.binary_cross_entropy(result, target).detach().cpu().item()
            model.train()
            results["val_loss"][vset] = val_loss
    log.write(json.dumps(results) + "\n")

def save_ckpt(log, steps):
    print("saving...")
    modelc, optimc = shared.checkpoint_for(steps)
    torch.save(optimizer.state_dict(), optimc)
    torch.save(model.state_dict(), modelc)

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.dtype):
            return str(o)
        else: return super().default(o)

logfile = f"logs/log-{time.time()}.jsonl"
with open(logfile, "w") as log:
    steps = 0
    log.write(JSONEncoder().encode(asdict(config)) + "\n")
    for epoch in range(config.epochs):
        for train in (trains if config.data_grouped_by_iter else [[ sample for trainss in trains for sample in trainss ]]):
            data_orders = shared.generate_random_permutations(train, config.model.n_ensemble)
            for bstart in range(0, len(train), config.batch_size):
                batch_input, target = batch_from_inputs([ order[bstart:bstart + config.batch_size] for order in data_orders ])
                loss = train_step(model, batch_input, target)
                print(steps, loss)
                log.write(json.dumps({"loss": loss, "step": steps, "time": time.time()}) + "\n")
                if steps % 10 == 0:
                    if steps % 50 == 0: save_ckpt(log, steps)
                    loss = evaluate(steps)
                    #print(loss)
                    #best = min(loss, best)
                steps += 1

        save_ckpt(log, steps)

print(logfile)
