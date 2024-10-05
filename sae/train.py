import torch.nn
import torch.nn.functional as F
import torch
import numpy
import json
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict

from model import SAEConfig, SAE
from shared import train_split, loaded_arrays, ckpt_path

device = "cuda"

@dataclass
class TrainConfig:
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool

config = TrainConfig(
    model=SAEConfig(
        d_emb=1152,
        d_hidden=65536,
        top_k=128,
        device=device,
        dtype=torch.float32,
        up_proj_bias=False
    ),
    lr=3e-4,
    weight_decay=0.0,
    batch_size=64,
    epochs=5,
    compile=True,
)

def exprange(min, max, n):
    lmin, lmax = math.log(min), math.log(max)
    step = (lmax - lmin) / (n - 1)
    return (math.exp(lmin + step * i) for i in range(n))

model = SAE(config.model)
params = sum(p.numel() for p in model.parameters())
print(f"{params/1e6:.1f}M parameters")
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def train_step(model, batch):
    optimizer.zero_grad()
    reconstructions = model(batch).float()
    loss = F.mse_loss(reconstructions, batch)
    loss.backward()
    optimizer.step()
    return loss

if config.compile:
    print("compiling...")
    train_step = torch.compile(train_step)

def save_ckpt(log, steps):
    #print("saving...")
    modelc, optimc = ckpt_path(steps)
    torch.save(optimizer.state_dict(), optimc)
    torch.save({"model": model.state_dict(), "config": config}, modelc)

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
        batch = []
        t = tqdm(range(0, int(len(loaded_arrays) * train_split), config.batch_size))
        for batch_start in t:
            batch = numpy.stack([ numpy.frombuffer(embedding.as_py(), dtype=numpy.float16) for embedding in loaded_arrays["embedding"][batch_start:batch_start + config.batch_size] ])

            if len(batch) == config.batch_size:
                batch = torch.Tensor(batch).to(device)
                loss = train_step(model, batch)
                loss = loss.detach().cpu().item()
                t.set_description_str(f"loss: {loss:.6f} epoch: {epoch}")
                log.write(json.dumps({"loss": loss, "step": steps, "time": time.time()}) + "\n")
                if steps % 5000 == 0: save_ckpt(log, steps)
                steps += 1

        save_ckpt(log, steps)
        print(model.feature_activation_counter.cpu())
        ctr = model.reset_counters()
        print(ctr)
        numpy.save(f"ckpt/{steps}.counters.npy", ctr)

print(logfile)