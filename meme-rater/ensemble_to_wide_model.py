import torch.nn
import torch.nn.functional as F
import torch
import sqlite3
import random
import numpy
import json
import msgpack
import sys
from safetensors.torch import save_file

from model import Config, BradleyTerry
import shared

device = "cpu"

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

torch.random.manual_seed(1)

for x in model.ensemble.models:
    x.output.bias.data.fill_(0)

out_layers = []
out_bias = []

with torch.inference_mode():
    # TODO: I don't think this actually works for more than 1 hidden layer
    for layer in range(config.n_hidden):
        big_layer = torch.zeros(config.n_ensemble * config.d_emb, config.d_emb)
        big_bias = torch.zeros(config.n_ensemble * config.d_emb)
        for i in range(config.n_ensemble):
            big_layer[i*config.d_emb:(i+1)*config.d_emb] = model.ensemble.models[i].hidden[layer].weight.data.clone()
            big_bias[i*config.d_emb:(i+1)*config.d_emb] = model.ensemble.models[i].hidden[layer].bias.data.clone()
        out_layers.append(big_layer)
        out_bias.append(big_bias)
    # we do not need to preserve the bias on the downprojection as the win probability calculation is shift-invariant
    downprojection = torch.zeros(config.output_channels, config.n_ensemble * config.d_emb)
    for i in range(config.n_ensemble):
        downprojection[:, i*config.d_emb:(i+1)*config.d_emb] = model.ensemble.models[i].output.weight.data.clone()

    for i in range(10):
        input = torch.randn(4, config.d_emb)
        ground_truth_result = model.ensemble(input.unsqueeze(0).expand((config.n_ensemble, *input.shape))).mean(dim=0).T
        r_result = input
        for (layer, bias) in zip(out_layers, out_bias):
            r_result = torch.matmul(layer, r_result.T) + bias.unsqueeze(-1).expand(config.n_ensemble * config.d_emb, input.shape[0])
            print(r_result.shape, bias.shape)
            r_result = F.silu(r_result)
        r_result = torch.matmul(downprojection, r_result) / config.n_ensemble
        error = torch.mean(r_result - ground_truth_result)
        print(error)
        assert error.detach().cpu().numpy() < 1e-4

        print("test vector:")
        #print(input.flatten().tolist())
        print("ground truth result:")
        print(ground_truth_result.shape)
        print(ground_truth_result.T.flatten().tolist())

    save_file({
        "up_proj": out_layers[0],
        "bias": out_bias[0],
        "down_proj": downprojection
    }, "model.safetensors")
