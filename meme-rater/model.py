import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial
import math

@dataclass
class Config:
    d_emb: int
    n_hidden: int
    n_ensemble: int
    device: str
    dtype: torch.dtype
    dropout: float

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = nn.ModuleList([ nn.Linear(config.d_emb, config.d_emb, dtype=config.dtype, device=config.device) for _ in range(config.n_hidden) ])
        self.dropout = nn.ModuleList([ nn.Dropout(p=config.dropout) for _ in range(config.n_hidden) ])
        self.output = nn.Linear(config.d_emb, 1, dtype=config.dtype, device=config.device)

    def forward(self, embs):
        x = embs
        for (layer, dropout) in zip(self.hidden, self.dropout):
            x = F.silu(layer(dropout(x)))
        return self.output(x)

class Ensemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.models = nn.ModuleList([ Model(config) for i in range(config.n_ensemble) ])

    # model batch
    def forward(self, embs):
        xs = torch.stack([ x(embs[i]) for i, x in enumerate(self.models) ]) # model batch output_dim=1
        return xs.squeeze(-1)

class BradleyTerry(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ensemble = Ensemble(config)

    def forward(self, embs): # model batch input=2 d_emb
        scores1 = self.ensemble(embs[:, :, 0]).float() # model batch
        scores2 = self.ensemble(embs[:, :, 1]).float()
        # win probabilities
        #print(scores1, scores2)
        probs = torch.sigmoid(scores1 - scores2) # model batch
        #print(probs)
        return probs