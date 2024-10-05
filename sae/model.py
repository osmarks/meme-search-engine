import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial

@dataclass
class SAEConfig:
    d_emb: int
    d_hidden: int
    top_k: int
    up_proj_bias: bool
    device: str
    dtype: torch.dtype

class SAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.d_emb, config.d_hidden, dtype=config.dtype, device=config.device, bias=config.up_proj_bias)
        self.down_proj = nn.Linear(config.d_hidden, config.d_emb, dtype=config.dtype, device=config.device)
        self.down_proj.weight = nn.Parameter(self.up_proj.weight.T.clone())
        self.feature_activation_counter = torch.zeros(config.d_hidden, dtype=torch.int32, device=config.device)
        self.reset_counters()

    def reset_counters(self):
        old = self.feature_activation_counter.detach().cpu().numpy()
        torch.zero_(self.feature_activation_counter)
        return old

    def forward(self, embs):
        x = self.up_proj(embs)
        x = F.relu(x)
        topk = torch.kthvalue(x, k=(self.config.d_hidden - self.config.top_k), dim=-1)
        thresholds = topk.values.unsqueeze(-1).expand_as(x)
        zero = torch.zeros_like(x)
        # If multiple values are the same, we don't actually pick exactly k values. This can happen quite easily if for some reason a lot of values are negative and thus get ReLUed to 0.
        # This should not really happen but it does.
        # This uses greater than rather than greater than or equal to work around this. We compensate for this by setting k off by one in the kthvalue call.
        mask = x > thresholds
        x = torch.where(mask, x, zero)
        self.feature_activation_counter += mask.sum(0)
        return self.down_proj(x)