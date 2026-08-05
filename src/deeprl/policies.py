from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self, logits_network: nn.Module) -> None:
        super().__init__()
        self.logits_network = logits_network
    
    def forward(self, observations: torch.Tensor) -> Categorical:
        scores = self.logits_network(observations)
        return Categorical(logits=scores)
        
