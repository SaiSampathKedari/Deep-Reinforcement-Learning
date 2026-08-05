from __future__ import annotations

import torch
import torch.nn as nn

class ValueFunction(nn.Module):
    def __init__(self, value_network: nn.Module) -> None:
        super().__init__()
        self.value_network = value_network
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_network(observations).squeeze(-1)