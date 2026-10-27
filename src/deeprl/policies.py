from __future__ import annotations

import torch.nn as nn
from torch.distributions import Categorical

from deeprl.buffers import TensorData


class CategoricalPolicy(nn.Module):
    """Categorical policy backed by a caller-supplied logits network."""

    def __init__(self, logits_network: nn.Module) -> None:
        super().__init__()
        self.logits_network = logits_network

    def forward(self, observations: TensorData) -> Categorical:
        logits = self.logits_network(observations)  # (B, num_actions)
        return Categorical(logits=logits)
