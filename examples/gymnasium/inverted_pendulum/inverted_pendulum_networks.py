"""Network architectures for InvertedPendulum."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Map observations to Gaussian action means."""

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # Small output weights initialize action means near zero.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                gain = 0.01 if layer is self.network[-1] else 2**0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Match the network dtype; MuJoCo observations arrive as float64.
        observations = observations.to(dtype=next(self.parameters()).dtype)
        return self.network(observations)  # (B, action_dim)


class ValueNetwork(nn.Module):
    """Map observations to scalar state-value predictions."""

    def __init__(self, obs_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Orthogonal weights with unit gain at the value output.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                gain = 1.0 if layer is self.network[-1] else 2**0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Match the network dtype; MuJoCo observations arrive as float64.
        observations = observations.to(dtype=next(self.parameters()).dtype)
        return self.network(observations)  # (B, 1)
