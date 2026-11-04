"""Policy and state-value architectures for Humanoid."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Map observations to Gaussian action means."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, action_dim),
        )

        # Match the inspected reference implementation, including its output head.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Match the network dtype; MuJoCo observations arrive as float64.
        observations = observations.to(dtype=next(self.parameters()).dtype)
        return self.network(observations)  # (B, action_dim)


class ValueNetwork(nn.Module):
    """Map observations to scalar state-value predictions."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, 1),
        )

        # The reference applies Tanh gain to every linear layer, including V(s).
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Match the network dtype; MuJoCo observations arrive as float64.
        observations = observations.to(dtype=next(self.parameters()).dtype)
        return self.network(observations)  # (B, 1)
