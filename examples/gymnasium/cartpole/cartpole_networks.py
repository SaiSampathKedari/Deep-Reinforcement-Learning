"""Policy and state-value architectures for CartPole."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Map observations to unnormalized action logits."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.Tanh(),
            nn.Linear(hidden_1, hidden_2),
            nn.Tanh(),
            nn.Linear(hidden_2, num_actions),
        )

        # Small output weights produce near-uniform initial action probabilities.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                gain = 0.01 if layer is self.network[-1] else 2**0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)  # (B, num_actions)


class ValueNetwork(nn.Module):
    """Map observations to scalar state-value predictions."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, int] = (64, 64),
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

        # Orthogonal weights with unit gain at the value output.
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                gain = 1.0 if layer is self.network[-1] else 2**0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)  # (B, 1)
