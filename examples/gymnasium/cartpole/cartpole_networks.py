"""Policy and state-value architectures for CartPole."""

import torch
from torch import nn

from deeprl.policies import CategoricalPolicy


def build_policy(config: dict) -> CategoricalPolicy:
    """Reconstruct the actor from an exported policy's constructor arguments."""
    return CategoricalPolicy(
        PolicyNetwork(
            config["obs_dim"],
            config["num_actions"],
            tuple(config["hidden_sizes"]),
        )
    )


class PolicyNetwork(nn.Module):
    """Map (B, obs_dim) observations to (B, num_actions) logits."""

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
    """Map (B, obs_dim) observations to (B, 1) state values."""

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
