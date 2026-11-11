"""Actor and critic for Isaac Lab's concatenated G1 policy observations."""

import torch
from tensordict import TensorDictBase
from torch import nn

from deeprl.policies import DiagonalGaussianPolicy


class PolicyNetwork(nn.Module):
    """Map the ``policy`` observation (B, obs_dim) to action means (B, A)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, int, int],
    ) -> None:
        super().__init__()
        hidden_1, hidden_2, hidden_3 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.ELU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ELU(),
            nn.Linear(hidden_2, hidden_3),
            nn.ELU(),
            nn.Linear(hidden_3, action_dim),
        )

    def forward(self, observations: TensorDictBase) -> torch.Tensor:
        return self.network(observations["policy"])


class ValueNetwork(nn.Module):
    """Map the same G1 observation (B, obs_dim) to a value (B, 1)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, int, int],
    ) -> None:
        super().__init__()
        hidden_1, hidden_2, hidden_3 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_1),
            nn.ELU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ELU(),
            nn.Linear(hidden_2, hidden_3),
            nn.ELU(),
            nn.Linear(hidden_3, 1),
        )

    def forward(self, observations: TensorDictBase) -> torch.Tensor:
        return self.network(observations["policy"])


def build_policy(config: dict) -> DiagonalGaussianPolicy:
    """Rebuild the actor from the model settings saved with a policy."""
    bounds = config["log_std_bounds"]
    return DiagonalGaussianPolicy(
        PolicyNetwork(
            config["obs_dim"],
            config["action_dim"],
            tuple(config["hidden_sizes"]),
        ),
        action_dim=config["action_dim"],
        log_std_init=config["log_std_init"],
        log_std_bounds=None if bounds is None else tuple(bounds),
    )
