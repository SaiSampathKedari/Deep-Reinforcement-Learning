from __future__ import annotations

import torch
import torch.nn as nn

from deeprl.buffers import TensorData


class ValueFunction(nn.Module):
    """State-value function backed by a caller-supplied scalar-output network."""

    def __init__(self, value_network: nn.Module) -> None:
        super().__init__()
        self.value_network = value_network

    def forward(self, observations: TensorData) -> torch.Tensor:
        values = self.value_network(observations)
        expected_shape = (observations.shape[0], 1)

        if values.shape != expected_shape:
            raise ValueError(
                f"value_network must return shape {expected_shape}, "
                f"got {tuple(values.shape)}"
            )

        return values.squeeze(-1)  # (B,), preserving the batch dimension.
