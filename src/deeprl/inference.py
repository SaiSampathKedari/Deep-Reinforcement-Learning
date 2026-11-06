"""Compose explicit input processing, model decisions and environment commands."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributions import Distribution

from deeprl.data_spec import TensorData
from deeprl.processors import ActionProcessor, ObservationProcessor


def distribution_action(distribution: Distribution, deterministic: bool) -> TensorData:
    """Sample a distribution, or use its mode/mean for deterministic inference."""
    if torch.is_tensor(distribution):
        raise TypeError(
            "tensor-valued actors and Q-functions require an explicit action selector"
        )
    if not deterministic:
        return distribution.sample()
    try:
        return distribution.mode
    except NotImplementedError:
        return distribution.mean


def greedy_action(q_values: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """Choose the largest Q-value; exploration belongs to the behaviour policy."""
    return q_values.argmax(dim=-1)


def deterministic_action(actions: TensorData, deterministic: bool) -> TensorData:
    """Use an actor's action output directly, without adding exploration noise."""
    return actions


class InferencePolicy(nn.Module):
    """A model plus explicit, shared preprocessing and action selection.

    ``forward`` returns model outputs for likelihoods and differentiable losses.
    ``act`` selects an action and returns the environment command without gradients.
    Neither method fits statistics. Tensor-valued models require an explicit
    selector so Q-values are not mistaken for deterministic actions.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        observation_processor: ObservationProcessor | None = None,
        action_processor: ActionProcessor | None = None,
        action_selector: Callable[[Any, bool], TensorData] | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be boolean")
        self.model = model
        self.observation_processor = observation_processor
        self.action_processor = action_processor
        self.action_selector = (
            distribution_action if action_selector is None else action_selector
        )
        self.deterministic = deterministic

    def forward(self, observations: TensorData) -> Any:
        """Evaluate the model on actor inputs using the current processor state."""
        if self.observation_processor is not None:
            observations = self.observation_processor(observations, group="actor")
        return self.model(observations)

    @torch.no_grad()
    def act(
        self,
        observations: TensorData,
        deterministic: bool | None = None,
    ) -> TensorData:
        """Return a command using the saved default unless explicitly overridden."""
        deterministic = self.deterministic if deterministic is None else deterministic
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be boolean")
        actions = self.action_selector(self(observations), deterministic)
        # Action conversion affects commands, not the model's action distribution.
        return actions if self.action_processor is None else self.action_processor(actions)
