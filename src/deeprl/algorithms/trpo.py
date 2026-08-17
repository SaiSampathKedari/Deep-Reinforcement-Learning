"""Trust Region Policy Optimization learner and training entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributions import Distribution
from gymnasium.vector import VectorEnv

from deeprl.algorithms.npg import (
    NPGConfig,
    NPGLearner,
    _set_flat_parameters,
)
from deeprl.algorithms.on_policy import train_on_policy
from deeprl.buffers import RolloutBatch
from deeprl.logger import Logger, MetricHistory


if TYPE_CHECKING:
    from deeprl.evaluate import Evaluator
    from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TRPOConfig(NPGConfig):
    """Optimization hyperparameters specific to TRPO."""

    backtrack_coeff :   float = 0.8
    max_backtracks  :   int = 10


class TRPOLearner(NPGLearner):
    """NPG with backtracking that enforces the sampled KL constraint."""

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   TRPOConfig,
    ) -> None:
        """Store the networks and refine the inherited configuration type."""
        super().__init__(policy, value_fn, cfg)
        self.cfg: TRPOConfig = cfg

    @torch.no_grad()
    def _apply_policy_step(
        self,
        *,
        batch                       :   RolloutBatch,
        advantages                  :   torch.Tensor,
        reference_distribution      :   Distribution,
        base_policy_parameters      :   torch.Tensor,
        full_step                   :   torch.Tensor,
        surrogate_before_step       :   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply the first improving candidate within the sampled KL bound."""

        step_fraction = 1.0

        for _ in range(self.cfg.max_backtracks):

            candidate_policy_parameters = (
                base_policy_parameters + step_fraction * full_step
            )

            _set_flat_parameters(
                self.policy_parameters,
                candidate_policy_parameters,
            )

            surrogate_after_step = self._surrogate_objective(
                batch,
                advantages,
            )

            mean_kl_after_step = self._mean_kl_divergence(
                batch.observations,
                reference_distribution,
            )

            accept_step = (
                torch.isfinite(surrogate_after_step)
                & torch.isfinite(mean_kl_after_step)
                & (surrogate_after_step > surrogate_before_step)
                & (mean_kl_after_step <= self.cfg.max_kl)
            ).item()

            if accept_step:
                return (
                    surrogate_after_step,
                    mean_kl_after_step,
                    step_fraction,
                )

            step_fraction *= self.cfg.backtrack_coeff

        # Reject every candidate and restore the pre-update policy.
        _set_flat_parameters(
            self.policy_parameters,
            base_policy_parameters,
        )

        return (
            surrogate_before_step,
            torch.zeros_like(surrogate_before_step),
            0.0,
        )


def trpo(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    cfg             :   TRPOConfig,
    writer          :   SummaryWriter | None = None,
    logger          :   Logger | None = None,
    evaluator       :   Evaluator | None = None,
) -> MetricHistory:
    """Construct a TRPO learner and run the shared on-policy training loop."""
    learner = TRPOLearner(policy, value_fn, cfg)

    return train_on_policy(
        envs=envs,
        learner=learner,
        writer=writer,
        logger=logger,
        evaluator=evaluator,
    )
