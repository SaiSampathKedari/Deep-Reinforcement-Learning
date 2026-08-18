"""Natural policy-gradient learner and training entry point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import Distribution, kl_divergence

from deeprl.algorithms.on_policy import (
    OnPolicyConfig,
    OnPolicyLearner,
    train_on_policy,
)
from deeprl.buffers import RolloutBatch, RolloutBuffer
from deeprl.logger import Logger, MetricHistory
from deeprl.stats import TrainingStats
from deeprl.utils import explained_variance

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from deeprl.evaluate import Evaluator


def _flat_grad(
    output          :   torch.Tensor,
    parameters      :   tuple[nn.Parameter, ...],
    *,
    create_graph    :   bool = False,
    retain_graph    :   bool | None = None,
) -> torch.Tensor:
    """Differentiate output and concatenate gradients in parameter order."""
    gradients = torch.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


@torch.no_grad()
def _flat_parameters(parameters: tuple[nn.Parameter, ...]) -> torch.Tensor:
    """Flatten and concatenate parameter values in their fixed order."""
    return torch.cat([parameter.reshape(-1) for parameter in parameters])

@torch.no_grad()
def _set_flat_parameters(
    parameters      :   tuple[nn.Parameter, ...],
    flat_parameters :   torch.Tensor,
) -> None:
    """Split a flat vector by parameter size and copy it back in order."""
    offset = 0

    for parameter in parameters:
        numel = parameter.numel()
        parameter.copy_(
            flat_parameters[offset : offset + numel].view_as(parameter)
        )
        offset += numel


def _conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Approximately solve matrix * solution = b from matrix-vector products."""

    # Starting from zero makes the initial residual and search direction b.
    solution = torch.zeros_like(b)
    residual = b.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)

    for _ in range(max_iterations):
        if residual_squared <= tolerance:
            break

        # Move along the current direction, then update the remaining error.
        matrix_direction = matrix_vector_product(direction)
        step_size = residual_squared / torch.dot(
            direction,
            matrix_direction,
        )

        solution = solution + step_size * direction
        residual = residual - step_size * matrix_direction

        # Combine the new residual with the previous conjugate direction.
        new_residual_squared = torch.dot(residual, residual)
        conjugate_coefficient = new_residual_squared / residual_squared

        direction = residual + conjugate_coefficient * direction
        residual_squared = new_residual_squared

    return solution


@dataclass(frozen=True)
class NPGConfig(OnPolicyConfig):
    """Optimization hyperparameters specific to synchronous NPG."""

    # Optimization
    lr_critic           :   float = 1e-3
    max_kl              :   float = 1e-2
    cg_damping          :   float = 1e-2
    cg_iters            :   int = 10
    cg_residual_tol     :   float = 1e-10
    max_grad_norm       :   float = 0.5
    normalize_advantage :   bool = True
    batch_size          :   int | None = None


class NPGLearner(OnPolicyLearner):
    """Natural policy-gradient and critic updates."""

    def __init__(
        self,
        policy      :   nn.Module,
        value_fn    :   nn.Module,
        cfg         :   NPGConfig,
    ) -> None:
        """Store the networks and initialize NPG-specific optimization state."""
        super().__init__(policy, value_fn, cfg)
        self.cfg: NPGConfig = cfg

        # Preserve one fixed parameter order for flattened gradients and steps.
        self.policy_parameters: tuple[nn.Parameter, ...] = tuple(
            self.policy.parameters()
        )

        # NPG computes the policy step directly; only the critic uses an optimizer.
        self.optim_critic = torch.optim.SGD(
            self.value_fn.parameters(),
            lr=self.cfg.lr_critic,
        )

    def _surrogate_objective(
        self,
        batch       :   RolloutBatch,
        advantages  :   torch.Tensor,
    ) -> torch.Tensor:
        """Return mean[importance_ratio * advantage] for one batch.

        The ratio compares the current policy with the rollout policy.
        """
        # Re-evaluate the rollout actions under the current policy.
        action_distribution : Distribution = self.policy(batch.observations)
        log_probs : torch.Tensor = action_distribution.log_prob(batch.actions)

        # importance_ratio = current_action_probability / rollout_probability
        importance_ratio = torch.exp(
            log_probs - batch.old_log_probs
        )

        return (importance_ratio * advantages).mean()

    def _mean_kl_divergence(
        self,
        observations           :   torch.Tensor,
        reference_distribution :   Distribution,
    ) -> torch.Tensor:
        """Return mean[KL(reference_policy || current_policy)]."""
        current_distribution: Distribution = self.policy(observations)

        return kl_divergence(
            reference_distribution,
            current_distribution,
        ).mean()

    def _fisher_vector_product(
        self,
        flat_kl_gradient :   torch.Tensor,
        vector           :   torch.Tensor,
    ) -> torch.Tensor:
        """Return Fisher * vector plus the conjugate-gradient damping term."""

        # Differentiating this dot product produces the KL Hessian-vector product.
        kl_gradient_vector_product = torch.dot(
            flat_kl_gradient,
            vector,
        )

        fisher_vector_product = _flat_grad(
            kl_gradient_vector_product,
            self.policy_parameters,
            retain_graph=True,
        )

        return fisher_vector_product.detach() + self.cfg.cg_damping * vector

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
        """Apply the full NPG step and evaluate its surrogate and mean KL."""

        # NPG takes the complete step directly. TRPO's override uses
        # surrogate_before_step when testing smaller candidate steps.
        _set_flat_parameters(
            self.policy_parameters,
            base_policy_parameters + full_step,
        )

        # These finite-step measurements are diagnostics; they do not alter NPG.
        surrogate_after_step = self._surrogate_objective(
            batch,
            advantages,
        )
        mean_kl_after_step = self._mean_kl_divergence(
            batch.observations,
            reference_distribution,
        )

        return surrogate_after_step, mean_kl_after_step, 1.0

    def update(
        self,
        buffer          :   RolloutBuffer,
        advantages      :   torch.Tensor,
        value_targets   :   torch.Tensor,
    ) -> TrainingStats:
        """Fit the critic and take one natural policy step per yielded batch."""

        # ------------------------------------------------------------------
        # 0. Setup
        # ------------------------------------------------------------------

        # Keep policy behavior consistent with rollout collection while enabling
        # training behavior for the value function. eval() does not disable gradients.
        self.policy.eval()
        self.value_fn.train()

        # GAE produced fixed, gradient-free advantages. Normalize them once over
        # the complete rollout so their scale is shared by every minibatch.
        if self.cfg.normalize_advantage and advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        # Accumulate sample-weighted metrics over every natural-gradient step.
        metric_sums: dict[str, float] = {
            "losses/policy_loss": 0.0,
            "losses/value_loss": 0.0,
            "diagnostics/entropy": 0.0,
            "diagnostics/mean_kl": 0.0,
            "diagnostics/surrogate_improvement": 0.0,
            "diagnostics/expected_improvement": 0.0,
            "diagnostics/step_fraction": 0.0,
            "grads/policy_norm": 0.0,
            "grads/value_norm": 0.0,
        }

        num_samples = 0
        gradient_steps = 0

        for batch in buffer.get(
            advantages=advantages,
            value_targets=value_targets,
            batch_size=self.cfg.batch_size,
        ):

            # --------------------------------------------------------------
            # 1. Minibatch policy reference
            # --------------------------------------------------------------

            # Save the current parameters. The minibatch step is measured from
            # this base point, which may differ from the original rollout policy.
            base_policy_parameters = _flat_parameters(
                self.policy_parameters
            )

            # Freeze the policy distribution at the base point. It defines the
            # local KL divergence and Fisher matrix for this minibatch.
            with torch.no_grad():
                reference_distribution: Distribution = self.policy(
                    batch.observations
                )

            # --------------------------------------------------------------
            # 2. Critic update
            # --------------------------------------------------------------

            # value_loss = 0.5 * mean[(value_target - current_value)^2]
            # The targets are fixed, so gradients flow only through current values.
            values : torch.Tensor = self.value_fn(batch.observations)
            value_loss = 0.5 * (batch.value_targets - values).square().mean()

            self.optim_critic.zero_grad(set_to_none=True)
            value_loss.backward()

            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_fn.parameters(),
                max_norm=self.cfg.max_grad_norm,
            )

            self.optim_critic.step()

            # --------------------------------------------------------------
            # 3. Surrogate objective and ordinary policy gradient
            # --------------------------------------------------------------

            # surrogate = mean[importance_ratio * advantage]
            surrogate_objective = self._surrogate_objective(
                batch,
                batch.advantages,
            )
            surrogate_before_step = surrogate_objective.detach()

            # Keep the conventional loss sign for logging. NPG directly uses
            # the ascent gradient of the surrogate objective below.
            policy_loss = -surrogate_objective

            # policy_gradient = gradient(surrogate) at the minibatch base point
            policy_gradient = _flat_grad(
                surrogate_objective,
                self.policy_parameters,
            ).detach()

            # --------------------------------------------------------------
            # 4. Fisher-vector product
            # --------------------------------------------------------------

            # At the base point, mean KL and its first derivative are zero, but
            # its second derivative is the local Fisher information matrix.
            mean_kl = self._mean_kl_divergence(
                batch.observations,
                reference_distribution,
            )

            # Preserve this derivative graph so a second derivative can produce
            # Fisher-vector products without constructing the Fisher matrix.
            flat_kl_gradient = _flat_grad(
                mean_kl,
                self.policy_parameters,
                create_graph=True,
            )

            # --------------------------------------------------------------
            # 5. Natural-gradient direction
            # --------------------------------------------------------------

            fisher_vector_product = partial(
                self._fisher_vector_product,
                flat_kl_gradient,
            )

            # Approximately solve:
            # (Fisher + damping * identity) * search_direction = policy_gradient
            search_direction = _conjugate_gradient(
                fisher_vector_product,
                policy_gradient,
                max_iterations=self.cfg.cg_iters,
                tolerance=self.cfg.cg_residual_tol,
            )

            # Measure the unscaled direction in the damped Fisher geometry.
            directional_curvature = torch.dot(
                search_direction,
                fisher_vector_product(search_direction),
            )

            # Scale the direction so the quadratic KL estimate reaches max_kl.
            max_step_size = torch.sqrt(
                (2.0 * self.cfg.max_kl) / (directional_curvature + 1e-8)
            )
            full_step = max_step_size * search_direction

            # First-order surrogate improvement predicted for the full step.
            expected_improvement = torch.dot(
                policy_gradient,
                full_step,
            )

            # --------------------------------------------------------------
            # 6. Policy parameter update
            # --------------------------------------------------------------

            # NPG applies the full step. TRPO can override this hook to test
            # progressively smaller fractions using backtracking line search.
            surrogate_after_step, mean_kl_after_step, step_fraction = (
                self._apply_policy_step(
                    batch=batch,
                    advantages=batch.advantages,
                    reference_distribution=reference_distribution,
                    base_policy_parameters=base_policy_parameters,
                    full_step=full_step,
                    surrogate_before_step=surrogate_before_step,
                )
            )

            # Measure the updated policy without retaining another graph.
            with torch.no_grad():
                updated_distribution: Distribution = self.policy(
                    batch.observations
                )
                entropy = updated_distribution.entropy().mean()

            # Nonlinear surrogate change actually observed after the step.
            surrogate_improvement = (
                surrogate_after_step - surrogate_before_step
            )

            # --------------------------------------------------------------
            # 7. Batch metric accumulation
            # --------------------------------------------------------------

            # Weight metrics by samples so a smaller final minibatch is not
            # given the same influence as a complete minibatch.
            num_batch_samples = batch.observations.shape[0]
            num_samples += num_batch_samples
            gradient_steps += 1

            for name, value in (
                ("losses/policy_loss", policy_loss),
                ("losses/value_loss", value_loss),
                ("diagnostics/entropy", entropy),
                ("diagnostics/mean_kl", mean_kl_after_step),
                ("diagnostics/surrogate_improvement", surrogate_improvement),
                ("diagnostics/expected_improvement", expected_improvement),
                ("diagnostics/step_fraction", step_fraction),
                ("grads/policy_norm", policy_gradient.norm()),
                ("grads/value_norm", value_grad_norm),
            ):
                metric_value = (
                    value.detach().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                metric_sums[name] += metric_value * num_batch_samples

        # ------------------------------------------------------------------
        # 8. Update metric reduction
        # ------------------------------------------------------------------

        # Reduce all accumulated batch measurements to one update-level record.
        metrics = {
            name: total / num_samples
            for name, total in metric_sums.items()
        }
        metrics.update(
            {
                "diagnostics/explained_variance": explained_variance(
                    buffer.values.flatten(),
                    value_targets.flatten(),
                ),
                "charts/value_learning_rate": self.optim_critic.param_groups[0]["lr"],
            }
        )

        return TrainingStats(
            metrics=metrics,
            gradient_steps=gradient_steps,
        )


def npg(
    envs            :   VectorEnv,
    policy          :   nn.Module,
    value_fn        :   nn.Module,
    cfg             :   NPGConfig,
    writer          :   SummaryWriter | None = None,
    logger          :   Logger | None = None,
    evaluator       :   Evaluator | None = None,
) -> MetricHistory:
    """Construct an NPG learner and run the shared on-policy training loop."""
    learner = NPGLearner(policy, value_fn, cfg)

    return train_on_policy(
        envs=envs,
        learner=learner,
        writer=writer,
        logger=logger,
        evaluator=evaluator,
    )
