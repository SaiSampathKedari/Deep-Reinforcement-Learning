"""The complete policy: input processing, model, action choice and output processing.

A network alone is not a usable policy. To act, raw observations must first be
processed, the network's output must be turned into an action, and that action
must be converted into a command the environment understands. ``InferencePolicy``
puts these four parts together::

    raw observation -> ObservationProcessor -> model -> action selector
                    -> ActionProcessor -> environment command

The same object is used for on-policy data collection, for evaluation and for
a saved policy, so all three process observations and actions in exactly the
same way. The chosen actions can still differ: collection samples actions,
while evaluation may take the most likely one. Off-policy collection uses a
separate behaviour policy, which adds exploration.

The action selector is the rule that turns the model's output into an action.
Three are built in:

* ``distribution_action``: the model returns a probability distribution
  (PPO, A2C, NPG, TRPO). Sample from it, or take its most likely action.
* ``greedy_action``: the model returns Q-values (DQN). Take the action with
  the highest value.
* ``deterministic_action``: the model returns the action itself (TD3, DDPG).
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.distributions import Distribution

from deeprl.data_spec import TensorData
from deeprl.processors import ActionProcessor, ObservationProcessor


def distribution_action(distribution: Distribution, deterministic: bool) -> TensorData:
    """Choose an action from a probability distribution.

    Args:
        distribution: The model's output, one distribution per batch entry.
        deterministic: ``False`` draws a random sample. ``True`` takes the most
            likely action (the mode), or the mean if the distribution has no
            mode.

    Returns:
        Batched actions.

    Raises:
        TypeError: If the model returned a plain tensor (Q-values or actions)
            instead of a distribution. Such models need a different selector.
    """
    # 1. A plain tensor is not a distribution. Q-values or direct actions need
    #    greedy_action or deterministic_action instead, so fail loudly here
    #    rather than guess what the numbers mean.
    if torch.is_tensor(distribution):
        raise TypeError(
            "tensor-valued actors and Q-functions require an explicit action selector"
        )

    # 2. Stochastic mode: draw a random sample, as during training.
    if not deterministic:
        return distribution.sample()

    # 3. Deterministic mode: the most likely action. Some distributions do not
    #    define a mode; for those, use the mean.
    try:
        return distribution.mode
    except NotImplementedError:
        return distribution.mean


def greedy_action(q_values: torch.Tensor, deterministic: bool) -> torch.Tensor:
    """Choose the action with the highest Q-value.

    Exploration, such as epsilon-greedy, belongs to the behaviour policy used
    while collecting training data, not to this policy, so ``deterministic``
    has no effect.

    Args:
        q_values: One value per action, shaped ``(B, num_actions)``.
        deterministic: Ignored; the choice is always greedy.

    Returns:
        The index of the best action for each batch entry, shaped ``(B,)``.
    """
    return q_values.argmax(dim=-1)


def deterministic_action(actions: TensorData, deterministic: bool) -> TensorData:
    """Use the model's output directly as the action.

    For actors that output actions instead of distributions (TD3, DDPG).
    Exploration noise belongs to the behaviour policy used while collecting
    training data, so it is not added here and ``deterministic`` has no
    effect.

    Args:
        actions: The model's output actions.
        deterministic: Ignored.

    Returns:
        The same actions, unchanged.
    """
    return actions


class InferencePolicy(nn.Module):
    """A model together with its input processing, action choice and output processing.

    It has two entry points:

    * ``forward(observations)``: process the observations and run the model.
      It returns the raw model output, for example a distribution. Training
      uses this to compute log-probabilities and losses, so gradients flow.
    * ``act(observations)``: process, run the model, choose an action and
      convert it into an environment command. No gradients. Evaluation and
      deployment use this.

    Neither method changes the processor's statistics.

    The processors are optional: without an observation processor, raw
    observations go straight to the model; without an action processor, the
    chosen action is the command.

    Args:
        model: The network, for example a ``DiagonalGaussianPolicy``.
        observation_processor: Turns raw observations into model inputs. The
            policy uses the actor's features only.
        action_processor: Turns chosen actions into environment commands.
        action_selector: Rule that turns the model output into an action. The
            default, ``distribution_action``, suits models that return a
            distribution. Q-networks and direct-action actors must pass
            ``greedy_action`` or ``deterministic_action``.
        deterministic: Default mode for ``act``. ``True`` takes the most likely
            action; ``False`` samples.

    Example:
        Illustrative; ``model``, ``processor`` and ``raw_observations`` come
        from your code.

        >>> policy = InferencePolicy(model, observation_processor=processor)
        >>> command = policy.act(raw_observations, deterministic=True)
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
        """Store the parts of the policy.

        Raises:
            TypeError: If ``deterministic`` is not a boolean.
        """
        super().__init__()

        # 1. The default mode must be a real boolean, not a value like 0 or "yes".
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be boolean")

        # 2. Store the parts. Registering the model and processors as
        #    submodules means ``.to(device)``, ``.eval()`` and ``state_dict()``
        #    reach all of them.
        self.model = model
        self.observation_processor = observation_processor
        self.action_processor = action_processor

        # 3. Without an explicit selector, assume the model returns a
        #    distribution.
        self.action_selector = (
            distribution_action if action_selector is None else action_selector
        )

        # 4. Default mode for ``act`` when the caller does not choose one.
        self.deterministic = deterministic

    def forward(self, observations: TensorData) -> Any:
        """Process raw observations and run the model.

        Args:
            observations: Raw observations, batched.

        Returns:
            The model's output, for example one action distribution per batch
            entry. Gradients flow through it.
        """
        # 1. Turn raw observations into model inputs, keeping the actor's
        #    features only. The statistics are used, never changed.
        if self.observation_processor is not None:
            observations = self.observation_processor(observations, group="actor")

        # 2. Run the network.
        return self.model(observations)

    @torch.no_grad()
    def act(
        self,
        observations: TensorData,
        deterministic: bool | None = None,
    ) -> TensorData:
        """Return environment commands for raw observations.

        On-policy data collection (``collect_rollout``) does not use this
        method: it needs the sampled action and its log-probability, so it
        calls ``forward`` and the action processor separately.

        Args:
            observations: Raw observations, batched.
            deterministic: ``True`` for the most likely action, ``False`` to
                sample, or ``None`` to use the policy's default mode.

        Returns:
            Batched environment commands.

        Raises:
            TypeError: If ``deterministic`` is not a boolean or ``None``.
        """
        # 1. Use the caller's mode if given, otherwise the policy's default.
        deterministic = self.deterministic if deterministic is None else deterministic
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be boolean")

        # 2. Process the observations, run the model and choose an action.
        actions = self.action_selector(self(observations), deterministic)

        # 3. Convert the action into the environment's command, for example by
        #    rescaling [-1, 1] to the joint limits. This changes only the
        #    command, not the model's action distribution.
        return actions if self.action_processor is None else self.action_processor(actions)
