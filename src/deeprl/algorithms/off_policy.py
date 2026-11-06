"""Shared settings and the learner interface for off-policy algorithms.

Off-policy algorithms learn from replayed transitions, which may have been
collected by earlier behaviour policies. This module holds what they share:

* ``OffPolicyConfig``: run length, replay capacity, sampling and update settings.
* ``OffPolicyLearner``: the base class each algorithm extends. Concrete learners
  own their networks, optimizers, losses and target-network updates.

One collection/learning cycle, as run by ``training/train_off_policy.py``::

    collect transitions   (behaviour_policy, collect_transitions)
    update statistics     (obs_processor.update, after each vector step, by the trainer)
    handle step events    (learner.on_environment_step, after each vector step)
    sample replay         (replay_buffer.sample, after learning_starts)
    update the networks   (learner.update, once per sampled minibatch)

The trainer owns collection, progress, reporting and checkpoints. The learner
owns target preparation and optimization. Unlike on-policy learning, there
is no shared target stage (like GAE) here: every algorithm builds its own
Bellman target (DQN takes the best next Q-value, TD3 the smaller of two
critics, SAC adds an entropy bonus), and it must be recomputed for every
sampled minibatch with the current target networks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch

from deeprl import learner_state
from deeprl.buffers import ReplayBatch
from deeprl.stats import TrainingProgress, TrainingStats


@dataclass(frozen=True)
class OffPolicyConfig:
    """Settings shared by off-policy algorithms.

    Each algorithm's own config extends this class with its optimization,
    exploration and target-network settings.

    Attributes:
        total_timesteps: Total environment transitions to collect, counted
            across all environments. Each vector step adds N, so the trainer
            rounds this down to a multiple of N. On resume this remains the
            cumulative budget, not an additional number of transitions.
        seed: Nonnegative random seed. The trainer seeds Torch and the initial
            environment reset. Network initialization follows this seed only
            if the caller seeds before building the networks.
        device: Device for networks and sampled replay batches. Concrete
            learners or their callers place the networks on this device.
        log_every: For the logger the trainer creates: a periodic training
            report becomes due every this many environment steps. 0 turns
            periodic reports off; finished episodes and the final report
            are still logged. A logger you pass in uses its own settings,
            and its final report is written when you call its
            ``finish()``.
        gamma: Discount factor for Bellman targets, between 0 and 1.
        buffer_size: Maximum stored transitions across all environments.
            Once full, replay overwrites the oldest transitions.
        replay_device: Device holding the complete replay storage. It can be
            CPU while sampled batches and networks use CUDA.
        batch_size: B, transitions sampled for one learner update. Sampling
            is with replacement, so B can exceed the current replay size.
        learning_starts: Fresh transitions collected before learning starts.
            Minibatches drawn from a nearly empty buffer would train on the
            same few transitions over and over. Resume starts with empty
            replay and collects this many again.
        train_freq: T, vector steps per collection cycle. With N environments,
            a full cycle inserts T * N transitions; each step inserts N.
        gradient_steps: Replay-minibatch updates per collection cycle once
            learning is enabled. Actor delays and target refreshes are
            scheduled by the concrete learner, not this setting. Together
            with ``train_freq`` it sets the replay ratio: about
            ``gradient_steps / (train_freq * N)`` minibatch updates per
            collected transition. That counts update calls. Each call
            samples ``batch_size`` transitions, so the sampled-transition
            ratio is ``batch_size`` times larger, and individual replay
            entries are sampled different numbers of times.
    """

    # Run control.
    total_timesteps : int          = 500_000  # Total transitions across all environments.
    seed            : int          = 0
    device          : torch.device = torch.device("cpu")
    log_every       : int          = 1_000    # Transition reporting interval; 0 disables.

    # Discount and replay storage.
    gamma           : float        = 0.99       # Discount factor for Bellman targets.
    buffer_size     : int          = 1_000_000  # Maximum stored transitions.
    replay_device   : torch.device = torch.device("cpu")  # Complete replay storage.
    batch_size      : int          = 256        # Transitions sampled per learner update.

    # Collection and learning schedule.
    learning_starts : int          = 1_000  # Fresh transitions before learning/refilling.
    train_freq      : int          = 1      # Vector steps collected per learning phase.
    gradient_steps  : int          = 1      # Replay-minibatch updates per learning phase.

    def __post_init__(self) -> None:
        """Check the settings when the config is created.

        Raises:
            ValueError: If a count is not a large enough integer, or gamma
                is not a finite number between 0 and 1.
            TypeError: If device or replay_device is not a torch.device.
        """
        # 1. Counts must be whole numbers (booleans are rejected) and at
        #    least the given minimum.
        for name, minimum in (
            ("total_timesteps", 1),
            ("buffer_size", 1),
            ("batch_size", 1),
            ("train_freq", 1),
            ("gradient_steps", 1),
            ("seed", 0),
            ("learning_starts", 0),
            ("log_every", 0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}, got {value!r}"
                )

        # 2. gamma must be a finite number between 0 and 1.
        if (
            isinstance(self.gamma, bool)
            or not isinstance(self.gamma, Real)
            or not isfinite(self.gamma)
            or not 0.0 <= self.gamma <= 1.0
        ):
            raise ValueError(
                f"gamma must be finite and between 0 and 1, got {self.gamma!r}"
            )

        # 3. Both devices must be torch.device objects (a plain string such as
        #    "cuda" is rejected). They may differ: replay can stay on the CPU
        #    while the networks and sampled batches use the GPU.
        for name in ("device", "replay_device"):
            if not isinstance(getattr(self, name), torch.device):
                raise TypeError(f"{name} must be a torch.device")


class OffPolicyLearner(ABC):
    """Base class for off-policy algorithms.

    The base stores the config without prescribing a fixed set of networks.
    A Q-learning implementation can have online and target Q-networks only;
    an actor-critic implementation can own an actor and one or more critics.

    What this base class provides:

    * Checks for config changes when resuming.
    * ``on_environment_step``: an optional collection-time event, a no-op by
      default. A learner may use it for environment-step target refreshes,
      for example a DQN that copies its network into the target network
      every C environment steps.

    What each algorithm must implement:

    * ``update``: its own training step on one sampled replay minibatch,
      including targets, losses and any delayed actor or target updates.
    * ``state_dict`` / ``load_state_dict``: save and restore its networks,
      optimizers and other learning state, explicitly. The trainer saves the
      rest of a checkpoint (progress, random-number state, observation
      statistics, configs).
    * ``apply_resume_overrides``: apply allowed config changes when resuming.

    Behaviour and inference policies are passed separately to the trainer.
    They must use the learner's networks and shared observation processing;
    this base does not create a mandatory actor or policy wrapper. Why not,
    when the on-policy base does? Off-policy algorithms have no common actor:
    DQN acts by picking the action with the highest Q-value, TD3 has an
    actor that outputs actions directly, SAC has one that outputs a
    distribution. Each also explores differently. So the caller builds both
    policies around the learner's networks.

    Args:
        cfg: The algorithm's config.

    Attributes:
        cfg: The algorithm's config.
        mutable_on_resume: Config fields allowed to change when training
            resumes from a checkpoint. Algorithms extend this set, for example
            with learning rates.
    """

    mutable_on_resume = frozenset({
        "total_timesteps",
        "device",
        "replay_device",
        "log_every",
    })

    # ------------------------------------------------------------------
    # 0. Learner setup
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: OffPolicyConfig,
    ) -> None:
        """Store the config; the concrete learner constructs its learning components."""
        self.cfg = cfg

    # ------------------------------------------------------------------
    # 1. Learner checkpoint state
    # ------------------------------------------------------------------
    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the learner's own state.

        That includes online and target networks, optimizers and any auxiliary
        learning or exploration state, each under its own key. It is not
        everything needed to resume: the trainer saves the rest of the
        checkpoint (progress, random-number state, observation statistics,
        configs). The current off-policy trainer does not save replay data.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state made by ``state_dict``.

        Implementations check the keys first, then restore into the existing
        networks and optimizers. Loading is not all-or-nothing: an error while
        loading one part's tensors can happen after earlier parts were
        already restored. After such an error, build a new learner.

        Args:
            state: The saved learner state, not the whole training checkpoint.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 2. Resume configuration
    # ------------------------------------------------------------------
    def validate_resume_config(
        self,
        saved_config: dict[str, Any],
        current_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Check that a resumed run only changes settings it is allowed to change.

        Args:
            saved_config: The config stored in the checkpoint.
            current_config: The config of the run that is resuming.

        Returns:
            The allowed fields that changed, with their new values, for
            ``apply_resume_overrides``.

        Raises:
            ValueError: If a field outside ``mutable_on_resume`` changed.
        """
        return learner_state.validate_resume_config(
            saved_config,
            current_config,
            mutable_on_resume=self.mutable_on_resume,
        )

    @abstractmethod
    def apply_resume_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply allowed config changes after the saved state was restored.

        For example, set a new learning rate in the restored optimizer.

        Args:
            overrides: Changed fields and their new values, from
                ``validate_resume_config``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 3. Off-policy collection events
    # ------------------------------------------------------------------
    def on_environment_step(self, *, progress: TrainingProgress) -> None:
        """Optional work after each vector step, using post-collection counters.

        Override for work on an environment-step clock, for example a DQN
        that copies its network into the target network every C environment
        steps. Work on a gradient-step clock, such as TD3's delayed actor
        update or the soft target updates of TD3 and SAC, belongs inside
        ``update``. The default does nothing; there is no universal
        target-update schedule. The trainer calls this without gradients.

        Args:
            progress: Counters after the vector step. global_step includes
                the newly collected N transitions; no replay update has run
                for this step yet.
        """

    # ------------------------------------------------------------------
    # 4. Algorithm-specific parameter updates
    # ------------------------------------------------------------------
    @abstractmethod
    def update(
        self,
        batch: ReplayBatch,
        *,
        progress: TrainingProgress,
    ) -> TrainingStats:
        """Process one replay minibatch using pre-update training progress.

        Own the losses, optimizer steps and any gradient-driven target updates.
        Return gradient_steps=1 for a completed minibatch update, or 0 if skipped;
        multiple optimizer steps on that batch still count as one update (for
        example, TD3's critic step and delayed actor step on the same batch).
        For collection, the trainer puts the inference policy's modules in
        evaluation mode; it does not touch separate critics or target
        networks. Each learner sets the modes its networks need here.

        Args:
            batch: B sampled transitions. Observations and actions keep their
                feature dimensions; rewards, terminations and truncations are
                (B,). Observations are raw; the learner processes them.
            progress: Counters before this minibatch update. Use these for
                learner-specific schedules; the trainer advances them after
                the update returns.

        Returns:
            Training statistics, such as losses and gradient steps, for
            logging.
        """
        raise NotImplementedError
