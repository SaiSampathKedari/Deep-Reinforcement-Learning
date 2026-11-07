"""Shared settings and the learner interface for off-policy algorithms.

Off-policy algorithms (DQN, DDPG, TD3, SAC) learn from a replay buffer: a
large store of past transitions, many of them collected by older versions
of the policy. This module holds what those algorithms share:

* ``OffPolicyConfig``: run length, replay size, sampling, update and
  target-network settings.
* ``OffPolicyLearner``: the base class each algorithm extends. The concrete
  learner owns its networks, optimizers, losses and target networks.

One collection/learning cycle, as run by ``training/train_off_policy.py``::

    collect transitions    (behaviour_policy, collect_transitions)
    update statistics      (obs_processor.update, after each vector step)
    once learning_starts fresh transitions have been collected,
    repeat gradient_steps times:
        sample replay      (replay_buffer.sample)
        update networks    (learner.update, on that one minibatch)
        refresh targets    (learner.update_targets, only when due)

A refresh happens right after the update that makes it due, before the
next minibatch is sampled, not once at the end of the learning phase.

Who does what
-------------
* The trainer owns the schedule: collection, when to learn, when to refresh
  target networks, progress counters, logging and checkpoints.
* The learner owns the math: networks, Bellman targets, losses, optimizers,
  and how its target networks are refreshed.

Target networks
---------------
A Bellman target such as ``reward + gamma * Q(next_state)`` uses the
network's own estimate of the next state. If that estimate came from the
network being trained, every update would also move the target it is
chasing, and learning can oscillate or diverge. So each algorithm keeps a
lagging copy, the target network, and uses it for the next-state estimate.

A refresh moves that copy toward the online network: a full copy
(``tau=1``) once in a while, or a small step (``tau`` < 1) often. The
trainer decides when (every ``target_update_interval`` completed minibatch
updates); the learner decides how (which networks have targets, and what to
do with buffers that are not parameters). Keeping the schedule in the trainer
means every algorithm counts the same way, and a resumed run needs no extra
counter: the saved ``progress.gradient_step`` already says where the
schedule stands.

Bellman targets
---------------
Unlike on-policy learning, there is no shared target stage (like GAE)
here. Every algorithm builds its own Bellman target, recomputed for each
sampled minibatch, because the networks change after every update and a
stored target would soon be out of date. DQN uses the target network's
largest next-state Q-value; Double DQN picks the next action with the online
network and scores it with the target network. The interface also leaves
room for TD3 (the smaller of two target critics) and SAC (an
entropy-adjusted target).
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

    Each algorithm's own config adds optimization and exploration settings,
    and may override the shared defaults.

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
            replay and collects at least this many again. The threshold is
            checked after collection, so learning starts at the end of the
            first cycle that reaches it. 0 enables the first learning phase
            immediately after its collection phase.
        train_freq: T, vector steps per collection cycle. With N environments,
            a full cycle inserts T * N transitions; each step inserts N.
        gradient_steps: Calls to ``learner.update`` per collection cycle once
            learning is enabled, each with a freshly sampled minibatch. A
            learner may report a skipped update. Actor delays belong to the
            concrete learner; target refreshes use their own interval below.
            Together with ``train_freq`` it sets the replay ratio for full
            cycles after warm-up:
            ``gradient_steps / (train_freq * N)`` minibatch updates per
            collected transition. That counts update calls. Each call
            samples ``batch_size`` transitions, so the sampled-transition
            ratio is ``batch_size`` times larger, and individual replay
            entries are sampled different numbers of times.
        tau: How far one refresh moves the target networks toward the
            online networks, in (0, 1]. For parameters, each refresh applies
            ``target = (1 - tau) * target + tau * online``. 1 is a full
            ("hard") copy. The default 0.005 is a small "soft" step, so
            the target trails the online network smoothly. Buffers that are
            not parameters (such as BatchNorm statistics) are each
            learner's choice; DQN copies them.
        target_update_interval: Completed minibatch updates between target
            refreshes. 1 refreshes after every completed update. It counts
            ``progress.gradient_step``, so skipped updates and environment
            steps do not count, and the count continues on resume. Typical
            settings: soft updates every update (small ``tau``, interval
            1), as in DDPG and SAC; soft updates every 2 critic updates, as
            in TD3, which delays its actor and target updates together; or
            a hard copy (``tau=1``) every few thousand updates, as in the
            2015 DQN paper (every 10,000). A TD3 learner should use this
            interval for its delayed actor update too, so the actor step and
            the target refresh land on the same minibatch.
    """

    # Run control.
    total_timesteps : int          = 500_000  # Total transitions across all environments.
    seed            : int          = 0
    device          : torch.device = torch.device("cpu")
    log_every       : int          = 1_000    # Periodic report interval; 0 disables periodic reports.

    # Discount and replay storage.
    gamma           : float        = 0.99       # Discount factor for Bellman targets.
    buffer_size     : int          = 1_000_000  # Maximum stored transitions.
    replay_device   : torch.device = torch.device("cpu")  # Complete replay storage.
    batch_size      : int          = 256        # Transitions sampled per learner update.

    # Collection and learning schedule.
    learning_starts : int          = 1_000  # Fresh transitions before learning; reapplied on resume.
    train_freq      : int          = 1      # Vector steps collected per learning phase.
    gradient_steps  : int          = 1      # Replay-minibatch update calls per learning phase.

    # Target networks: the learner applies tau, the trainer the interval.
    tau                    : float = 0.005  # Refresh step size; 1 is a full copy.
    target_update_interval : int   = 1      # Completed updates between refreshes.

    def __post_init__(self) -> None:
        """Check the settings when the config is created.

        Raises:
            ValueError: If a count, gamma or tau is invalid.
            TypeError: If device or replay_device is not a torch.device.
        """
        # 1. Counts must be whole numbers (booleans are rejected) and at
        #    least the given minimum. target_update_interval needs at least
        #    1: the trainer checks ``gradient_step % target_update_interval``,
        #    and 0 would divide by zero.
        for name, minimum in (
            ("total_timesteps", 1),
            ("buffer_size", 1),
            ("batch_size", 1),
            ("train_freq", 1),
            ("gradient_steps", 1),
            ("target_update_interval", 1),
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

        # 3. tau must be a finite number in (0, 1]. With 0 the target would
        #    never move from its initial weights; above 1 a refresh would
        #    overshoot past the online network.
        if (
            isinstance(self.tau, bool)
            or not isinstance(self.tau, Real)
            or not isfinite(self.tau)
            or not 0.0 < self.tau <= 1.0
        ):
            raise ValueError(f"tau must be finite and in (0, 1], got {self.tau!r}")

        # 4. Both devices must be torch.device objects (a plain string such as
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

    What each algorithm must implement:

    * ``update``: its own training step on one sampled replay minibatch,
      including Bellman targets, losses and any delayed actor updates.
    * ``update_targets``: move its target networks one step toward the
      online networks. The trainer decides when to call it; the learner
      only decides how.
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
    distribution. Each also explores differently. So the caller supplies both
    policies around the learner's networks; a learner's behaviour-policy
    method can be passed directly.

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
    # 3. Target-network updates
    # ------------------------------------------------------------------
    @abstractmethod
    def update_targets(self) -> None:
        """Move the target networks one step toward the online networks.

        Why target networks exist (see the module docstring): the Bellman
        target uses a lagging copy, so that an update does not also move
        the target it is chasing. This method lets that copy catch up a
        little.

        How (the learner's job): one call is one refresh, using
        ``cfg.tau``. For parameters,
        ``target = (1 - tau) * target + tau * online``; ``tau=1`` is a
        full copy. The learner chooses the online/target pairs (DQN: one
        Q-network; TD3: an actor and two critics) and what to do with buffers
        that are not parameters, such as BatchNorm statistics.

        When (the trainer's job): right after a completed minibatch update
        that brings ``progress.gradient_step`` to a multiple of
        ``cfg.target_update_interval``, inside ``torch.no_grad()``. Skipped
        updates and environment steps never trigger it. So an implementation
        keeps no schedule of its own and takes no optimizer steps here.
        """
        raise NotImplementedError

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

        Own the losses and optimizer steps, including delayed actor updates.
        Do not refresh target networks here; the trainer calls
        ``update_targets`` after this method returns, when a refresh is due.

        Delayed actor updates (TD3): ``progress`` holds the counts from
        before this update. If this update completes, ``gradient_step``
        becomes ``progress.gradient_step + 1``, and the trainer refreshes
        targets when that number is a multiple of
        ``target_update_interval``. So test the same number for the
        delayed actor step, after a successful critic update:
        ``(progress.gradient_step + 1) % cfg.target_update_interval == 0``.
        The actor step then comes first, and the target refresh follows on
        the same minibatch.

        Return ``gradient_steps=1`` for a completed minibatch update, or 0 if
        skipped. Multiple optimizer steps on that batch still count as one
        update (for example, a critic step and delayed actor step on the
        same batch). This count differs from ``progress.update``, which
        counts completed collection/learning cycles, including warm-up.

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
