"""Processors: convert raw observations into network inputs, and network
outputs into environment commands.

This module has two classes:

* ``ObservationProcessor`` (input side) picks the observation features each
  network may see and normalizes chosen features with running statistics
  (mean and variance).
* ``ActionProcessor`` (output side) converts the network's actions into the
  range the environment expects, with optional clipping.

Both are part of the saved policy, so training, evaluation and deployment
convert data in exactly the same way::

    raw observation -> ObservationProcessor -> network -> ActionProcessor -> env

Two rules hold throughout:

* Applying a processor never changes its statistics. Statistics change only
  when the training code calls ``update`` or ``fit``.
* Processors never modify the data they receive in place. An output that
  needs no change (for example, a feature that is not normalized and is
  already on the right device) may be the same tensor as the input, or
  share its memory.

Reward scaling is not done here. It is training-only and belongs to the
learner.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from deeprl.data_spec import DataSpec, TensorData
from deeprl.normalization import RunningNormalizer


class ObservationProcessor(nn.Module):
    """Turn raw observations into network inputs.

    The processor does three jobs:

    1. **Select:** give each network only the features it may read, based on
       each feature's ``groups``:

       * empty groups: shared, both actor and critic read it;
       * ``("actor",)`` or ``("critic",)``: only that network reads it;
       * ``"privileged"``: only the critic reads it.
    2. **Normalize:** for each feature listed in ``normalize``, compute
       ``(x - mean) / sqrt(var + epsilon)`` and clip the result to
       ``[-clip, clip]``. Other features pass through unchanged.
    3. **Keep the statistics:** one mean, variance and count per normalized
       feature, saved with the policy.

    One instance is shared by the actor and the critic, so both see the same
    statistics. Before the first ``update`` or ``fit``, normalization is
    identity: normalized features keep their values (only converted to
    float32) and are not clipped. Selection and moving to the processor's
    device still happen.

    Normalized features come out as float32. Features that are not normalized
    keep their dtype, so a uint8 image stays uint8. The output has the same
    structure as the input: a tensor in gives a tensor out, a TensorDict in
    gives a TensorDict out.

    Args:
        spec: Description of the observation features (see ``data_spec.py``).
        normalize: Names of the features to normalize. They must be declared
            in ``spec`` and have a floating-point dtype.
        epsilon: Small number added to the variance before the square root,
            so the processor never divides by zero.
        clip: Normalized values are limited to ``[-clip, clip]``. ``None``
            turns clipping off.

    Example:
        Illustrative; ``rollout_observations`` and ``observations`` are raw
        observation batches from your environment.

        >>> spec = DataSpec(observations=(FeatureSpec("state", (11,), "float32"),), actions=())
        >>> processor = ObservationProcessor(spec, normalize=("state",))
        >>> processor.update(rollout_observations)       # learn the statistics
        >>> inputs = processor(observations, group="actor")
    """

    def __init__(
        self,
        spec: DataSpec,
        *,
        normalize: Sequence[str] = (),
        epsilon: float = 1e-8,
        clip: float | None = 10.0,
    ) -> None:
        """Check the settings and create one statistics object per normalized feature.

        Raises:
            ValueError: If the spec has no observations, ``epsilon`` or
                ``clip`` is not a finite positive number, or a ``normalize``
                name is repeated, undeclared or not a floating-point feature.
            TypeError: If ``normalize`` is a single string instead of a
                sequence of names.
        """
        super().__init__()

        # 1. There must be observation features to process.
        if not isinstance(spec, DataSpec) or not spec.observations:
            raise ValueError("spec must declare observation features")

        # 2. epsilon and clip must be finite positive numbers (clip may be None).
        for name, value in (("epsilon", epsilon), ("clip", clip)):
            if name == "clip" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

        # 3. normalize must be a list of names, not one string: the string
        #    "state" would otherwise be read as the names "s", "t", "a", ...
        if isinstance(normalize, str):
            raise TypeError("normalize must be a sequence of feature names")
        normalize = tuple(normalize)

        # 4. Every name must be unique and declared in the spec.
        features = {feature.name: feature for feature in spec.observations}
        if len(set(normalize)) != len(normalize) or set(normalize) - features.keys():
            raise ValueError("normalize must contain unique declared observation names")

        # 5. Only floating-point features can be normalized. Integer data such
        #    as a uint8 image would lose its meaning.
        for name in normalize:
            if not getattr(torch, features[name].dtype).is_floating_point:
                raise ValueError(f"only floating-point features can be normalized: {name}")

        # 6. Store the settings.
        self.spec = spec
        self.normalize = normalize
        self.epsilon = float(epsilon)
        self.clip = None if clip is None else float(clip)

        # 7. Create one RunningNormalizer (mean, var, count) per normalized
        #    feature. ModuleDict keys cannot contain ".", and feature names may
        #    contain "/" or ".", so each normalizer is stored under its position
        #    ("0", "1", ...). _indices maps a feature name to that position.
        self.normalizers = nn.ModuleDict(
            {
                str(index): RunningNormalizer(
                    features[name].shape,
                    epsilon=epsilon,
                    clip=clip,
                )
                for index, name in enumerate(normalize)
            }
        )
        self._indices = {name: str(index) for index, name in enumerate(normalize)}

        # 8. An empty tensor that moves with ``.to(device)``. It tells the
        #    processor which device it is on, even when it has no normalizers.
        #    persistent=False keeps it out of saved state.
        self.register_buffer("_device", torch.empty(0), persistent=False)

    @property
    def config(self) -> dict[str, Any]:
        """Settings needed to rebuild this processor, as plain JSON data.

        The data spec is saved separately; the statistics are saved by
        ``state_dict()``.
        """
        return {
            "normalize": list(self.normalize),
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    def _prepare(
        self,
        observations: TensorData,
        group: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
        """Pick the features a network may see and check them against the spec.

        Args:
            observations: A tensor (only when the spec has one feature) or a
                TensorDict holding the observation features.
            group: ``"actor"``, ``"critic"`` or ``None`` for all features.
                ``None`` is used when updating statistics.

        Returns:
            A dictionary from feature name to tensor, and the batch shape
            (for example ``(N,)`` or ``(T, N)``).

        Raises:
            ValueError: If ``group`` is unknown, no feature is visible to it,
                or the data does not match the spec.
            TypeError: If ``observations`` is neither a tensor nor a
                TensorDict.
        """
        # 1. Only these groups exist.
        if group not in (None, "actor", "critic"):
            raise ValueError("group must be actor, critic or None")

        # 2. Choose the visible features:
        #    - group None: every feature;
        #    - privileged features: critic only;
        #    - other features: everyone if their groups are empty, otherwise
        #      only the groups listed.
        features = tuple(
            feature for feature in self.spec.observations
            if group is None or (
                group == "critic" if "privileged" in feature.groups
                else not feature.groups or group in feature.groups
            )
        )
        if not features:
            raise ValueError(f"no observation features declared for {group}")

        # 3. Find the batch shape and keep only the visible features.
        if isinstance(observations, torch.Tensor):
            # A plain tensor holds one feature. Its batch shape is everything
            # before the feature's own shape: (N, 11) with shape (11,) -> (N,).
            event_dims = len(features[0].shape)
            batch_shape = tuple(
                observations.shape[:-event_dims] if event_dims else observations.shape
            )
            selected = observations
        elif isinstance(observations, TensorDictBase):
            # A TensorDict knows its batch shape. select() keeps only the
            # visible keys; strict=True fails if one of them is missing.
            batch_shape = tuple(observations.batch_size)
            selected = observations if group is None else observations.select(
                *(tuple(feature.name.split("/")) for feature in features),
                strict=True,
            )
        else:
            raise TypeError("observations must be a Tensor or TensorDict")

        # 4. Check keys, shapes and dtypes against the visible features.
        DataSpec(observations=features, actions=()).validate_observations(
            selected,
            batch_shape=batch_shape,
        )

        # 5. Return one tensor per visible feature, by name.
        return {
            feature.name: selected if isinstance(selected, torch.Tensor)
            else selected[tuple(feature.name.split("/"))]
            for feature in features
        }, batch_shape

    def forward(
        self,
        observations: TensorData,
        *,
        group: str | None = None,
    ) -> TensorData:
        """Turn raw observations into network inputs, using the current statistics.

        This never changes the statistics, in train or eval mode, so the same
        inputs give the same outputs until ``update`` or ``fit`` is called.

        Args:
            observations: Raw observations: a tensor (one feature) or a
                TensorDict.
            group: ``"actor"``, ``"critic"`` or ``None`` for all features.

        Returns:
            The processed observations, with the same structure as the input
            and only the features visible to ``group``.
        """
        # 1. Pick this network's features and check them against the spec.
        values, batch_shape = self._prepare(observations, group)

        # 2. Move each feature to the processor's device and normalize the
        #    features listed in ``normalize``. Other features stay unchanged.
        processed = {}
        for name, value in values.items():
            value = value.to(device=self._device.device)
            if name in self._indices:
                value = self.normalizers[self._indices[name]](value)
            processed[name] = value

        # 3. Return the same structure as the input: tensor in, tensor out.
        if isinstance(observations, torch.Tensor):
            return next(iter(processed.values()))

        # TensorDict in, TensorDict out, rebuilding nested keys such as
        # "images/wrist" -> result["images"]["wrist"].
        result = TensorDict(
            {},
            batch_size=batch_shape,
            device=self._device.device,
        )
        for name, value in processed.items():
            result[tuple(name.split("/"))] = value
        return result

    @torch.no_grad()
    def update(self, observations: TensorData) -> None:
        """Add new observations to the running statistics.

        All leading batch axes are pooled: a rollout shaped ``(T, N, ...)``
        adds ``T * N`` samples. New samples are merged with everything seen so
        far. In on-policy training the trainer calls this once per rollout,
        after the learner update.

        Args:
            observations: Raw observations with every declared feature.

        Raises:
            ValueError: If a normalized feature contains NaN or infinity. The
                statistics are left unchanged.
        """
        # 1. Use every feature (group None) and check it against the spec.
        values, _ = self._prepare(observations)

        # 2. Reject NaN or infinity before changing anything: one bad value
        #    would corrupt the mean and variance for the rest of training.
        for name in self.normalize:
            if not torch.isfinite(values[name]).all().item():
                raise ValueError(f"normalization samples must be finite: {name}")

        # 3. Merge the new samples into each feature's statistics.
        for name, index in self._indices.items():
            self.normalizers[index].update(values[name])

    @torch.no_grad()
    def fit(self, observations: TensorData) -> None:
        """Replace the statistics with ones computed from this data alone.

        Unlike ``update``, earlier statistics are discarded. This is used when
        statistics come from a fixed dataset, as in offline RL or imitation
        learning: fit once, then keep them frozen.

        Args:
            observations: The complete training data, with every declared
                feature.

        Raises:
            ValueError: If the data is empty or invalid. The existing
                statistics are left unchanged.
        """
        # 1. Use every feature and check it against the spec.
        values, batch_shape = self._prepare(observations)

        # 2. Statistics from zero samples would be meaningless.
        if 0 in batch_shape:
            raise ValueError("cannot fit observation statistics from an empty batch")

        # 3. Compute new statistics in fresh normalizers first, without touching
        #    the existing ones.
        fitted = {}
        for name, index in self._indices.items():
            normalizer = self.normalizers[index]
            candidate = RunningNormalizer(
                normalizer.shape,
                epsilon=self.epsilon,
                clip=self.clip,
            ).to(self._device.device)
            candidate.update(values[name])
            fitted[index] = candidate.state_dict()

        # 4. Only after every feature succeeded, copy the new statistics in.
        #    A failure in step 3 therefore never leaves the processor half-fitted.
        for index, state in fitted.items():
            self.normalizers[index].load_state_dict(state)


class ActionProcessor(nn.Module):
    """Turn network actions into environment commands.

    Networks usually work in a fixed range such as ``[-1, 1]``, while the
    environment expects its own range, for example ``[-0.4, 0.4]`` for
    Humanoid. This processor does two optional steps, in order:

    1. **Clip** actions to ``source_bounds`` (only if ``clip=True``).
    2. **Rescale** linearly from ``source_bounds`` to ``target_bounds`` (only
       if ``target_bounds`` is given). Values outside the source range are
       stretched beyond the target range, exactly like Gymnasium's
       ``RescaleAction``.

    With no bounds at all it returns actions unchanged, which is what discrete
    actions need. It only produces the command sent to the environment; the
    original sampled action, which PPO needs for its probability calculation,
    is never changed.

    Args:
        source_bounds: ``(low, high)`` of the network's action range. Each
            value can be a number or one number per action dimension.
        target_bounds: ``(low, high)`` of the environment's action range.
            Omit it to clip without rescaling.
        clip: Whether to clip actions to ``source_bounds`` first.

    Example:
        Illustrative; ``low`` and ``high`` are the environment's action
        bounds, ``env.action_space.low`` and ``env.action_space.high``.

        >>> # Humanoid: rescale [-1, 1] to the native [-0.4, 0.4], no clipping.
        >>> ActionProcessor(source_bounds=(-1.0, 1.0), target_bounds=(low, high))
        >>> # Hopper: clip to the action space only.
        >>> ActionProcessor(source_bounds=(low, high), clip=True)
    """

    def __init__(
        self,
        *,
        source_bounds: tuple[Any, Any] | None = None,
        target_bounds: tuple[Any, Any] | None = None,
        clip: bool = False,
    ) -> None:
        """Check the bounds and store them as tensors.

        Raises:
            TypeError: If ``clip`` is not a boolean.
            ValueError: If the bounds are missing, not finite, not increasing,
                or cannot be matched to each other.
        """
        super().__init__()

        # 1. Check that the settings make sense together:
        #    - clipping or rescaling needs source bounds;
        #    - source bounds alone are useless unless clipping is on.
        if not isinstance(clip, bool):
            raise TypeError("clip must be boolean")
        if source_bounds is None and (target_bounds is not None or clip):
            raise ValueError("action rescaling/clipping requires source bounds")
        if source_bounds is not None and target_bounds is None and not clip:
            raise ValueError("source bounds require clipping or target bounds")
        self.clip = clip

        # 2. Convert each (low, high) pair to float64 tensors and check it.
        #    The bounds are settings, not learned values, so they are saved in
        #    ``config`` and kept out of ``state_dict`` (persistent=False).
        for name, bounds in (("source", source_bounds), ("target", target_bounds)):
            if bounds is None:
                low = high = None
            else:
                if len(bounds) != 2:
                    raise ValueError(f"{name}_bounds must contain low and high")
                low, high = (
                    torch.as_tensor(value, dtype=torch.float64).clone()
                    for value in bounds
                )
                if low.shape != high.shape or low.ndim > 1:
                    raise ValueError(f"{name} bounds must be matching scalars or vectors")
                if not (
                    torch.isfinite(low).all()
                    and torch.isfinite(high).all()
                    and (low < high).all()
                ):
                    raise ValueError(f"{name} bounds must be finite and strictly increasing")
            self.register_buffer(f"{name}_low", low, persistent=False)
            self.register_buffer(f"{name}_high", high, persistent=False)

        # 3. Give all four bounds the same shape. A scalar source range such
        #    as (-1, 1) is expanded to one value per joint when the target
        #    bounds are per joint.
        if target_bounds is not None:
            try:
                self.source_low, self.source_high, self.target_low, self.target_high = torch.broadcast_tensors(
                    self.source_low,
                    self.source_high,
                    self.target_low,
                    self.target_high,
                )
            except RuntimeError as error:
                raise ValueError("source and target bounds must have broadcastable shapes") from error

    @property
    def config(self) -> dict[str, Any]:
        """Settings needed to rebuild this processor, as plain JSON data.

        The bounds are fixed settings, not learned statistics, so this is
        everything the processor needs.
        """
        return {
            "source_bounds": None if self.source_low is None else [
                self.source_low.tolist(),
                self.source_high.tolist(),
            ],
            "target_bounds": None if self.target_low is None else [
                self.target_low.tolist(),
                self.target_high.tolist(),
            ],
            "clip": self.clip,
        }

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert network actions into environment commands.

        The output has the same shape and dtype as the input. The input tensor
        itself is never modified.

        Args:
            actions: Batched actions, for example shaped ``(N, action_dim)``.
                With per-joint bounds, the last dimension must match them.

        Returns:
            The commands to send to the environment.

        Raises:
            TypeError: If ``actions`` is not a tensor, or is not floating-point
                while bounds are set.
            ValueError: If the last dimension does not match per-joint bounds.
        """
        # 1. Actions must be a tensor.
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a Tensor")

        # 2. No bounds means nothing to do, for example discrete actions.
        if self.source_low is None:
            return actions

        # 3. Clipping and rescaling only make sense for continuous actions, and
        #    per-joint bounds need one action value per joint.
        if not actions.is_floating_point():
            raise TypeError("continuous action conversion requires floating-point actions")
        if self.source_low.ndim and (
            not actions.ndim or actions.shape[-1] != self.source_low.numel()
        ):
            raise ValueError("actions must match the bounds' trailing dimension")

        # 4. Optionally clip to the source range. clamp() returns a new tensor,
        #    so the caller's sampled actions stay unchanged.
        source_low, source_high = self.source_low.to(actions), self.source_high.to(actions)
        if self.clip:
            actions = actions.clamp(min=source_low, max=source_high)

        # 5. Without target bounds there is no rescaling: return the clipped actions.
        if self.target_low is None:
            return actions

        # 6. Rescale linearly from the source range to the target range:
        #        command = target_low + (a - source_low) * target_span / source_span
        #    Written here with a gradient and an intercept, in the same order of
        #    operations as Gymnasium's RescaleAction (gymnasium/wrappers/utils.py,
        #    rescale_box, MIT License), so commands match Gymnasium bit for bit.
        target_low, target_high = self.target_low.to(actions), self.target_high.to(actions)
        # Form the native span in higher precision, then round the coefficient
        # to the action dtype before applying RescaleAction's inverse mapping.
        target_span = target_high.to(torch.float64) - target_low.to(torch.float64)
        gradient = ((source_high - source_low).to(torch.float64) / target_span).to(actions)
        intercept = gradient * -target_low + source_low
        return (actions - intercept) / gradient
