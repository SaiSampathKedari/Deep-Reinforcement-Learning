"""Explicit observation processing and action-coordinate conversion.

Application never fits statistics or modifies the supplied data. The caller
schedules observation updates; training-only reward processing is separate.
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
    """Select features and apply shared, explicitly updated running statistics.

    TensorDict structure and unnormalized feature dtypes are preserved.
    Each selected feature keeps its leading batch axes and trailing shape.
    Normalized features are float32. Move the processor with ``to(device)``;
    its statistics are ordinary registered buffers, saved by ``state_dict``.
    Actor and critic use this same instance with different ``group`` selections.
    """

    def __init__(
        self,
        spec: DataSpec,
        *,
        normalize: Sequence[str] = (),
        epsilon: float = 1e-8,
        clip: float | None = 10.0,
    ) -> None:
        super().__init__()
        if not isinstance(spec, DataSpec) or not spec.observations:
            raise ValueError("spec must declare observation features")
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
        if isinstance(normalize, str):
            raise TypeError("normalize must be a sequence of feature names")
        normalize = tuple(normalize)
        features = {feature.name: feature for feature in spec.observations}
        if len(set(normalize)) != len(normalize) or set(normalize) - features.keys():
            raise ValueError("normalize must contain unique declared observation names")
        for name in normalize:
            if not getattr(torch, features[name].dtype).is_floating_point:
                raise ValueError(f"only floating-point features can be normalized: {name}")

        self.spec = spec
        self.normalize = normalize
        self.epsilon = float(epsilon)
        self.clip = None if clip is None else float(clip)
        # Numeric keys also support feature names containing '/' or '.'.
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
        self.register_buffer("_device", torch.empty(0), persistent=False)

    @property
    def config(self) -> dict[str, Any]:
        """Constructor settings, excluding the separately serialized data spec."""
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
        """Select consumer-visible features and validate their shared batch axes."""
        if group not in (None, "actor", "critic"):
            raise ValueError("group must be actor, critic or None")
        features = tuple(
            feature for feature in self.spec.observations
            if group is None or (
                group == "critic" if "privileged" in feature.groups
                else not feature.groups or group in feature.groups
            )
        )
        if not features:
            raise ValueError(f"no observation features declared for {group}")
        if isinstance(observations, torch.Tensor):
            event_dims = len(features[0].shape)
            batch_shape = tuple(
                observations.shape[:-event_dims] if event_dims else observations.shape
            )
            selected = observations
        elif isinstance(observations, TensorDictBase):
            batch_shape = tuple(observations.batch_size)
            selected = observations if group is None else observations.select(
                *(tuple(feature.name.split("/")) for feature in features),
                strict=True,
            )
        else:
            raise TypeError("observations must be a Tensor or TensorDict")
        DataSpec(observations=features, actions=()).validate_observations(
            selected,
            batch_shape=batch_shape,
        )
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
        """Apply the current statistics; never update them in train or eval mode."""
        values, batch_shape = self._prepare(observations, group)
        processed = {}
        for name, value in values.items():
            value = value.to(device=self._device.device)
            if name in self._indices:
                value = self.normalizers[self._indices[name]](value)
            processed[name] = value
        if isinstance(observations, torch.Tensor):
            return next(iter(processed.values()))
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
        """Accumulate each normalized feature once across all batch dimensions."""
        values, _ = self._prepare(observations)
        for name in self.normalize:
            if not torch.isfinite(values[name]).all().item():
                raise ValueError(f"normalization samples must be finite: {name}")
        for name, index in self._indices.items():
            self.normalizers[index].update(values[name])

    @torch.no_grad()
    def fit(self, observations: TensorData) -> None:
        """Replace statistics from a complete training batch, without partial fits."""
        values, batch_shape = self._prepare(observations)
        if 0 in batch_shape:
            raise ValueError("cannot fit observation statistics from an empty batch")
        # Build all replacements before changing any existing feature statistics.
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
        for index, state in fitted.items():
            self.normalizers[index].load_state_dict(state)


class ActionProcessor(nn.Module):
    """Optional clipping in source coordinates, followed by affine rescaling.

    With no bounds this is identity, including for discrete actions. With bounds,
    clipping is opt-in: out-of-range samples otherwise extrapolate, as Gymnasium's
    RescaleAction does. Scalar bounds broadcast across joint coordinates.
    Omit target_bounds for clipping only. The original policy samples are never
    overwritten.
    """

    def __init__(
        self,
        *,
        source_bounds: tuple[Any, Any] | None = None,
        target_bounds: tuple[Any, Any] | None = None,
        clip: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(clip, bool):
            raise TypeError("clip must be boolean")
        if source_bounds is None and (target_bounds is not None or clip):
            raise ValueError("action rescaling/clipping requires source bounds")
        if source_bounds is not None and target_bounds is None and not clip:
            raise ValueError("source bounds require clipping or target bounds")
        self.clip = clip
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
        """JSON-compatible transform settings; bounds are not fitted statistics."""
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
        """Convert commands while preserving shape and dtype.

        Vector bounds require a matching trailing action dimension; scalar
        bounds also accept scalar actions. Input samples are not overwritten.
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a Tensor")
        if self.source_low is None:
            return actions
        if not actions.is_floating_point():
            raise TypeError("continuous action conversion requires floating-point actions")
        if self.source_low.ndim and (
            not actions.ndim or actions.shape[-1] != self.source_low.numel()
        ):
            raise ValueError("actions must match the bounds' trailing dimension")
        source_low, source_high = self.source_low.to(actions), self.source_high.to(actions)
        if self.clip:
            actions = actions.clamp(min=source_low, max=source_high)
        if self.target_low is None:
            return actions
        target_low, target_high = self.target_low.to(actions), self.target_high.to(actions)
        # Form the native span in higher precision, then round the coefficient
        # to the action dtype before applying RescaleAction's inverse mapping.
        target_span = target_high.to(torch.float64) - target_low.to(torch.float64)
        gradient = ((source_high - source_low).to(torch.float64) / target_span).to(actions)
        intercept = gradient * -target_low + source_low
        return (actions - intercept) / gradient
