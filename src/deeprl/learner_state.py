"""Named learner state and resume rules independent of any update contract."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from torch import nn
from torch.optim import Optimizer


def _state_objects(
    modules_by_role: Mapping[str, nn.Module],
    optimizers_by_role: Mapping[str, Optimizer],
) -> dict[str, nn.Module | Optimizer]:
    if modules_by_role.keys() & optimizers_by_role.keys():
        raise ValueError("module and optimizer state keys must be distinct")
    return {**modules_by_role, **optimizers_by_role}


def learner_state_dict(
    modules_by_role: Mapping[str, nn.Module],
    optimizers_by_role: Mapping[str, Optimizer],
) -> dict[str, Any]:
    """Return module and optimizer states under their existing flat role keys."""
    return {
        name: component.state_dict()
        for name, component in _state_objects(modules_by_role, optimizers_by_role).items()
    }


def load_learner_state_dict(
    state: Mapping[str, Any],
    modules_by_role: Mapping[str, nn.Module],
    optimizers_by_role: Mapping[str, Optimizer],
) -> None:
    """Check all role names before restoring existing modules and optimizers."""
    components = _state_objects(modules_by_role, optimizers_by_role)
    missing = components.keys() - state.keys()
    unexpected = state.keys() - components.keys()
    if missing or unexpected:
        raise ValueError(
            f"checkpoint learner state differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name, component in components.items():
        component.load_state_dict(state[name])


def validate_resume_config(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    mutable_on_resume: Collection[str],
    learning_rate_fields: Mapping[str, str],
) -> dict[str, float]:
    """Require matching fields, validate settings, and return changed rates."""
    missing = current_config.keys() - saved_config.keys()
    unexpected = saved_config.keys() - current_config.keys()
    if missing or unexpected:
        raise ValueError(
            f"checkpoint learner config differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    changed = [
        name for name, value in current_config.items()
        if name not in mutable_on_resume
        and saved_config[name] != value
    ]
    if changed:
        raise ValueError(f"checkpoint learner config differs: {', '.join(changed)}")

    return {
        role: current_config[name]
        for name, role in learning_rate_fields.items()
        if saved_config[name] != current_config[name]
    }


def apply_resume_overrides(
    learning_rates: Mapping[str, float],
    optimizers_by_role: Mapping[str, Optimizer],
) -> None:
    """Apply configured rate changes after restoring optimizer state."""
    for role, learning_rate in learning_rates.items():
        for group in optimizers_by_role[role].param_groups:
            group["lr"] = learning_rate
