"""Validation for explicit learner state and resume configuration."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def validate_state_keys(
    state: Mapping[str, Any],
    expected_keys: Collection[str],
) -> None:
    """Check component names before a learner restores any state."""
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint learner state must be a mapping")
    expected = set(expected_keys)
    missing = expected - state.keys()
    unexpected = state.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"checkpoint learner state differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def validate_resume_config(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    mutable_on_resume: Collection[str],
) -> dict[str, Any]:
    """Validate configuration and return changed allowed fields by name."""
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
        name: current_config[name]
        for name in current_config
        if saved_config[name] != current_config[name]
    }
