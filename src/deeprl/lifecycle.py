"""Configuration and random-state helpers shared by training workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _config_json_default(value: Any) -> Any:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (torch.device, Path)):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def config_to_jsonable(config: Any) -> dict[str, Any]:
    """Copy a dataclass or mapping into plain JSON-compatible configuration.

    Classes become qualified names, devices and paths become strings, and
    tuples become lists. Unsupported objects and nonfinite numbers are rejected.
    """
    if not isinstance(config, Mapping) and not (
        is_dataclass(config) and not isinstance(config, type)
    ):
        raise TypeError("configuration must be a dataclass instance or mapping")
    return json.loads(
        json.dumps(config, default=_config_json_default, allow_nan=False)
    )


def validate_run_config(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> None:
    """Check the experiment fields that must match when resuming.

    The caller chooses the fields. Missing saved fields compare as None so
    older runs can omit optional, disabled preprocessing settings.
    """
    if not isinstance(saved_config, Mapping):
        raise ValueError("Checkpoint has no valid experiment configuration.")
    if not isinstance(current_config, Mapping):
        raise ValueError("Current experiment configuration must be a mapping.")
    saved = config_to_jsonable(saved_config)
    current = config_to_jsonable(current_config)
    for name in keys:
        if name not in current:
            raise ValueError(f"Current experiment configuration is missing {name}.")
        if saved.get(name) != current[name]:
            raise ValueError(f"Checkpoint {name} does not match this experiment.")


def seed_torch(seed: int) -> None:
    """Seed PyTorch's default generators; seed environments separately at reset.

    Call before network construction to reproduce initial weights. This does
    not seed Python, NumPy, action spaces, or explicitly created Generators.
    """
    torch.manual_seed(seed)


def get_rng_state(
    *, device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor | None]:
    """Capture the default CPU generator and, for CUDA, the selected device.

    CPU calls do not access CUDA. The tensor-or-None CUDA field preserves the
    existing single-device checkpoint format; this is not distributed state.
    """
    device = torch.device(device)
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def set_rng_state(
    state: Mapping[str, Any], *, device: str | torch.device = "cpu",
) -> None:
    """Restore default generators onto the run's explicitly selected device.

    CPU resumes ignore saved CUDA state. CUDA resumes restore the saved
    single-device tensor onto the requested device, which need not be cuda:0.
    Restore after setup so initialization cannot consume the restored stream.
    """
    device = torch.device(device)
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=device)