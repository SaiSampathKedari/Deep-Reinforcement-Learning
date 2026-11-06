"""Configuration, random-state, and provenance helpers for training workflows."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from deeprl import __version__


def get_provenance() -> dict[str, str | bool | None]:
    """Return checkpoint-safe versions and best-effort source-checkout details.

    Git is queried only for this package's ``src/deeprl`` checkout, never the
    caller's working directory or an installed wheel's ancestor repository.
    Missing checkout metadata, Git failures, and timeouts leave Git fields None.
    """
    provenance: dict[str, str | bool | None] = {
        "deeprl_version": str(__version__),
        "torch_version": str(torch.__version__),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        package_dir = Path(__file__).resolve().parent
        root = package_dir.parent.parent
        if package_dir.parent.name != "src" or not (root / ".git").exists():
            return provenance
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return provenance
    provenance.update(git_commit=commit, git_dirty=bool(status))
    return provenance


def _config_json_default(value: Any) -> Any:
    """Encode non-JSON configuration values as descriptions, not live objects."""
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

    The caller chooses the fields. Both configurations must contain each field,
    including explicitly disabled preprocessing settings represented by None.
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
        if name not in saved:
            raise ValueError(f"Checkpoint experiment configuration is missing {name}.")
        if saved[name] != current[name]:
            raise ValueError(f"Checkpoint {name} does not match this experiment.")


def seed_torch(seed: int) -> None:
    """Seed PyTorch's default generators; seed environments separately at reset.

    Call before network construction to reproduce initial weights. This does
    not seed Python, NumPy, action spaces, or explicitly created Generators.
    """
    torch.manual_seed(seed)


def get_rng_state(
    *,
    device: str | torch.device = "cpu",
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
    state: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
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
