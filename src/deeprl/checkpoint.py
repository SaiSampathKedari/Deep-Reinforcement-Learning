"""Save and load training checkpoints."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    checkpoint: dict[str, Any],
) -> None:
    """Atomically replace a checkpoint containing state dicts and plain metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Read a checkpoint onto the requested device."""
    return torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
