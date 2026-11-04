"""Statistics exchanged between algorithms and run-level infrastructure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProgress:
    """Trainer-owned counters passed to logging and evaluation as a snapshot.

    ``global_step`` counts transitions across all environments.
    ``gradient_step`` counts completed learner minibatch updates, not individual
    optimizer.step() calls. ``update`` counts completed training iterations.
    """

    global_step     : int = 0
    gradient_step   : int = 0
    update          : int = 0
    
    def __post_init__(self) -> None:
        for name in ("global_step", "gradient_step", "update"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"progress {name} must be a non-negative integer, got {value!r}")
            
@dataclass(frozen=True)
class TrainingStats:
    """Summary of one complete algorithm-update phase.

    ``metrics`` contains reduced diagnostics for the phase.
    ``gradient_steps`` counts completed learner minibatch updates and may
    be zero when optimization is skipped. The logger weights summaries
    by this count, using weight one for zero-step diagnostics.
    """

    metrics: dict[str, float]
    gradient_steps: int
