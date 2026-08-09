"""Statistics exchanged between algorithms and run-level infrastructure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingStats:
    """Summary of one complete algorithm update phase.

    ``metrics`` contains values already reduced across the phase's internal
    minibatches. ``gradient_steps`` is the number of learner minibatches that
    produced those values and is used when combining several updates between
    logger dumps.
    """

    metrics: dict[str, float]
    gradient_steps: int
