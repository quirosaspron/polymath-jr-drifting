"""High-level training loops.

The existing ``src/training.py`` remains the home of low-level objective and
gradient utilities.  This package owns complete phase-level loops.
"""

from .loops import (
    TrainingHistory,
    train_drift_generator,
    train_generator_with_frozen_phi,
    train_phi,
    train_representation,
)

__all__ = [
    "TrainingHistory",
    "train_drift_generator",
    "train_generator_with_frozen_phi",
    "train_phi",
    "train_representation",
]
