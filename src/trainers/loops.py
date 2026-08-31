"""Small, user-owned phase-level training loops.

The surrounding project supplies paired inputs, fields, evaluators, and an
EvolutionRecorder. Keeping optimizer steps here lets the researcher retain
direct control of the core algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import nn


@dataclass
class TrainingHistory:
    """Serializable scalar history plus paths to saved checkpoints."""

    metrics: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_paths: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def train_phi(
    *,
    phi: nn.Module,
    vae: nn.Module,
    generator: nn.Module | None,
    train_loader: Any,
    validation_loader: Any,
    config: Mapping[str, Any],
    recorder: Any | None = None,
) -> TrainingHistory:
    """USER IMPLEMENTATION: train phi while the VAE stays frozen.

    Call ``recorder.record(...)`` at step zero and every configured checkpoint.
    Do not build a DriftXpress cache while phi changes.
    """

    raise NotImplementedError("Implement after the known-signal toy control succeeds.")


def train_generator_with_frozen_phi(
    *,
    generator: nn.Module,
    frozen_vae: nn.Module,
    frozen_phi: nn.Module,
    paired_inputs: Any,
    config: Mapping[str, Any],
    recorder: Any | None = None,
) -> TrainingHistory:
    """USER IMPLEMENTATION: train a fresh generator with frozen VAE and phi.

    Build the representation-aware DriftXpress cache once before the loop, use
    the supplied paired tensors at each step, and call the recorder periodically.
    """

    raise NotImplementedError("Write the explicit final drift loop here.")


# Compatibility names used by the first scaffold.
train_representation = train_phi
train_drift_generator = train_generator_with_frozen_phi
