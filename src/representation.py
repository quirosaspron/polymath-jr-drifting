"""User-owned phi objective plus implemented representation health checks.

The experimental infrastructure intentionally does not choose the research
loss. Implement the first phi objective in :func:`compute_phi_loss`; use
``src.field.compute_field_at_queries`` if starting with direction consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class RepresentationLossReport:
    """The differentiable total plus detached named terms for logging."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RepresentationHealthReport:
    """Diagnostics that reveal collapse or extreme geometric distortion."""

    mean_feature_std: float
    minimum_feature_std: float
    pairwise_distance_correlation: float
    effective_rank: float


@dataclass
class PhiTrainingResult:
    """Loss history for the supervised signal-recovery integration check."""

    steps: list[int]
    losses: list[float]


def compute_phi_loss(
    *,
    phi: nn.Module,
    real_latents: torch.Tensor,
    generated_latents: torch.Tensor,
    context: Mapping[str, Any],
) -> RepresentationLossReport:
    """Implement the chosen phi-training loss here.

    Recommended first candidate after the known-signal control succeeds:
    directional agreement between two independently estimated fields at fixed
    queries. Keep this function small and return every term separately.
    """

    raise NotImplementedError("This research objective is intentionally user-owned.")


def train_phi_to_known_signal(
    *,
    phi: nn.Module,
    inputs: torch.Tensor,
    target_features: torch.Tensor,
    steps: int = 500,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> PhiTrainingResult:
    """Train phi to reproduce known toy signal coordinates.

    This is deliberately a supervised integration sanity check, not the
    proposed research objective. It answers whether a neural representation
    can be trained, frozen, and passed through the Xpress field correctly
    before an unsupervised drift-aware loss is introduced.
    """

    if inputs.ndim != 2 or target_features.ndim != 2:
        raise ValueError("inputs and target_features must be two-dimensional")
    if inputs.shape[0] != target_features.shape[0] or inputs.shape[0] < 2:
        raise ValueError("inputs and target_features must share at least two samples")
    if steps < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("steps must be positive and optimizer values valid")
    if batch_size is not None and not 1 <= batch_size <= inputs.shape[0]:
        raise ValueError("batch_size must be between one and the sample count")

    try:
        device = next(phi.parameters()).device
    except StopIteration as exc:
        raise ValueError("phi must contain trainable parameters") from exc
    inputs = inputs.to(device)
    target_features = target_features.to(device)
    optimizer = torch.optim.Adam(
        phi.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    was_training = phi.training
    phi.train()
    history_steps: list[int] = []
    history_losses: list[float] = []
    for step in range(steps):
        if batch_size is None or batch_size == inputs.shape[0]:
            indices = slice(None)
        else:
            indices = torch.randperm(inputs.shape[0], generator=generator)[:batch_size].to(device)
        prediction = phi(inputs[indices])
        loss = torch.nn.functional.mse_loss(prediction, target_features[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history_steps.append(step + 1)
        history_losses.append(float(loss.detach().cpu()))
    if not was_training:
        phi.eval()
    return PhiTrainingResult(steps=history_steps, losses=history_losses)


def validate_representation(
    *,
    phi: nn.Module,
    fixed_latents: torch.Tensor,
    max_pairwise_samples: int = 512,
) -> RepresentationHealthReport:
    """Measure feature spread, distance preservation, and covariance rank."""

    if fixed_latents.ndim != 2 or fixed_latents.shape[0] < 2:
        raise ValueError("fixed_latents must have shape (samples >= 2, latent_dim)")
    was_training = phi.training
    phi.eval()
    with torch.no_grad():
        features = phi(fixed_latents)
        if features.ndim != 2 or features.shape[0] != fixed_latents.shape[0]:
            raise ValueError("metric must return shape (samples, features)")
        feature_std = features.std(dim=0, unbiased=True)
        sample_count = min(fixed_latents.shape[0], max_pairwise_samples)
        latents_subset = fixed_latents[:sample_count]
        features_subset = features[:sample_count]
        latent_distances = torch.pdist(latents_subset.float())
        feature_distances = torch.pdist(features_subset.float())
        if latent_distances.std() <= 1e-12 or feature_distances.std() <= 1e-12:
            distance_correlation = 0.0
        else:
            stacked = torch.stack([latent_distances, feature_distances])
            distance_correlation = float(torch.corrcoef(stacked)[0, 1].cpu())

        centered = features.float() - features.float().mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, features.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        positive = eigenvalues[eigenvalues > torch.finfo(eigenvalues.dtype).eps]
        if positive.numel() == 0:
            effective_rank = 0.0
        else:
            probabilities = positive / positive.sum()
            effective_rank = float(
                torch.exp(-(probabilities * probabilities.log()).sum()).cpu()
            )
    if was_training:
        phi.train()
    return RepresentationHealthReport(
        mean_feature_std=float(feature_std.mean().cpu()),
        minimum_feature_std=float(feature_std.min().cpu()),
        pairwise_distance_correlation=distance_correlation,
        effective_rank=effective_rank,
    )


def freeze_representation(phi: nn.Module) -> nn.Module:
    """Freeze phi before building its DriftXpress cache."""

    phi.eval()
    for parameter in phi.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return phi


__all__ = [
    "PhiTrainingResult",
    "RepresentationHealthReport",
    "RepresentationLossReport",
    "compute_phi_loss",
    "freeze_representation",
    "train_phi_to_known_signal",
    "validate_representation",
]
