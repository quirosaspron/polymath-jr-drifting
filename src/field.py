"""Representation-aware drift fields.

This module deliberately does not replace :mod:`src.drift` or
:mod:`src.driftXpress`.  Those modules remain stable for existing notebooks.
The functions here are the experimental variants used when kernel distances
must be measured after a fixed representation ``r(z)``.

The representation changes only the kernel weights.  Attraction, repulsion,
and the returned vector field remain in the original latent coordinate system,
so the VAE decoder still receives ordinary latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


Representation = Callable[[torch.Tensor], torch.Tensor] | nn.Module | None


def identity_representation(values: torch.Tensor) -> torch.Tensor:
    """Return the original coordinates (the raw-kernel baseline)."""

    return values


def apply_representation(
    values: torch.Tensor, representation: Representation = None
) -> torch.Tensor:
    """Apply a representation and validate its sample-wise feature output."""

    features = values if representation is None else representation(values)
    if not isinstance(features, torch.Tensor):
        raise TypeError("representation must return a torch.Tensor")
    if features.ndim != 2:
        raise ValueError("representation output must have shape (batch, features)")
    if features.shape[0] != values.shape[0]:
        raise ValueError("representation must preserve the number of samples")
    if not torch.isfinite(features).all():
        raise ValueError("representation output contains NaN or infinite values")
    return features


def exponential_distance_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    temperature: float,
    representation: Representation = None,
) -> torch.Tensor:
    """Compute ``exp(-||r(x)-r(y)|| / temperature)``."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x_features = apply_representation(x, representation)
    y_features = apply_representation(y, representation)
    if x_features.shape[1] != y_features.shape[1]:
        raise ValueError("represented x and y must share their feature dimension")
    return torch.exp(-torch.cdist(x_features.float(), y_features.float()) / temperature)


def compute_exact_field(
    negative: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float,
    representation: Representation = None,
) -> torch.Tensor:
    """Representation-aware version of ``src.drift.compute_V``.

    With ``representation=None`` this follows the legacy algorithm: it applies
    the same row/column normalization and self-interaction mask, but calculates
    distances in the raw coordinates.  With a representation, only those
    distances change; weighted averages still use ``negative`` and ``positive``.
    """

    if negative.ndim != 2 or positive.ndim != 2:
        raise ValueError("negative and positive must be 2-D tensors")
    if negative.shape[1] != positive.shape[1]:
        raise ValueError("negative and positive must share their latent dimension")
    if negative.shape[0] < 1 or positive.shape[0] < 1:
        raise ValueError("negative and positive must be non-empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    n_negative = negative.shape[0]
    negative_features = apply_representation(negative, representation)
    positive_features = apply_representation(positive, representation)
    feature_points = torch.cat([negative_features, positive_features], dim=0).float()
    distances = torch.cdist(negative_features.float(), feature_points)
    distances[:, :n_negative] = distances[:, :n_negative] + torch.eye(
        n_negative, device=distances.device, dtype=distances.dtype
    ) * 1e6

    logits = -distances / temperature
    row_weights = torch.softmax(logits, dim=1)
    column_weights = torch.softmax(logits, dim=0)
    weights = torch.sqrt(row_weights * column_weights)

    negative_weights = weights[:, :n_negative]
    positive_weights = weights[:, n_negative:]
    weighted_positive = positive_weights * negative_weights.sum(dim=1, keepdim=True)
    weighted_negative = negative_weights * positive_weights.sum(dim=1, keepdim=True)
    return weighted_positive @ positive - weighted_negative @ negative


def compute_field_at_queries(
    queries: torch.Tensor,
    *,
    negative_reference: torch.Tensor,
    positive_reference: torch.Tensor,
    temperature: float,
    representation: Representation = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Evaluate a diagnostic attraction-minus-repulsion field at fixed queries.

    Unlike :func:`compute_exact_field`, queries and reference batches are
    separate.  This makes it suitable for comparing two independent field
    estimates at identical query points during representation research.
    """

    if queries.ndim != 2 or negative_reference.ndim != 2 or positive_reference.ndim != 2:
        raise ValueError("queries and reference batches must be 2-D tensors")
    latent_dims = {queries.shape[1], negative_reference.shape[1], positive_reference.shape[1]}
    if len(latent_dims) != 1:
        raise ValueError("queries and reference batches must share their latent dimension")

    positive_kernel = exponential_distance_kernel(
        queries,
        positive_reference,
        temperature=temperature,
        representation=representation,
    )
    negative_kernel = exponential_distance_kernel(
        queries,
        negative_reference,
        temperature=temperature,
        representation=representation,
    )
    attraction = (positive_kernel @ positive_reference) / (
        positive_kernel.sum(dim=1, keepdim=True) + eps
    )
    repulsion = (negative_kernel @ negative_reference) / (
        negative_kernel.sum(dim=1, keepdim=True) + eps
    )
    return attraction - repulsion


@dataclass(frozen=True)
class XpressFieldCache:
    """Positive-data quantities cached for a frozen representation."""

    landmarks: torch.Tensor
    landmark_features: torch.Tensor
    inverse_sqrt_kernel: torch.Tensor
    positive_summary: torch.Tensor
    positive_mass: torch.Tensor
    temperature: float


def build_xpress_cache(
    positive: torch.Tensor,
    *,
    landmark_indices: torch.Tensor,
    temperature: float,
    representation: Representation = None,
    jitter: float = 1e-5,
) -> XpressFieldCache:
    """Build DriftXpress summaries for one fixed representation.

    Rebuild this cache whenever the representation or positive latents change.
    The intended final experiment freezes both before calling this function.
    """

    if positive.ndim != 2 or positive.shape[0] < 1:
        raise ValueError("positive must have shape (samples, latent_dim)")
    if temperature <= 0 or jitter <= 0:
        raise ValueError("temperature and jitter must be positive")
    indices = torch.as_tensor(landmark_indices, device=positive.device, dtype=torch.long).reshape(-1)
    if indices.numel() < 1:
        raise ValueError("at least one landmark index is required")
    if indices.min().item() < 0 or indices.max().item() >= positive.shape[0]:
        raise IndexError("landmark index is outside the positive batch")

    if isinstance(representation, nn.Module):
        trainable = [parameter for parameter in representation.parameters() if parameter.requires_grad]
        if trainable:
            raise ValueError("freeze the representation before building an Xpress cache")

    with torch.no_grad():
        positive_features = apply_representation(positive, representation).detach()
        landmarks = positive[indices].detach()
        landmark_features = positive_features[indices].detach()
        landmark_kernel = torch.exp(
            -torch.cdist(landmark_features.float(), landmark_features.float()) / temperature
        )
        landmark_kernel = landmark_kernel + jitter * torch.eye(
            indices.numel(), device=positive.device, dtype=landmark_kernel.dtype
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
        eigenvalues = torch.clamp(eigenvalues, min=jitter)
        inverse_sqrt = (
            eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.T
        )
        positive_kernel = torch.exp(
            -torch.cdist(positive_features.float(), landmark_features.float()) / temperature
        )
        nystrom_features = (positive_kernel @ inverse_sqrt).T
        positive_summary = nystrom_features @ positive
        positive_mass = nystrom_features.sum(dim=1, keepdim=True)

    return XpressFieldCache(
        landmarks=landmarks,
        landmark_features=landmark_features,
        inverse_sqrt_kernel=inverse_sqrt.detach(),
        positive_summary=positive_summary.detach(),
        positive_mass=positive_mass.detach(),
        temperature=float(temperature),
    )


def build_differentiable_xpress_cache(
    positive: torch.Tensor,
    *,
    landmark_indices: torch.Tensor,
    temperature: float,
    representation: Representation = None,
    jitter: float = 1e-5,
) -> XpressFieldCache:
    """Build an Xpress cache while retaining gradients through a trainable phi.

    This path is for the offline representation-training phase only. Unlike
    :func:`build_xpress_cache`, it does not enter ``no_grad`` and does not
    detach represented target features. Because changing phi changes every
    target-side kernel quantity, this cache must be rebuilt after each phi
    optimizer step (or deliberately refreshed on a schedule). Once phi is
    frozen, use :func:`build_xpress_cache` for the fast drift loop.
    """

    if positive.ndim != 2 or positive.shape[0] < 1:
        raise ValueError("positive must have shape (samples, latent_dim)")
    if temperature <= 0 or jitter <= 0:
        raise ValueError("temperature and jitter must be positive")
    indices = torch.as_tensor(
        landmark_indices, device=positive.device, dtype=torch.long
    ).reshape(-1)
    if indices.numel() < 1:
        raise ValueError("at least one landmark index is required")
    if indices.min().item() < 0 or indices.max().item() >= positive.shape[0]:
        raise IndexError("landmark index is outside the positive batch")

    positive_features = apply_representation(positive, representation)
    landmarks = positive[indices]
    landmark_features = positive_features[indices]
    landmark_kernel = torch.exp(
        -torch.cdist(landmark_features.float(), landmark_features.float())
        / temperature
    )
    landmark_kernel = landmark_kernel + jitter * torch.eye(
        indices.numel(), device=positive.device, dtype=landmark_kernel.dtype
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
    eigenvalues = torch.clamp(eigenvalues, min=jitter)
    inverse_sqrt = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.T
    positive_kernel = torch.exp(
        -torch.cdist(positive_features.float(), landmark_features.float())
        / temperature
    )
    nystrom_features = (positive_kernel @ inverse_sqrt).T
    positive_summary = nystrom_features @ positive
    positive_mass = nystrom_features.sum(dim=1, keepdim=True)
    return XpressFieldCache(
        landmarks=landmarks,
        landmark_features=landmark_features,
        inverse_sqrt_kernel=inverse_sqrt,
        positive_summary=positive_summary,
        positive_mass=positive_mass,
        temperature=float(temperature),
    )


def compute_xpress_field(
    negative: torch.Tensor,
    *,
    cache: XpressFieldCache,
    representation: Representation = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a representation-aware DriftXpress field in original coordinates."""

    if negative.ndim != 2 or negative.shape[1] != cache.landmarks.shape[1]:
        raise ValueError("negative must match the cached latent dimension")
    negative_features = apply_representation(negative, representation)
    if negative_features.shape[1] != cache.landmark_features.shape[1]:
        raise ValueError("representation output does not match the cached features")

    landmark_kernel = torch.exp(
        -torch.cdist(negative_features.float(), cache.landmark_features.float())
        / cache.temperature
    )
    nystrom_features = (landmark_kernel @ cache.inverse_sqrt_kernel).T
    attraction_numerator = (cache.positive_summary.T @ nystrom_features).T
    attraction_denominator = nystrom_features.T @ cache.positive_mass + eps
    attraction = attraction_numerator / attraction_denominator

    negative_kernel = torch.exp(
        -torch.cdist(negative_features.float(), negative_features.float())
        / cache.temperature
    )
    negative_kernel = negative_kernel.clone()
    negative_kernel.fill_diagonal_(0.0)
    repulsion = (negative_kernel @ negative) / (
        negative_kernel.sum(dim=1, keepdim=True) + eps
    )
    return attraction - repulsion


__all__ = [
    "Representation",
    "XpressFieldCache",
    "apply_representation",
    "build_differentiable_xpress_cache",
    "build_xpress_cache",
    "compute_exact_field",
    "compute_field_at_queries",
    "compute_xpress_field",
    "exponential_distance_kernel",
    "identity_representation",
]
