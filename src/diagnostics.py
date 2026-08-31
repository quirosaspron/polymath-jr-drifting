"""Convergence and mechanism diagnostics for fair drift comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .field import compute_field_at_queries


@dataclass(frozen=True)
class FieldConsistencyReport:
    mean_cosine: float
    directional_variance: float
    signal_to_noise_ratio: float


@dataclass(frozen=True)
class KernelNeighborhoodReport:
    mean_entropy: float
    mean_effective_neighbors: float
    zero_weight_fraction: float


@dataclass(frozen=True)
class GramSpectrumReport:
    eigenvalues: np.ndarray
    effective_rank: float
    condition_number: float


def _curve_arrays(
    steps: Sequence[int], discrepancies: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    step_array = np.asarray(steps, dtype=float)
    discrepancy_array = np.asarray(discrepancies, dtype=float)
    if step_array.ndim != 1 or discrepancy_array.ndim != 1:
        raise ValueError("steps and discrepancies must be one-dimensional")
    if step_array.size != discrepancy_array.size or step_array.size < 1:
        raise ValueError("steps and discrepancies must have the same non-zero length")
    if not np.all(np.isfinite(step_array)) or not np.all(np.isfinite(discrepancy_array)):
        raise ValueError("curve values must be finite")
    if np.any(np.diff(step_array) <= 0):
        raise ValueError("steps must be strictly increasing")
    return step_array, discrepancy_array


def steps_to_threshold(
    steps: Sequence[int], discrepancies: Sequence[float], *, epsilon: float
) -> float:
    """Return the first evaluated step at or below epsilon, else infinity."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    step_array, discrepancy_array = _curve_arrays(steps, discrepancies)
    hits = np.flatnonzero(discrepancy_array <= epsilon)
    return float(step_array[hits[0]]) if hits.size else float("inf")


def discrepancy_auc(steps: Sequence[int], discrepancies: Sequence[float]) -> float:
    """Integrate discrepancy over steps; lower means good states arrived sooner."""

    step_array, discrepancy_array = _curve_arrays(steps, discrepancies)
    if step_array.size == 1:
        return 0.0
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(discrepancy_array, step_array))


def fixed_budget_discrepancy(
    steps: Sequence[int], discrepancies: Sequence[float], *, budget: int
) -> float:
    """Interpolate discrepancy at a common budget inside the measured range."""

    step_array, discrepancy_array = _curve_arrays(steps, discrepancies)
    if budget < step_array[0] or budget > step_array[-1]:
        raise ValueError("budget must lie inside the measured step range")
    return float(np.interp(float(budget), step_array, discrepancy_array))


def field_consistency(
    *,
    fixed_queries: torch.Tensor,
    positive_batch_a: torch.Tensor,
    negative_batch_a: torch.Tensor,
    positive_batch_b: torch.Tensor,
    negative_batch_b: torch.Tensor,
    metric: Any = None,
    drift_options: dict[str, Any] | None = None,
) -> FieldConsistencyReport:
    """Compare independent field estimates at identical queries.

    This is an evaluation diagnostic. A future user-authored phi loss may use a
    differentiable version of the same calculation.
    """

    options = dict(drift_options or {})
    temperature = float(options.pop("temperature", options.pop("T", 1.0)))
    if options:
        raise ValueError(f"unused drift options: {sorted(options)}")
    with torch.no_grad():
        field_a = compute_field_at_queries(
            fixed_queries,
            negative_reference=negative_batch_a,
            positive_reference=positive_batch_a,
            temperature=temperature,
            representation=metric,
        )
        field_b = compute_field_at_queries(
            fixed_queries,
            negative_reference=negative_batch_b,
            positive_reference=positive_batch_b,
            temperature=temperature,
            representation=metric,
        )
        eps = torch.finfo(field_a.dtype).eps
        cosine = torch.nn.functional.cosine_similarity(field_a, field_b, dim=1, eps=eps)
        mean_field = 0.5 * (field_a + field_b)
        noise_energy = 0.5 * (field_a - field_b).pow(2).sum(dim=1)
        signal_energy = mean_field.pow(2).sum(dim=1)
        normalized_variance = noise_energy / (signal_energy + eps)
        snr = signal_energy.mean() / (noise_energy.mean() + eps)
    return FieldConsistencyReport(
        mean_cosine=float(cosine.mean().cpu()),
        directional_variance=float(normalized_variance.mean().cpu()),
        signal_to_noise_ratio=float(snr.cpu()),
    )


def kernel_effective_neighbors(kernel_weights: Any) -> KernelNeighborhoodReport:
    """Summarize normalized kernel locality and numerical zero weights."""

    weights = torch.as_tensor(kernel_weights, dtype=torch.float64)
    if weights.ndim != 2 or weights.shape[1] < 1:
        raise ValueError("kernel_weights must have shape (queries, references)")
    if (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError("kernel weights must be finite and non-negative")
    row_mass = weights.sum(dim=1, keepdim=True)
    zero_rows = row_mass.squeeze(1) <= torch.finfo(weights.dtype).eps
    probabilities = weights / row_mass.clamp_min(torch.finfo(weights.dtype).eps)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=1)
    effective = 1.0 / probabilities.pow(2).sum(dim=1).clamp_min(1e-300)
    effective = torch.where(zero_rows, torch.zeros_like(effective), effective)
    return KernelNeighborhoodReport(
        mean_entropy=float(entropy.mean()),
        mean_effective_neighbors=float(effective.mean()),
        zero_weight_fraction=float(zero_rows.double().mean()),
    )


def gram_spectrum(gram_matrix: Any) -> GramSpectrumReport:
    """Compute eigenvalue decay, entropy effective rank, and conditioning."""

    matrix = np.asarray(
        gram_matrix.detach().cpu() if hasattr(gram_matrix, "detach") else gram_matrix,
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram_matrix must be square")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    if positive.size == 0:
        return GramSpectrumReport(eigenvalues=eigenvalues, effective_rank=0.0, condition_number=float("inf"))
    probabilities = positive / positive.sum()
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    condition_number = float(positive.max() / positive.min())
    return GramSpectrumReport(
        eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        condition_number=condition_number,
    )


def capture_parameters(model: Any) -> dict[str, torch.Tensor]:
    """Clone trainable parameters immediately before an optimizer step."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def generator_update_diagnostics(
    generator: Any,
    *,
    before_parameters: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Report parameter, gradient, and optional optimizer-update norms."""

    parameter_sq = 0.0
    gradient_sq = 0.0
    update_sq = 0.0
    for name, parameter in generator.named_parameters():
        parameter_sq += float(parameter.detach().pow(2).sum().cpu())
        if parameter.grad is not None:
            gradient_sq += float(parameter.grad.detach().pow(2).sum().cpu())
        if before_parameters is not None and name in before_parameters:
            difference = parameter.detach().cpu() - before_parameters[name]
            update_sq += float(difference.pow(2).sum())
    report = {
        "parameter_norm": parameter_sq**0.5,
        "gradient_norm": gradient_sq**0.5,
    }
    if before_parameters is not None:
        report["update_norm"] = update_sq**0.5
    return report


def plot_discrepancy_curves(
    result: Any,
    *,
    metric_name: str,
    show_individual_seeds: bool = True,
) -> Any:
    """Plot external discrepancy versus steps, grouped by method."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_discrepancy_curves requires matplotlib") from exc
    runs = result.runs if hasattr(result, "runs") else list(result)
    methods = sorted({run.method for run in runs})
    figure, axis = plt.subplots(figsize=(8, 5))
    for method in methods:
        method_runs = [run for run in runs if run.method == method]
        common_steps = np.asarray(method_runs[0].steps, dtype=float)
        curves = []
        for run in method_runs:
            if metric_name not in run.metrics:
                raise KeyError(f"{metric_name!r} is missing from {method}/{run.seed}")
            curve = np.interp(common_steps, run.steps, run.metrics[metric_name])
            curves.append(curve)
            if show_individual_seeds:
                axis.plot(common_steps, curve, alpha=0.2)
        stacked = np.vstack(curves)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        axis.plot(common_steps, mean, linewidth=2, label=method)
        axis.fill_between(common_steps, mean - std, mean + std, alpha=0.15)
    axis.set(xlabel="drift step", ylabel=metric_name, title=f"{metric_name} during drift")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis


__all__ = [
    "FieldConsistencyReport",
    "GramSpectrumReport",
    "KernelNeighborhoodReport",
    "capture_parameters",
    "discrepancy_auc",
    "field_consistency",
    "fixed_budget_discrepancy",
    "generator_update_diagnostics",
    "gram_spectrum",
    "kernel_effective_neighbors",
    "plot_discrepancy_curves",
    "steps_to_threshold",
]
