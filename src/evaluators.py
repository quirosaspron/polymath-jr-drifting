"""Frozen external evaluators used at training checkpoints.

These wrappers reuse the metric implementations in :mod:`src.eval` while
fixing reference samples, random projections, and MMD bandwidths across the
entire comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .eval import evaluate_mnist_generation, mmd_rbf, sliced_wasserstein_distance


View = Callable[[Any], Any]


def _to_numpy_2d(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("evaluation views must return shape (samples, features)")
    return array.astype(np.float64, copy=False)


def _reference_bandwidth(reference: np.ndarray, max_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if reference.shape[0] > max_samples:
        indices = rng.choice(reference.shape[0], size=max_samples, replace=False)
        reference = reference[indices]
    squared_norms = np.sum(reference * reference, axis=1, keepdims=True)
    squared_distances = np.maximum(
        squared_norms + squared_norms.T - 2.0 * (reference @ reference.T), 0.0
    )
    distances = np.sqrt(squared_distances)
    nonzero = distances[distances > 0]
    return float(np.median(nonzero)) if nonzero.size else 1.0


class DistributionEvaluator:
    """Evaluate fixed distribution metrics in one or more declared views."""

    def __init__(
        self,
        reference: Any,
        *,
        views: Mapping[str, View] | None = None,
        metrics: Sequence[str] = ("swd", "mmd"),
        random_state: int = 1234,
        swd_projections: int = 128,
        swd_quantiles: int = 256,
        mmd_max_samples: int = 2048,
    ) -> None:
        self.views = dict(views or {"observed": lambda values: values})
        self.metrics = tuple(metrics)
        unknown = set(self.metrics) - {"swd", "mmd"}
        if unknown:
            raise ValueError(f"unsupported distribution metrics: {sorted(unknown)}")
        if not self.views:
            raise ValueError("at least one evaluation view is required")
        self.random_state = int(random_state)
        self.swd_projections = int(swd_projections)
        self.swd_quantiles = int(swd_quantiles)
        self.mmd_max_samples = int(mmd_max_samples)
        self.reference_views = {
            name: _to_numpy_2d(transform(reference))
            for name, transform in self.views.items()
        }
        self.mmd_bandwidths = {
            name: _reference_bandwidth(values, self.mmd_max_samples, self.random_state)
            for name, values in self.reference_views.items()
        }

    def evaluate(self, generated: Any) -> dict[str, float]:
        """Compare generated samples with the fixed reference in every view."""

        results: dict[str, float] = {}
        for view_name, transform in self.views.items():
            generated_view = _to_numpy_2d(transform(generated))
            reference_view = self.reference_views[view_name]
            if "swd" in self.metrics:
                results[f"{view_name}_swd"] = sliced_wasserstein_distance(
                    reference_view,
                    generated_view,
                    n_projections=self.swd_projections,
                    n_quantiles=self.swd_quantiles,
                    random_state=self.random_state,
                )
            if "mmd" in self.metrics:
                results[f"{view_name}_mmd"] = mmd_rbf(
                    reference_view,
                    generated_view,
                    bandwidth=self.mmd_bandwidths[view_name],
                    max_samples=self.mmd_max_samples,
                    random_state=self.random_state,
                )
        return results

    __call__ = evaluate


def make_toy_evaluator(
    problem: Any,
    *,
    metrics: Sequence[str] = ("swd", "mmd"),
    random_state: int = 1234,
) -> DistributionEvaluator:
    """Create observed, signal, and nuisance evaluators for a toy problem."""

    views: dict[str, View] = {
        "observed": lambda values: values,
        "signal": problem.project_signal,
    }
    if problem.nuisance_projection.shape[1] > 0:
        views["nuisance"] = problem.project_nuisance
    return DistributionEvaluator(
        problem.target,
        views=views,
        metrics=metrics,
        random_state=random_state,
    )


class MNISTExternalEvaluator:
    """Hold real MNIST data and one frozen feature evaluator fixed across runs."""

    def __init__(
        self,
        real_images: Any,
        *,
        feature_evaluator: Any,
        real_labels: Any | None = None,
        n_classes: int = 10,
        random_state: int = 1234,
        max_samples: int = 2048,
    ) -> None:
        self.real_images = real_images
        self.real_labels = real_labels
        self.feature_evaluator = feature_evaluator
        self.n_classes = n_classes
        self.random_state = random_state
        self.max_samples = max_samples
        if hasattr(feature_evaluator, "eval"):
            feature_evaluator.eval()
        if hasattr(feature_evaluator, "parameters"):
            for parameter in feature_evaluator.parameters():
                parameter.requires_grad_(False)

    def evaluate(
        self, generated_images: Any, *, generated_labels: Any | None = None
    ) -> dict[str, float]:
        return evaluate_mnist_generation(
            self.real_images,
            generated_images,
            evaluator=self.feature_evaluator,
            real_labels=self.real_labels,
            generated_labels=generated_labels,
            n_classes=self.n_classes,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )


__all__ = ["DistributionEvaluator", "MNISTExternalEvaluator", "make_toy_evaluator"]
