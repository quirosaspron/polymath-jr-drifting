"""Small evaluation and visualization helpers for the VAE experiments.

Typical notebook usage:

    from src.eval import (
        plot_loss_history,
        evaluate_generation,
        pca_embedding,
        umap_embedding,
        mutual_information_gap,
        total_correlation,
    )

Most distribution metrics expect feature vectors, not raw images. For the
MNIST-specific evaluator below, pixel-space metrics are computed on flattened
images in [0, 1], while feature-space metrics use one frozen classifier
trained on the real MNIST training set.

Example:

    history = {
        "train_losses": train_losses,
        "recon_losses": recon_losses,
        "kl_losses": kl_losses,
        "drift_losses": drift_losses,
        "var_losses": var_losses,
        "covar_losses": covar_losses,
        "average_V": average_V,
    }
    fig, axes = plot_loss_history(history)

    metrics = evaluate_generation(
        real_features,
        generated_features,
        generated_labels=generated_labels,
        n_classes=10,
        coverage_min_count=5,
    )
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy arrays, PyTorch tensors, or array-like values to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_2d(value: Any, name: str = "values") -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features).")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _as_image_matrix(value: Any, name: str = "images") -> np.ndarray:
    """Convert image batches to flattened floating-point pixels in [0, 1]."""
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.ndim < 2:
        raise ValueError(f"{name} must contain a batch of images.")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    array = array.reshape(array.shape[0], -1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if array.min() < 0:
        raise ValueError(f"{name} must be non-negative image data.")
    if array.max() > 1.0 + 1e-6:
        if array.max() <= 255.0 + 1e-6:
            array = array / 255.0
        else:
            raise ValueError(f"{name} must be scaled to [0, 1] or [0, 255].")
    return array


def _subsample_rows(
    values: np.ndarray,
    max_samples: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_samples is None or values.shape[0] <= max_samples:
        return values
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2 or None.")
    indices = rng.choice(values.shape[0], size=max_samples, replace=False)
    return values[indices]


def _mean_pairwise_distance(
    values: Any,
    max_samples: int = 512,
    random_state: int | None = 0,
) -> float:
    array = _as_2d(values, "values")
    rng = np.random.default_rng(random_state)
    array = _subsample_rows(array, max_samples, rng)
    if array.shape[0] < 2:
        return 0.0
    distances = np.sqrt(_pairwise_squared_distances(array, array))
    upper = distances[np.triu_indices(array.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _check_pair(real: Any, generated: Any) -> tuple[np.ndarray, np.ndarray]:
    real_array = _as_2d(real, "real")
    generated_array = _as_2d(generated, "generated")
    if real_array.shape[1] != generated_array.shape[1]:
        raise ValueError("real and generated must have the same feature dimension.")
    return real_array, generated_array


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared distances without a 3-D broadcast tensor."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * (x @ y.T)
    )
    return np.maximum(distances, 0.0)


def _matrix_sqrt_psd(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Square root of a symmetric positive-semidefinite matrix."""
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, eps, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _labels_to_codes(labels: Any) -> tuple[np.ndarray, np.ndarray]:
    labels_array = _to_numpy(labels).reshape(-1)
    codes, unique = np.unique(labels_array, return_inverse=True)
    return unique.astype(int), codes


def _discretize(values: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Discretize one continuous variable using quantile bins."""
    values = _to_numpy(values).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot discretize an empty array.")
    if np.issubdtype(values.dtype, np.integer) or np.unique(values).size <= n_bins:
        return np.unique(values, return_inverse=True)[1].astype(int)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values.astype(float), quantiles))
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=int)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _entropy_from_codes(codes: np.ndarray) -> float:
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib in the notebook environment."
        ) from exc
    return plt


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def sliced_wasserstein_distance(
    real: Any,
    generated: Any,
    n_projections: int = 128,
    n_quantiles: int = 256,
    p: int = 1,
    random_state: int | None = 0,
) -> float:
    """Estimate sliced Wasserstein distance between two feature distributions.

    Both inputs should have shape (n_samples, n_features). The same
    projection directions are used for both distributions.
    """
    if n_projections < 1 or n_quantiles < 2:
        raise ValueError("n_projections must be positive and n_quantiles >= 2.")
    if p < 1:
        raise ValueError("p must be at least 1.")

    real_array, generated_array = _check_pair(real, generated)
    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(real_array.shape[1], n_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12

    real_projection = real_array @ directions
    generated_projection = generated_array @ directions
    quantiles = np.linspace(0.0, 1.0, n_quantiles)

    real_quantiles = np.quantile(real_projection, quantiles, axis=0)
    generated_quantiles = np.quantile(generated_projection, quantiles, axis=0)
    distances = np.abs(real_quantiles - generated_quantiles) ** p

    return float(np.mean(np.mean(distances, axis=0) ** (1.0 / p)))


def mmd_rbf(
    real: Any,
    generated: Any,
    bandwidth: float | None = None,
    max_samples: int | None = 2048,
    random_state: int | None = 0,
) -> float:
    """Compute biased RBF-kernel MMD squared.

    If bandwidth is omitted, the median pairwise distance from the combined
    sample is used as a simple heuristic.
    """
    real_array, generated_array = _check_pair(real, generated)
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2 or None.")
        rng = np.random.default_rng(random_state)

        def subsample(values: np.ndarray) -> np.ndarray:
            if values.shape[0] <= max_samples:
                return values
            indices = rng.choice(values.shape[0], size=max_samples, replace=False)
            return values[indices]

        real_array = subsample(real_array)
        generated_array = subsample(generated_array)
    combined = np.concatenate([real_array, generated_array], axis=0)
    pairwise = np.sqrt(_pairwise_squared_distances(combined, combined))
    nonzero = pairwise[pairwise > 0]
    sigma = float(np.median(nonzero)) if bandwidth is None and nonzero.size else bandwidth
    sigma = 1.0 if sigma is None or sigma <= 0 else float(sigma)

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-_pairwise_squared_distances(x, y) / (2.0 * sigma**2))

    value = (
        np.mean(kernel(real_array, real_array))
        + np.mean(kernel(generated_array, generated_array))
        - 2.0 * np.mean(kernel(real_array, generated_array))
    )
    return float(max(0.0, value))


def frechet_distance(
    real_features: Any,
    generated_features: Any,
    eps: float = 1e-6,
) -> float:
    """Compute FID-style Frechet distance between two feature distributions.

    Pass features from a fixed evaluator. This function does not contain an
    image feature extractor, so raw images should not be passed directly.
    """
    real_array, generated_array = _check_pair(real_features, generated_features)
    mu_real = real_array.mean(axis=0)
    mu_generated = generated_array.mean(axis=0)

    def covariance(features: np.ndarray) -> np.ndarray:
        if features.shape[0] < 2:
            return np.zeros((features.shape[1], features.shape[1]))
        return np.atleast_2d(np.cov(features, rowvar=False))

    cov_real = covariance(real_array)
    cov_generated = covariance(generated_array)
    dimension = cov_real.shape[0]
    cov_real = cov_real + eps * np.eye(dimension)
    cov_generated = cov_generated + eps * np.eye(dimension)

    sqrt_real = _matrix_sqrt_psd(cov_real)
    cov_product = sqrt_real @ cov_generated @ sqrt_real
    covariance_term = _matrix_sqrt_psd(cov_product)
    mean_term = np.sum((mu_real - mu_generated) ** 2)

    score = mean_term + np.trace(cov_real + cov_generated - 2.0 * covariance_term)
    return float(max(0.0, score))


def fid(
    real_features: Any,
    generated_features: Any,
    eps: float = 1e-6,
) -> float:
    """Short alias for frechet_distance."""
    return frechet_distance(real_features, generated_features, eps=eps)


def pixel_fid(
    real_images: Any,
    generated_images: Any,
    eps: float = 1e-6,
    max_samples: int | None = 2048,
    random_state: int | None = 0,
) -> float:
    """Compute Fréchet distance directly on final image pixels.

    Images may be shaped as (n, 28, 28), (n, 1, 28, 28), or (n, 784), and
    may be scaled either to [0, 1] or [0, 255]. This is intentionally a
    pixel-space diagnostic; it is not a replacement for a learned-feature
    metric when semantic quality is the main concern.
    """
    real_array = _as_image_matrix(real_images, "real_images")
    generated_array = _as_image_matrix(generated_images, "generated_images")
    if real_array.shape[1] != generated_array.shape[1]:
        raise ValueError("real_images and generated_images must have the same shape.")
    rng = np.random.default_rng(random_state)
    real_array = _subsample_rows(real_array, max_samples, rng)
    generated_array = _subsample_rows(generated_array, max_samples, rng)
    return frechet_distance(real_array, generated_array, eps=eps)


def _as_image_tensor(value: Any, device: Any = None):
    """Convert image data to a float tensor with shape (n, pixels)."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("MNIST evaluation requires PyTorch.") from exc

    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.float()
    if tensor.ndim < 2:
        raise ValueError("images must contain a batch of images.")
    tensor = tensor.reshape(tensor.shape[0], -1)
    if not torch.isfinite(tensor).all():
        raise ValueError("images contains NaN or infinite values.")
    if tensor.min().item() < 0:
        raise ValueError("images must be non-negative image data.")
    if tensor.max().item() > 1.0 + 1e-6:
        if tensor.max().item() <= 255.0 + 1e-6:
            tensor = tensor / 255.0
        else:
            raise ValueError("images must be scaled to [0, 1] or [0, 255].")
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def make_mnist_feature_evaluator(n_classes: int = 10, feature_dim: int = 64):
    """Create a small classifier whose penultimate layer is MNIST features."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("MNIST evaluation requires PyTorch.") from exc

    class _Evaluator(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, feature_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(feature_dim, n_classes)

        def extract_features(self, images):
            images = images.reshape(images.shape[0], -1)
            return self.feature_extractor(images)

        def forward(self, images):
            features = self.extract_features(images)
            return self.classifier(features), features

    return _Evaluator()


def fit_mnist_evaluator(
    images: Any,
    labels: Any,
    n_classes: int,
    feature_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    device: Any = None,
    random_state: int = 1234,
):
    """Train a fixed MNIST classifier for fair cross-model evaluation.

    Train this evaluator once and reuse the returned model for every
    algorithm. The classifier is not part of any generator's optimization.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("MNIST evaluation requires PyTorch.") from exc

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    torch.manual_seed(random_state)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_tensor = _as_image_tensor(images)
    label_tensor = torch.as_tensor(_to_numpy(labels).reshape(-1), dtype=torch.long)
    if image_tensor.shape[0] != label_tensor.shape[0]:
        raise ValueError("images and labels must contain the same number of samples.")
    if label_tensor.min().item() < 0 or label_tensor.max().item() >= n_classes:
        raise ValueError("labels must be integer class codes in [0, n_classes).")

    evaluator = make_mnist_feature_evaluator(
        n_classes=n_classes,
        feature_dim=feature_dim,
    )
    evaluator = evaluator.to(target_device)
    evaluator.train()
    dataset = TensorDataset(image_tensor, label_tensor)
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(target_device, non_blocking=True)
            batch_labels = batch_labels.to(target_device, non_blocking=True)
            logits, _ = evaluator(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    evaluator.eval()
    return evaluator


def _mnist_evaluator_outputs(
    evaluator: Any,
    images: Any,
    batch_size: int = 1024,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("MNIST evaluation requires PyTorch.") from exc

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model_device = device
    if model_device is None:
        model_device = next(evaluator.parameters()).device
    model_device = torch.device(model_device)
    image_tensor = _as_image_tensor(images)
    loader = DataLoader(
        TensorDataset(image_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits_list, feature_list, prediction_list = [], [], []
    evaluator.eval()
    with torch.no_grad():
        for (batch_images,) in loader:
            logits, features = evaluator(batch_images.to(model_device, non_blocking=True))
            logits_list.append(logits.cpu())
            feature_list.append(features.cpu())
            prediction_list.append(logits.argmax(dim=1).cpu())
    return (
        torch.cat(logits_list).numpy(),
        torch.cat(feature_list).numpy(),
        torch.cat(prediction_list).numpy(),
    )


def evaluate_mnist_generation(
    real_images: Any,
    generated_images: Any,
    evaluator: Any | None = None,
    real_labels: Any | None = None,
    generated_labels: Any | None = None,
    n_classes: int | None = None,
    max_samples: int | None = 2048,
    mmd_max_samples: int | None = 1024,
    random_state: int | None = 0,
    evaluator_batch_size: int = 1024,
) -> dict[str, float]:
    """Evaluate MNIST generations in pixel and fixed classifier spaces.

    The primary pixel metric is ``pixel_fid``. When a frozen evaluator from
    :func:`fit_mnist_evaluator` is supplied, ``mnist_fid`` and ``mnist_mmd``
    measure distribution match in a common learned MNIST feature space, and
    classifier accuracy/coverage measure conditional correctness and mode
    coverage. All returned values are scalar and suitable for a comparison
    table.
    """
    real_pixels = _as_image_matrix(real_images, "real_images")
    generated_pixels = _as_image_matrix(generated_images, "generated_images")
    if real_pixels.shape[1] != generated_pixels.shape[1]:
        raise ValueError("real_images and generated_images must have the same shape.")

    rng = np.random.default_rng(random_state)
    real_sample = _subsample_rows(real_pixels, max_samples, rng)
    generated_sample = _subsample_rows(generated_pixels, max_samples, rng)
    metrics: dict[str, float] = {
        "pixel_fid": frechet_distance(real_sample, generated_sample),
        "pixel_mmd_rbf": mmd_rbf(
            real_sample,
            generated_sample,
            max_samples=mmd_max_samples,
            random_state=random_state,
        ),
        "pixel_sliced_wasserstein": sliced_wasserstein_distance(
            real_sample,
            generated_sample,
            random_state=random_state,
        ),
    }

    real_label_array = None
    generated_label_array = None
    if real_labels is not None:
        real_label_array = _to_numpy(real_labels).reshape(-1).astype(int)
        if real_label_array.shape[0] != real_pixels.shape[0]:
            raise ValueError("real_labels must match real_images.")
    if generated_labels is not None:
        generated_label_array = _to_numpy(generated_labels).reshape(-1).astype(int)
        if generated_label_array.shape[0] != generated_pixels.shape[0]:
            raise ValueError("generated_labels must match generated_images.")
    if n_classes is not None and n_classes < 1:
        raise ValueError("n_classes must be positive.")

    if generated_label_array is not None:
        metrics["requested_class_coverage"] = class_coverage(
            generated_label_array, n_classes=n_classes
        )
        metrics["requested_class_entropy"] = class_entropy(
            generated_label_array, n_classes=n_classes, normalized=True
        )

    real_features = generated_features = None
    real_predictions = generated_predictions = None
    if evaluator is not None:
        _, real_features, real_predictions = _mnist_evaluator_outputs(
            evaluator, real_images, batch_size=evaluator_batch_size
        )
        _, generated_features, generated_predictions = _mnist_evaluator_outputs(
            evaluator, generated_images, batch_size=evaluator_batch_size
        )
        real_feature_sample = _subsample_rows(real_features, max_samples, rng)
        generated_feature_sample = _subsample_rows(generated_features, max_samples, rng)
        metrics["mnist_fid"] = frechet_distance(real_feature_sample, generated_feature_sample)
        metrics["mnist_mmd_rbf"] = mmd_rbf(
            real_feature_sample,
            generated_feature_sample,
            max_samples=mmd_max_samples,
            random_state=random_state,
        )
        metrics["real_classifier_accuracy"] = (
            float(np.mean(real_predictions == real_label_array))
            if real_label_array is not None else float("nan")
        )
        if generated_label_array is not None:
            metrics["conditional_accuracy"] = float(
                np.mean(generated_predictions == generated_label_array)
            )
        metrics["predicted_class_coverage"] = class_coverage(
            generated_predictions, n_classes=n_classes
        )
        metrics["predicted_class_entropy"] = class_entropy(
            generated_predictions, n_classes=n_classes, normalized=True
        )
        real_diversity = _mean_pairwise_distance(
            real_features, random_state=random_state
        )
        generated_diversity = _mean_pairwise_distance(
            generated_features, random_state=random_state
        )
        metrics["real_feature_diversity"] = real_diversity
        metrics["generated_feature_diversity"] = generated_diversity
        metrics["feature_diversity_ratio"] = (
            generated_diversity / real_diversity if real_diversity > 1e-12 else float("nan")
        )

    if (
        n_classes is not None
        and real_label_array is not None
        and generated_label_array is not None
    ):
        for class_id in range(n_classes):
            real_mask = real_label_array == class_id
            generated_mask = generated_label_array == class_id
            if not real_mask.any() or not generated_mask.any():
                continue
            real_class_pixels = real_pixels[real_mask]
            generated_class_pixels = generated_pixels[generated_mask]
            if real_class_pixels.shape[0] >= 2 and generated_class_pixels.shape[0] >= 2:
                metrics[f"pixel_fid_class_{class_id}"] = pixel_fid(
                    real_class_pixels,
                    generated_class_pixels,
                    max_samples=max_samples,
                    random_state=random_state,
                )
            if evaluator is not None and real_features is not None and generated_features is not None:
                real_class_features = real_features[real_mask]
                generated_class_features = generated_features[generated_mask]
                if real_class_features.shape[0] >= 2 and generated_class_features.shape[0] >= 2:
                    metrics[f"mnist_fid_class_{class_id}"] = frechet_distance(
                        real_class_features,
                        generated_class_features,
                    )
                metrics[f"conditional_accuracy_class_{class_id}"] = float(
                    np.mean(generated_predictions[generated_mask] == class_id)
                )
            metrics[f"generated_count_class_{class_id}"] = float(generated_mask.sum())

    return metrics


def class_coverage(
    labels: Any,
    n_classes: int | None = None,
    min_count: int = 1,
) -> float:
    """Return the fraction of classes represented by at least min_count samples."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1.")
    codes, unique_labels = _labels_to_codes(labels)

    if n_classes is None:
        n_classes = len(unique_labels)
        counts = np.bincount(codes, minlength=n_classes)
    else:
        if n_classes < 1:
            raise ValueError("n_classes must be positive.")
        raw_labels = _to_numpy(labels).reshape(-1)
        if np.issubdtype(raw_labels.dtype, np.integer):
            counts = np.bincount(raw_labels.astype(int), minlength=n_classes)[:n_classes]
        else:
            counts = np.bincount(codes, minlength=n_classes)[:n_classes]

    return float(np.mean(counts >= min_count))


def class_entropy(
    labels_or_probabilities: Any,
    n_classes: int | None = None,
    normalized: bool = True,
) -> float:
    """Return entropy of predicted class frequencies.

    Input may be a label vector or an (n_samples, n_classes) probability
    matrix. Normalized entropy is in [0, 1].
    """
    values = _to_numpy(labels_or_probabilities)
    if values.ndim == 2:
        if values.shape[1] < 2:
            raise ValueError("Probability input needs at least two classes.")
        probabilities = np.clip(values.astype(float), 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True) + 1e-12
        probabilities = probabilities.mean(axis=0)
    else:
        codes, unique_labels = _labels_to_codes(values)
        n_classes = len(unique_labels) if n_classes is None else n_classes
        counts = np.bincount(codes, minlength=n_classes)[:n_classes]
        probabilities = counts / max(1, counts.sum())

    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    if normalized:
        divisor = np.log(len(probabilities))
        return float(0.0 if divisor <= 0 else entropy / divisor)
    return float(entropy)


def evaluate_generation(
    real_features: Any,
    generated_features: Any,
    generated_labels: Any | None = None,
    n_classes: int | None = None,
    coverage_min_count: int = 1,
    random_state: int | None = 0,
    n_projections: int = 128,
    mmd_max_samples: int | None = 2048,
    include_mmd: bool = True,
    include_fid: bool = True,
) -> dict[str, float]:
    """Compute SWD, optional MMD/FID, and optional class metrics."""
    metrics = {
        "sliced_wasserstein": sliced_wasserstein_distance(
            real_features,
            generated_features,
            n_projections=n_projections,
            random_state=random_state,
        ),
    }
    if include_mmd:
        metrics["mmd_rbf"] = mmd_rbf(
            real_features,
            generated_features,
            max_samples=mmd_max_samples,
            random_state=random_state,
        )
    if include_fid:
        metrics["fid"] = frechet_distance(real_features, generated_features)

    if generated_labels is not None:
        metrics["class_coverage"] = class_coverage(
            generated_labels,
            n_classes=n_classes,
            min_count=coverage_min_count,
        )
        metrics["class_entropy"] = class_entropy(
            generated_labels,
            n_classes=n_classes,
            normalized=True,
        )
    return metrics


# ---------------------------------------------------------------------------
# Reusable VAE collection and evaluation
# ---------------------------------------------------------------------------

def collect_latent_samples(
    model: Any,
    data_loader: Any,
    *,
    device: Any | None = None,
    n_samples: int | None = None,
    n_classes: int | None = None,
    class_to_label: Any | None = None,
    random_state: int | None = 0,
) -> dict[str, np.ndarray]:
    """Collect real encoder means and equally many conditional generations.

    The loader must yield ``(images, class_ids)``.  ``class_ids`` are the
    model's internal labels; ``class_to_label`` optionally maps them back to
    original digit labels for plots.  Returned arrays are CPU NumPy arrays.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("collect_latent_samples requires PyTorch.") from exc

    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be positive or None.")
    if n_classes is None:
        n_classes = getattr(model, "num_classes", None)
    if n_classes is None or int(n_classes) < 1:
        raise ValueError("Pass n_classes or give the model a positive num_classes.")
    n_classes = int(n_classes)

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    was_training = bool(getattr(model, "training", False))
    real_latent_batches = []
    real_image_batches = []
    real_class_batches = []
    collected = 0

    model.eval()
    with torch.no_grad():
        for images, class_ids in data_loader:
            if n_samples is not None and collected >= n_samples:
                break
            images = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            class_ids = class_ids.to(target_device, non_blocking=True).long()
            take = images.shape[0]
            if n_samples is not None:
                take = min(take, n_samples - collected)
            if take <= 0:
                break
            images = images[:take]
            class_ids = class_ids[:take]
            mu, _, _ = model.get_latent(images)
            real_latent_batches.append(mu.cpu())
            real_image_batches.append(images.cpu())
            real_class_batches.append(class_ids.cpu())
            collected += take

        if not real_latent_batches:
            raise ValueError("data_loader did not yield any samples.")

        total = int(sum(batch.shape[0] for batch in real_latent_batches))
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=target_device).manual_seed(random_state)
        noise = torch.randn(
            total,
            int(model.latent_dim),
            device=target_device,
            generator=generator,
        )
        generated_class_ids = torch.randint(
            0,
            n_classes,
            (total,),
            device=target_device,
            generator=generator,
        )
        generated_latents, generated_images = model.generate(noise, generated_class_ids)

    if was_training:
        model.train()

    real_class_ids = torch.cat(real_class_batches).numpy()
    generated_class_ids_array = generated_class_ids.cpu().numpy()
    label_lookup = None if class_to_label is None else _to_numpy(class_to_label)
    if label_lookup is not None:
        if label_lookup.ndim != 1 or label_lookup.shape[0] < n_classes:
            raise ValueError("class_to_label must map every internal class id.")
        real_labels = label_lookup[real_class_ids]
        generated_labels = label_lookup[generated_class_ids_array]
    else:
        real_labels = real_class_ids.copy()
        generated_labels = generated_class_ids_array.copy()

    return {
        "real_latents": torch.cat(real_latent_batches).numpy(),
        "generated_latents": generated_latents.cpu().numpy(),
        "real_images": torch.cat(real_image_batches).numpy(),
        "generated_images": generated_images.cpu().numpy(),
        "real_class_ids": real_class_ids,
        "generated_class_ids": generated_class_ids_array,
        "real_labels": np.asarray(real_labels),
        "generated_labels": np.asarray(generated_labels),
    }


def latent_statistics(
    real_latents: Any,
    generated_latents: Any,
    *,
    factors: Any | None = None,
) -> dict[str, float]:
    """Summarize latent location, scale, and optional disentanglement scores."""
    real_array, generated_array = _check_pair(real_latents, generated_latents)
    statistics = {
        "real_latent_mean": float(real_array.mean()),
        "generated_latent_mean": float(generated_array.mean()),
        "real_latent_std": float(real_array.std(axis=0).mean()),
        "generated_latent_std": float(generated_array.std(axis=0).mean()),
        "generated_to_real_std_ratio": float(
            generated_array.std(axis=0).mean()
            / max(real_array.std(axis=0).mean(), 1e-12)
        ),
        "real_total_correlation": total_correlation(real_array),
        "generated_total_correlation": total_correlation(generated_array),
    }
    if factors is not None:
        statistics["real_mig"] = mutual_information_gap(real_array, factors)
    return statistics


def evaluate_latent_model(
    model: Any,
    data_loader: Any,
    *,
    device: Any | None = None,
    n_samples: int | None = None,
    n_classes: int | None = None,
    class_to_label: Any | None = None,
    random_state: int | None = 0,
    n_projections: int = 128,
    include_mmd: bool = False,
    include_fid: bool = True,
) -> dict[str, Any]:
    """Run the common latent-space post-training evaluation in one call.

    The returned ``metrics['fid']`` is Fréchet distance in latent space.  For
    image FID with a frozen MNIST evaluator, use :func:`evaluate_mnist_generation`.
    """
    samples = collect_latent_samples(
        model,
        data_loader,
        device=device,
        n_samples=n_samples,
        n_classes=n_classes,
        class_to_label=class_to_label,
        random_state=random_state,
    )
    metrics = evaluate_generation(
        samples["real_latents"],
        samples["generated_latents"],
        generated_labels=samples["generated_class_ids"],
        n_classes=n_classes,
        random_state=random_state,
        n_projections=n_projections,
        include_mmd=include_mmd,
        include_fid=include_fid,
    )
    return {
        "samples": samples,
        "metrics": metrics,
        "statistics": latent_statistics(
            samples["real_latents"],
            samples["generated_latents"],
        ),
    }


def evaluate_per_class_generation(
    real_features: Any,
    generated_features: Any,
    real_class_ids: Any,
    generated_class_ids: Any,
    *,
    n_classes: int,
    **metric_kwargs: Any,
) -> dict[int, dict[str, float]]:
    """Compute the same feature metrics separately for each non-empty class."""
    real_array, generated_array = _check_pair(real_features, generated_features)
    real_ids = _to_numpy(real_class_ids).reshape(-1).astype(int)
    generated_ids = _to_numpy(generated_class_ids).reshape(-1).astype(int)
    if real_ids.shape[0] != real_array.shape[0]:
        raise ValueError("real_class_ids must match real_features.")
    if generated_ids.shape[0] != generated_array.shape[0]:
        raise ValueError("generated_class_ids must match generated_features.")
    if n_classes < 1:
        raise ValueError("n_classes must be positive.")

    results: dict[int, dict[str, float]] = {}
    for class_id in range(n_classes):
        real_mask = real_ids == class_id
        generated_mask = generated_ids == class_id
        if real_mask.sum() < 2 or generated_mask.sum() < 2:
            continue
        results[class_id] = evaluate_generation(
            real_array[real_mask],
            generated_array[generated_mask],
            **metric_kwargs,
        )
    return results


# ---------------------------------------------------------------------------
# Latent-space metrics
# ---------------------------------------------------------------------------

def mutual_information(x: Any, y: Any, n_bins: int = 20) -> float:
    """Estimate mutual information after discretizing continuous inputs."""
    x_codes = _discretize(_to_numpy(x), n_bins=n_bins)
    y_codes = _discretize(_to_numpy(y), n_bins=n_bins)
    if x_codes.shape[0] != y_codes.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    _, x_inverse = np.unique(x_codes, return_inverse=True)
    _, y_inverse = np.unique(y_codes, return_inverse=True)
    joint = np.zeros((x_inverse.max() + 1, y_inverse.max() + 1), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])))


def mutual_information_gap(
    latents: Any,
    factors: Any,
    n_bins: int = 20,
    return_per_factor: bool = False,
) -> float | dict[str, Any]:
    """Estimate MIG using known factors or labels.

    latents has shape (n_samples, latent_dim). factors has shape
    (n_samples,) or (n_samples, n_factors). A higher score indicates that each
    factor is concentrated in fewer latent dimensions. This metric requires
    meaningful observed factors or labels.
    """
    latent_array = _as_2d(latents, "latents")
    factor_array = _to_numpy(factors)
    if factor_array.ndim == 1:
        factor_array = factor_array[:, None]
    if factor_array.ndim != 2 or factor_array.shape[0] != latent_array.shape[0]:
        raise ValueError("factors must have shape (n_samples,) or (n_samples, n_factors).")

    latent_bins = np.column_stack(
        [_discretize(latent_array[:, j], n_bins=n_bins) for j in range(latent_array.shape[1])]
    )
    scores = []

    for factor_index in range(factor_array.shape[1]):
        factor_bins = _discretize(factor_array[:, factor_index], n_bins=n_bins)
        factor_entropy = _entropy_from_codes(factor_bins)
        if factor_entropy <= 1e-12:
            continue

        mutual_informations = np.array(
            [
                mutual_information(factor_bins, latent_bins[:, j], n_bins=n_bins)
                for j in range(latent_array.shape[1])
            ]
        )
        top_two = np.sort(mutual_informations)[-2:]
        scores.append(float((top_two[-1] - top_two[-2]) / factor_entropy))

    value = float(np.mean(scores)) if scores else float("nan")
    if return_per_factor:
        return {"mig": value, "per_factor": scores}
    return value


def total_correlation(latents: Any, eps: float = 1e-6) -> float:
    """Estimate total correlation with a Gaussian covariance approximation.

    TC is zero when latent coordinates are independent under this
    approximation. Lower values indicate less linear dependence, but TC alone
    does not prove semantic disentanglement.
    """
    latent_array = _as_2d(latents, "latents")
    if latent_array.shape[0] < 2:
        raise ValueError("At least two latent samples are required.")

    covariance = np.atleast_2d(np.cov(latent_array, rowvar=False))
    covariance = covariance + eps * np.eye(covariance.shape[0])
    diagonal = np.diag(np.diag(covariance))

    _, logdet_covariance = np.linalg.slogdet(covariance)
    _, logdet_diagonal = np.linalg.slogdet(diagonal)
    return float(max(0.0, 0.5 * (logdet_diagonal - logdet_covariance)))


# ---------------------------------------------------------------------------
# Dimensionality reduction and visualization
# ---------------------------------------------------------------------------

def pca_embedding(latents: Any, n_components: int = 2) -> np.ndarray:
    """Return a PCA projection using only NumPy."""
    latent_array = _as_2d(latents, "latents")
    if not 1 <= n_components <= min(latent_array.shape):
        raise ValueError("n_components must fit the number of samples and dimensions.")

    centered = latent_array - latent_array.mean(axis=0, keepdims=True)
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ right_singular_vectors[:n_components].T


def umap_embedding(
    latents: Any,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Return a UMAP projection using the optional umap-learn package."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requires the optional package umap-learn."
        ) from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs,
    )
    return np.asarray(reducer.fit_transform(_as_2d(latents, "latents")))


def plot_loss_history(
    history: Mapping[str, Sequence[float]],
    keys: Sequence[str] = ("total", "recon", "kl", "drift", "var", "covar", "force"),
    log_scale: bool = True,
    figsize: tuple[float, float] = (10.0, 14.0),
):
    """Plot the loss histories used in the notebook training loops."""
    plt = _import_matplotlib()
    aliases = {
        "total": ("total", "train_losses"),
        "recon": ("recon", "recon_losses"),
        "kl": ("kl", "kl_losses"),
        "drift": ("drift", "drift_losses"),
        "var": ("var", "var_losses"),
        "covar": ("covar", "covar_losses"),
        "force": ("force", "average_V", "mean_V"),
    }

    selected = []
    for key in keys:
        source_key = next(
            (candidate for candidate in aliases.get(key, (key,)) if candidate in history),
            None,
        )
        if source_key is not None:
            selected.append((key, np.asarray(history[source_key], dtype=float)))

    if not selected:
        raise ValueError("history does not contain any requested loss keys.")

    figure, axes = plt.subplots(
        len(selected),
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for axis, (key, values) in zip(axes, selected):
        axis.plot(values)
        axis.set_ylabel(key)
        if log_scale and np.all(values > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Epoch")
    figure.suptitle("Training losses")
    figure.tight_layout()
    return figure, axes


def plot_embedding(
    embedding: Any,
    labels: Any | None = None,
    ax: Any | None = None,
    title: str | None = None,
    point_size: float = 12.0,
    alpha: float = 0.7,
):
    """Scatter a two-dimensional PCA or UMAP embedding."""
    plt = _import_matplotlib()
    points = _as_2d(embedding, "embedding")
    if points.shape[1] != 2:
        raise ValueError("embedding must have exactly two columns.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=alpha)
    else:
        labels_array = _to_numpy(labels).reshape(-1)
        if labels_array.shape[0] != points.shape[0]:
            raise ValueError("labels and embedding must have the same number of samples.")
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=labels_array,
            s=point_size,
            alpha=alpha,
            cmap="tab10",
        )
        ax.figure.colorbar(scatter, ax=ax, label="label")

    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if title:
        ax.set_title(title)
    return ax


def plot_latent_projections(
    latents: Any,
    labels: Any | None = None,
    methods: Sequence[str] = ("pca", "umap"),
    figsize: tuple[float, float] = (12.0, 5.0),
):
    """Plot PCA and/or UMAP views of the same latent samples."""
    plt = _import_matplotlib()
    methods = tuple(methods)
    figure, axes = plt.subplots(1, len(methods), figsize=figsize, squeeze=False)
    axes = axes[0]

    for axis, method in zip(axes, methods):
        method_name = method.lower()
        if method_name == "pca":
            embedding = pca_embedding(latents, n_components=2)
        elif method_name == "umap":
            embedding = umap_embedding(latents, n_components=2)
        else:
            raise ValueError("methods must contain only 'pca' or 'umap'.")
        plot_embedding(embedding, labels=labels, ax=axis, title=method_name.upper())

    figure.tight_layout()
    return figure, axes


def plot_real_generated_projections(
    real_latents: Any,
    generated_latents: Any,
    *,
    real_labels: Any | None = None,
    generated_labels: Any | None = None,
    methods: Sequence[str] = ("pca", "umap"),
) -> dict[str, tuple[Any, Any]]:
    """Plot latent views by digit (when supplied) and always by source."""
    real_array, generated_array = _check_pair(real_latents, generated_latents)
    combined = np.concatenate([real_array, generated_array], axis=0)
    source_labels = np.concatenate([
        np.zeros(len(real_array), dtype=int),
        np.ones(len(generated_array), dtype=int),
    ])
    figures: dict[str, tuple[Any, Any]] = {
        "by_source": plot_latent_projections(
            combined,
            labels=source_labels,
            methods=methods,
        )
    }
    if real_labels is not None and generated_labels is not None:
        figures["by_label"] = plot_latent_projections(
            combined,
            labels=np.concatenate([
                _to_numpy(real_labels).reshape(-1),
                _to_numpy(generated_labels).reshape(-1),
            ]),
            methods=methods,
        )
    return figures


def plot_conditional_mnist_results(
    model: Any,
    data_loader: Any,
    *,
    device: Any | None = None,
    samples: int = 10,
    use_mean_for_reconstruction: bool = True,
):
    """Show real inputs, reconstructions, and label-conditioned generations."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("plot_conditional_mnist_results requires PyTorch.") from exc

    if samples < 1:
        raise ValueError("samples must be positive.")
    plt = _import_matplotlib()
    if device is None:
        target_device = next(model.parameters()).device
    else:
        target_device = torch.device(device)
    images, labels = next(iter(data_loader))
    images = images[:samples].to(target_device).flatten(start_dim=1)
    labels = labels[:samples].to(target_device).long()
    if images.shape[0] == 0:
        raise ValueError("data_loader yielded an empty batch.")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        mu, _, z = model.get_latent(images)
        reconstruction_latents = mu if use_mean_for_reconstruction else z
        reconstructions = model.decode(reconstruction_latents)
        noise = torch.randn(images.shape[0], model.latent_dim, device=target_device)
        _, generated = model.generate(noise, labels)
    if was_training:
        model.train()

    count = images.shape[0]
    figure, axes = plt.subplots(3, count, figsize=(1.5 * count, 4.8), squeeze=False)
    rows = (images, reconstructions, generated)
    for row, values in enumerate(rows):
        for column in range(count):
            axes[row, column].imshow(
                values[column].detach().cpu().reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("real")
    axes[1, 0].set_ylabel("recon")
    axes[2, 0].set_ylabel("generated")
    figure.suptitle("Rows: real MNIST | reconstructions | conditional generations")
    figure.tight_layout()
    return figure, axes


__all__ = [
    "sliced_wasserstein_distance",
    "mmd_rbf",
    "frechet_distance",
    "fid",
    "pixel_fid",
    "make_mnist_feature_evaluator",
    "fit_mnist_evaluator",
    "evaluate_mnist_generation",
    "class_coverage",
    "class_entropy",
    "evaluate_generation",
    "collect_latent_samples",
    "latent_statistics",
    "evaluate_latent_model",
    "evaluate_per_class_generation",
    "mutual_information",
    "mutual_information_gap",
    "total_correlation",
    "pca_embedding",
    "umap_embedding",
    "plot_loss_history",
    "plot_embedding",
    "plot_latent_projections",
    "plot_real_generated_projections",
    "plot_conditional_mnist_results",
]
