"""Toy data, MNIST loaders, and deterministic paired-run inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ToyDatasetBundle:
    """One observed dataset with recoverable signal and nuisance coordinates."""

    observed: torch.Tensor
    signal: torch.Tensor
    nuisance: torch.Tensor
    labels: torch.Tensor
    signal_projection: torch.Tensor
    nuisance_projection: torch.Tensor
    true_representation: torch.Tensor

    def project_signal(self, values: torch.Tensor) -> torch.Tensor:
        """Project observed coordinates back to the known signal coordinates."""

        return values @ self.signal_projection.to(values.device, values.dtype)

    def project_nuisance(self, values: torch.Tensor) -> torch.Tensor:
        """Project observed coordinates back to the nuisance coordinates."""

        return values @ self.nuisance_projection.to(values.device, values.dtype)


@dataclass(frozen=True)
class ToyDriftProblem:
    """Target and initial particles for the raw-versus-known-signal control."""

    target: torch.Tensor
    initial: torch.Tensor
    target_signal: torch.Tensor
    initial_signal: torch.Tensor
    target_nuisance: torch.Tensor
    initial_nuisance: torch.Tensor
    labels: torch.Tensor
    signal_projection: torch.Tensor
    nuisance_projection: torch.Tensor

    def project_signal(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.signal_projection.to(values.device, values.dtype)

    def project_nuisance(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.nuisance_projection.to(values.device, values.dtype)


@dataclass(frozen=True)
class PairedRunInputs:
    """Random quantities that every compared method must reuse exactly."""

    real_batch_indices: torch.Tensor
    training_noise: torch.Tensor
    landmark_indices: torch.Tensor
    evaluation_noise: torch.Tensor

    @property
    def batch_indices(self) -> torch.Tensor:
        """Compatibility alias for the original scaffold name."""

        return self.real_batch_indices

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "real_batch_indices": self.real_batch_indices,
            "training_noise": self.training_noise,
            "landmark_indices": self.landmark_indices,
            "evaluation_noise": self.evaluation_noise,
        }


def make_blob_dataset(
    *,
    n_samples: int,
    ambient_dim: int = 32,
    signal_dim: int = 2,
    centers: int = 8,
    cluster_std: float = 1.0,
    nuisance_scale: float = 1.0,
    rotate: bool = False,
    random_state: int = 0,
) -> ToyDatasetBundle:
    """Create blobs with known signal coordinates and Gaussian nuisance columns.

    With ``rotate=False``, the known representation is simply ``x[:, :2]`` for
    the default configuration. Rotation is optional so the first experiment
    remains easy to inspect.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if signal_dim < 1 or ambient_dim < signal_dim:
        raise ValueError("require 1 <= signal_dim <= ambient_dim")
    if centers < 1 or cluster_std <= 0 or nuisance_scale < 0:
        raise ValueError("centers/cluster_std must be positive and nuisance_scale non-negative")

    try:
        from sklearn.datasets import make_blobs
    except ImportError as exc:
        raise ImportError("make_blob_dataset requires scikit-learn") from exc

    signal, labels = make_blobs(
        n_samples=n_samples,
        n_features=signal_dim,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state + 1)
    nuisance_dim = ambient_dim - signal_dim
    nuisance = rng.normal(
        loc=0.0,
        scale=nuisance_scale,
        size=(n_samples, nuisance_dim),
    )
    base = np.concatenate([signal, nuisance], axis=1)

    rotation = np.eye(ambient_dim)
    if rotate:
        rotation, _ = np.linalg.qr(rng.normal(size=(ambient_dim, ambient_dim)))
    observed = base @ rotation
    recovery = rotation.T

    observed_tensor = torch.as_tensor(observed, dtype=torch.float32)
    signal_tensor = torch.as_tensor(signal, dtype=torch.float32)
    nuisance_tensor = torch.as_tensor(nuisance, dtype=torch.float32)
    return ToyDatasetBundle(
        observed=observed_tensor,
        signal=signal_tensor,
        nuisance=nuisance_tensor,
        labels=torch.as_tensor(labels, dtype=torch.long),
        signal_projection=torch.as_tensor(recovery[:, :signal_dim], dtype=torch.float32),
        nuisance_projection=torch.as_tensor(recovery[:, signal_dim:], dtype=torch.float32),
        true_representation=signal_tensor,
    )


def make_toy_drift_problem(
    *,
    n_samples: int = 2048,
    ambient_dim: int = 32,
    signal_dim: int = 2,
    centers: int = 8,
    cluster_std: float = 0.6,
    nuisance_scale: float = 1.0,
    initial_signal_scale: float = 1.5,
    rotate: bool = False,
    random_state: int = 0,
) -> ToyDriftProblem:
    """Build a paired toy problem whose initial nuisance marginal is correct.

    Target and initial particles have identical nuisance coordinates, while the
    initial signal is a broad Gaussian. This isolates whether irrelevant
    dimensions make signal drift harder. Nuisance discrepancy should be tracked
    because an already-correct marginal should not be damaged.
    """

    if initial_signal_scale <= 0:
        raise ValueError("initial_signal_scale must be positive")
    target = make_blob_dataset(
        n_samples=n_samples,
        ambient_dim=ambient_dim,
        signal_dim=signal_dim,
        centers=centers,
        cluster_std=cluster_std,
        nuisance_scale=nuisance_scale,
        rotate=rotate,
        random_state=random_state,
    )
    generator = torch.Generator(device="cpu").manual_seed(random_state + 2)
    target_mean = target.signal.mean(dim=0, keepdim=True)
    target_std = target.signal.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-6)
    initial_signal = target_mean + initial_signal_scale * target_std * torch.randn(
        n_samples, signal_dim, generator=generator
    )
    initial_nuisance = target.nuisance.clone()
    initial_base = torch.cat([initial_signal, initial_nuisance], dim=1)

    recovery = torch.cat([target.signal_projection, target.nuisance_projection], dim=1)
    rotation = recovery.T
    initial_observed = initial_base @ rotation
    return ToyDriftProblem(
        target=target.observed,
        initial=initial_observed,
        target_signal=target.signal,
        initial_signal=initial_signal,
        target_nuisance=target.nuisance,
        initial_nuisance=initial_nuisance,
        labels=target.labels,
        signal_projection=target.signal_projection,
        nuisance_projection=target.nuisance_projection,
    )


def build_mnist_loaders(
    *,
    batch_size: int = 500,
    selected_digits: Sequence[int] | None = None,
    data_dir: str | Path = "data",
    validation_fraction: float = 0.1,
    random_state: int = 0,
    num_workers: int = 0,
    download: bool = True,
) -> dict[str, Any]:
    """Build deterministic MNIST loaders and expose their exact split indices."""

    if batch_size < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("batch_size must be positive and validation_fraction in [0, 1)")
    try:
        from torch.utils.data import DataLoader, Subset, TensorDataset
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError("MNIST loading requires torch and torchvision") from exc

    digits = tuple(range(10)) if selected_digits is None else tuple(selected_digits)
    if not digits or len(set(digits)) != len(digits) or any(digit not in range(10) for digit in digits):
        raise ValueError("selected_digits must contain unique digits from 0 through 9")
    digit_to_class = {digit: index for index, digit in enumerate(digits)}
    class_to_digit = {index: digit for digit, index in digit_to_class.items()}

    def filtered_dataset(train: bool) -> TensorDataset:
        source = MNIST(root=str(data_dir), train=train, download=download)
        mask = torch.zeros_like(source.targets, dtype=torch.bool)
        for digit in digits:
            mask |= source.targets == digit
        images = source.data[mask].float().unsqueeze(1) / 255.0
        original_labels = source.targets[mask]
        labels = torch.as_tensor(
            [digit_to_class[int(value)] for value in original_labels], dtype=torch.long
        )
        return TensorDataset(images, labels)

    full_train = filtered_dataset(train=True)
    test_dataset = filtered_dataset(train=False)
    split_generator = torch.Generator(device="cpu").manual_seed(random_state)
    permutation = torch.randperm(len(full_train), generator=split_generator)
    n_validation = int(round(len(full_train) * validation_fraction))
    validation_indices = permutation[:n_validation]
    train_indices = permutation[n_validation:]
    train_dataset = Subset(full_train, train_indices.tolist())
    validation_dataset = Subset(full_train, validation_indices.tolist())

    shuffle_generator = torch.Generator(device="cpu").manual_seed(random_state + 1)
    loader_options = {"batch_size": batch_size, "num_workers": num_workers}
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "digit_to_class": digit_to_class,
        "class_to_digit": class_to_digit,
    }


def _cpu_generator(seed: int, stream: int) -> torch.Generator:
    modulus = 2**63 - 1
    return torch.Generator(device="cpu").manual_seed(
        (int(seed) + 1_000_003 * stream) % modulus
    )


def make_fixed_evaluation_noise(
    *, evaluation_size: int, noise_dim: int, seed: int
) -> torch.Tensor:
    """Create generator queries reused at every evaluation checkpoint."""

    if evaluation_size < 1 or noise_dim < 1:
        raise ValueError("evaluation_size and noise_dim must be positive")
    return torch.randn(
        evaluation_size,
        noise_dim,
        generator=_cpu_generator(seed, stream=4),
    )


def make_landmark_schedule(
    *, n_steps: int, dataset_size: int, n_landmarks: int, seed: int
) -> torch.Tensor:
    """Create deterministic per-step landmark indices without replacement."""

    if n_steps < 1 or dataset_size < 1 or not 1 <= n_landmarks <= dataset_size:
        raise ValueError("invalid step, dataset, or landmark count")
    generator = _cpu_generator(seed, stream=3)
    return torch.stack(
        [
            torch.randperm(dataset_size, generator=generator)[:n_landmarks]
            for _ in range(n_steps)
        ]
    )


def make_paired_run_inputs(
    *,
    n_steps: int,
    dataset_size: int,
    batch_size: int,
    noise_dim: int,
    n_landmarks: int,
    evaluation_size: int,
    seed: int,
) -> PairedRunInputs:
    """Pre-generate every stochastic input shared by compared methods."""

    if n_steps < 1 or dataset_size < 1 or batch_size < 1 or noise_dim < 1:
        raise ValueError("step, dataset, batch, and noise sizes must be positive")
    if not 1 <= n_landmarks <= dataset_size:
        raise ValueError("n_landmarks must be between 1 and dataset_size")
    batch_indices = torch.randint(
        dataset_size,
        size=(n_steps, batch_size),
        generator=_cpu_generator(seed, stream=1),
    )
    training_noise = torch.randn(
        n_steps,
        batch_size,
        noise_dim,
        generator=_cpu_generator(seed, stream=2),
    )
    fixed_landmarks = make_landmark_schedule(
        n_steps=1,
        dataset_size=dataset_size,
        n_landmarks=n_landmarks,
        seed=seed,
    )[0]
    return PairedRunInputs(
        real_batch_indices=batch_indices,
        training_noise=training_noise,
        landmark_indices=fixed_landmarks,
        evaluation_noise=make_fixed_evaluation_noise(
            evaluation_size=evaluation_size,
            noise_dim=noise_dim,
            seed=seed,
        ),
    )


def save_paired_run_inputs(inputs: PairedRunInputs, path: str | Path) -> Path:
    """Persist a paired schedule so another machine can replay it exactly."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs.to_dict(), destination)
    return destination


def load_paired_run_inputs(path: str | Path) -> PairedRunInputs:
    """Load a schedule saved by :func:`save_paired_run_inputs`."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return PairedRunInputs(**payload)


__all__ = [
    "PairedRunInputs",
    "ToyDatasetBundle",
    "ToyDriftProblem",
    "build_mnist_loaders",
    "load_paired_run_inputs",
    "make_blob_dataset",
    "make_fixed_evaluation_noise",
    "make_landmark_schedule",
    "make_paired_run_inputs",
    "make_toy_drift_problem",
    "save_paired_run_inputs",
]
