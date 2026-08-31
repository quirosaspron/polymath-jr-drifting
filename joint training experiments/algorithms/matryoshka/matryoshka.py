"""Matryoshka representation learning helpers for the drifting experiments.

The implementation keeps the two-stage convention used by ``separate.ipynb``:
the VAE is trained first and a conditional latent generator is trained second.
At every nested dimension, the VAE must reconstruct from a zero-padded prefix
and the generator must follow its own prefix-level conditional drift field.
This makes the first ``d`` coordinates useful on their own while retaining the
full latent representation at the largest dimension.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .driftXpress import (
    build_nystrom_cache,
    covariance_loss,
    drift_loss,
    kzu,
    kl_loss,
    nystrom_map,
    pre_compute_summaries,
    recon_loss,
    select_landmarks,
    variance_loss,
)
from .training import compose_objective, format_epoch_losses


def default_matryoshka_dims(latent_dim: int) -> tuple[int, ...]:
    """Return useful nested dimensions ending at ``latent_dim``."""
    latent_dim = int(latent_dim)
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive.")
    candidates = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
    dimensions = [dimension for dimension in candidates if dimension < latent_dim]
    dimensions.append(latent_dim)
    return tuple(dimensions)


def validate_matryoshka_dims(
    dimensions: Sequence[int] | None,
    latent_dim: int | None = None,
    *,
    require_full: bool = False,
) -> tuple[int, ...]:
    """Validate and normalize nested prefix dimensions."""
    if dimensions is None:
        if latent_dim is None:
            raise ValueError("Pass dimensions or latent_dim.")
        dimensions = default_matryoshka_dims(latent_dim)

    normalized = tuple(int(dimension) for dimension in dimensions)
    if not normalized or any(dimension < 1 for dimension in normalized):
        raise ValueError("dimensions must contain positive integers.")
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise ValueError("dimensions must be strictly increasing and unique.")
    if latent_dim is not None and normalized[-1] > int(latent_dim):
        raise ValueError("No matryoshka dimension may exceed latent_dim.")
    if require_full and latent_dim is not None and normalized[-1] != int(latent_dim):
        raise ValueError("The final matryoshka dimension must equal latent_dim.")
    return normalized


def prefix_latent(latents: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return the first ``dimension`` coordinates of a latent matrix."""
    if not isinstance(latents, torch.Tensor) or latents.ndim < 2:
        raise ValueError("latents must be a tensor with a feature dimension.")
    dimension = int(dimension)
    if not 1 <= dimension <= latents.shape[-1]:
        raise ValueError("dimension must fit the latent feature dimension.")
    return latents[..., :dimension]


def _normalized_weights(
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None,
) -> tuple[float, ...]:
    if weights is None:
        raw = [1.0] * len(dimensions)
    elif isinstance(weights, Mapping):
        raw = [float(weights.get(dimension, 1.0)) for dimension in dimensions]
    else:
        raw = [float(value) for value in weights]
        if len(raw) != len(dimensions):
            raise ValueError("A sequence of weights must match dimensions.")
    if any(value < 0 for value in raw) or sum(raw) <= 0:
        raise ValueError("Matryoshka weights must be non-negative and non-zero.")
    total = sum(raw)
    return tuple(value / total for value in raw)


def _average_prefix_losses(
    losses: Mapping[int, torch.Tensor],
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    if not losses:
        raise ValueError("At least one prefix loss is required.")
    coefficients = _normalized_weights(dimensions, weights)
    result = next(iter(losses.values())).new_zeros(())
    for coefficient, dimension in zip(coefficients, dimensions):
        result = result + coefficient * losses[dimension]
    return result


def matryoshka_component_losses(
    values: torch.Tensor,
    loss_function: Callable[[torch.Tensor], torch.Tensor],
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Evaluate one scalar loss on every prefix and return its mean."""
    dimensions = validate_matryoshka_dims(dimensions, values.shape[-1])
    losses = {
        dimension: loss_function(prefix_latent(values, dimension))
        for dimension in dimensions
    }
    return _average_prefix_losses(losses, dimensions, weights), losses


def matryoshka_kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Average VAE KL loss over all nested latent prefixes."""
    dimensions = validate_matryoshka_dims(dimensions, mu.shape[-1])
    losses = {
        dimension: kl_loss(
            prefix_latent(mu, dimension),
            prefix_latent(logvar, dimension),
        )
        for dimension in dimensions
    }
    return _average_prefix_losses(losses, dimensions, weights)


def matryoshka_reconstruction_loss(
    model: Any,
    z: torch.Tensor,
    target: torch.Tensor,
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Average reconstruction loss after decoding every zero-padded prefix."""
    dimensions = validate_matryoshka_dims(dimensions, model.latent_dim, require_full=True)
    losses = {
        dimension: recon_loss(model.decode_prefix(z, dimension), target)
        for dimension in dimensions
    }
    return _average_prefix_losses(losses, dimensions, weights)


def matryoshka_variance_loss(
    latents: torch.Tensor,
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Average variance regularization over all nested prefixes."""
    loss, _ = matryoshka_component_losses(latents, variance_loss, dimensions, weights)
    return loss


def matryoshka_covariance_loss(
    latents: torch.Tensor,
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Average covariance regularization over all nested prefixes."""
    loss, _ = matryoshka_component_losses(latents, covariance_loss, dimensions, weights)
    return loss


class MatryoshkaVAE(nn.Module):
    """The VAE architecture from ``separate.ipynb`` with nested decoding."""

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 16,
        matryoshka_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.matryoshka_dims = validate_matryoshka_dims(
            matryoshka_dims,
            self.latent_dim,
            require_full=True,
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192), nn.LeakyReLU(0.2),
            nn.Linear(192, 96), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(96, self.latent_dim)
        self.fc_var = nn.Linear(96, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 96), nn.LeakyReLU(0.2),
            nn.Linear(96, 192), nn.LeakyReLU(0.2),
            nn.Linear(192, input_dim), nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def get_latent(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        return mu, logvar, self.reparameterize(mu, logvar)

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deterministic encoder mean used by full-joint training."""
        return self.fc_mu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError("decode expects a full latent vector.")
        return self.decoder(z)

    def decode_prefix(self, z: torch.Tensor, dimension: int) -> torch.Tensor:
        """Decode a prefix after zero-padding it to the full latent size."""
        dimension = int(dimension)
        if not 1 <= dimension <= self.latent_dim:
            raise ValueError("dimension must fit latent_dim.")
        prefix = prefix_latent(z, dimension)
        padded = prefix.new_zeros(*prefix.shape[:-1], self.latent_dim)
        padded[..., :dimension] = prefix
        return self.decode(padded)


class MatryoshkaLatentGenerator(nn.Module):
    """The conditional latent generator used by the separate experiment."""

    def __init__(
        self,
        latent_dim: int = 16,
        num_classes: int = 6,
        label_dim: int = 16,
        matryoshka_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.matryoshka_dims = validate_matryoshka_dims(
            matryoshka_dims,
            self.latent_dim,
            require_full=True,
        )
        self.embedding = nn.Embedding(num_classes, label_dim)
        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim + label_dim, 48), nn.SiLU(),
            nn.Linear(48, 48), nn.SiLU(),
            nn.Linear(48, 48), nn.SiLU(),
            nn.Linear(48, self.latent_dim),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = torch.cat([noise, self.embedding(labels.long())], dim=1)
        return self.generator(h)


class MatryoshkaModel(nn.Module):
    """Frozen-VAE plus conditional-generator wrapper used by ``src.eval``."""

    def __init__(
        self,
        vae_model: MatryoshkaVAE,
        generator_model: MatryoshkaLatentGenerator,
        num_classes: int = 6,
    ):
        super().__init__()
        self.vae = vae_model
        self.generator_model = generator_model
        self.latent_dim = vae_model.latent_dim
        self.matryoshka_dims = vae_model.matryoshka_dims
        self.num_classes = int(num_classes)

    def get_latent(self, x: torch.Tensor):
        return self.vae.get_latent(x)

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_mu(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def decode_prefix(self, z: torch.Tensor, dimension: int) -> torch.Tensor:
        return self.vae.decode_prefix(z, dimension)

    def generate(self, noise: torch.Tensor, labels: torch.Tensor):
        z = self.generator_model(noise, labels)
        return z, self.vae.decode(z)


# These aliases keep the notebook’s original model names intact.
VAE = MatryoshkaVAE
LatentGenerator = MatryoshkaLatentGenerator
f = MatryoshkaModel


def matryoshka_vae_losses(
    model: MatryoshkaVAE,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    dimensions: Sequence[int],
    lambda_kl: float,
    lambda_var: float,
    lambda_cov: float,
) -> dict[str, torch.Tensor]:
    """Return the VAE-side loss terms aggregated over nested prefixes."""
    dimensions = validate_matryoshka_dims(dimensions, model.latent_dim, require_full=True)
    L_recon = matryoshka_reconstruction_loss(model, z, x, dimensions)
    L_kl = matryoshka_kl_loss(mu, logvar, dimensions)
    L_var = matryoshka_variance_loss(mu, dimensions)
    L_cov = matryoshka_covariance_loss(mu, dimensions)
    representation = L_recon + lambda_kl * L_kl + lambda_var * L_var + lambda_cov * L_cov
    return {
        "recon": L_recon,
        "kl": L_kl,
        "var": L_var,
        "covar": L_cov,
        "representation": representation,
    }


def build_conditional_drift_cache(
    model: Any,
    data_loader: Any,
    num_classes: int,
    num_landmarks: int,
    T: float,
    matryoshka_dims: Sequence[int] | None = None,
    use_mean: bool = False,
):
    """Build one fixed class-conditional Nyström cache per prefix dimension."""
    if num_landmarks < 1:
        raise ValueError("num_landmarks must be positive.")
    latent_dim = int(getattr(model, "latent_dim"))
    dimensions = validate_matryoshka_dims(
        matryoshka_dims,
        latent_dim,
        require_full=True,
    )
    model.eval()
    references = {dimension: {digit: [] for digit in range(num_classes)} for dimension in dimensions}
    full_reference_batches = []
    with torch.no_grad():
        for images, labels in data_loader:
            parameter = next(model.parameters(), None)
            target_device = parameter.device if parameter is not None else images.device
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            if use_mean and hasattr(model, "encode_mu"):
                z = model.encode_mu(x)
            else:
                _, _, z = model.get_latent(x)
            labels = labels.to(target_device, non_blocking=True).long()
            full_reference_batches.append(z.detach())
            for dimension in dimensions:
                prefix = prefix_latent(z, dimension)
                for digit in range(num_classes):
                    mask = labels == digit
                    if mask.any():
                        references[dimension][digit].append(prefix[mask].detach())

    if not full_reference_batches:
        raise ValueError("data_loader did not yield any reference samples.")

    cache = {}
    for dimension in dimensions:
        dimension_cache = {}
        for digit in range(num_classes):
            if not references[dimension][digit]:
                raise ValueError(f"No reference samples found for digit {digit}.")
            support = torch.cat(references[dimension][digit], dim=0)
            landmarks = select_landmarks(support, num_landmarks)
            W = build_nystrom_cache(landmarks, T)
            A, b = pre_compute_summaries(support, landmarks, W, T)
            dimension_cache[digit] = (landmarks, W, A, b)
        cache[dimension] = dimension_cache
    return cache, torch.cat(full_reference_batches, dim=0)


def conditional_drift_loss(
    generated: torch.Tensor,
    labels: torch.Tensor,
    cache: Mapping[int, Mapping[int, tuple[torch.Tensor, ...]]],
    T: float,
    matryoshka_dims: Sequence[int] | None = None,
    weights: Mapping[int, float] | Sequence[float] | None = None,
):
    """Average conditional drift losses and fields over all latent prefixes."""
    if generated.ndim != 2:
        raise ValueError("generated must be a 2-D latent matrix.")
    if labels.shape[0] != generated.shape[0]:
        raise ValueError("labels and generated must have the same batch size.")
    dimensions = validate_matryoshka_dims(
        tuple(cache) if matryoshka_dims is None else matryoshka_dims,
        generated.shape[-1],
    )
    coefficients = _normalized_weights(dimensions, weights)
    total = generated.new_zeros(())
    field = torch.zeros_like(generated)
    batch_size = generated.shape[0]

    for coefficient, dimension in zip(coefficients, dimensions):
        if dimension not in cache:
            raise KeyError(f"Missing conditional drift cache for dimension {dimension}.")
        dimension_generated = prefix_latent(generated, dimension)
        dimension_field = torch.zeros_like(dimension_generated)
        dimension_total = generated.new_zeros(())
        for digit_tensor in torch.unique(labels):
            digit = int(digit_tensor.item())
            mask = labels == digit
            if digit not in cache[dimension]:
                raise KeyError(f"Missing drift cache for class {digit} at dimension {dimension}.")
            landmarks, W, A, b = cache[dimension][digit]
            digit_loss, digit_field = drift_loss(
                dimension_generated[mask], landmarks, W, A, b, T
            )
            fraction = mask.float().sum() / batch_size
            dimension_total = dimension_total + fraction * digit_loss
            dimension_field[mask] = digit_field
        total = total + coefficient * dimension_total
        field[..., :dimension] = field[..., :dimension] + coefficient * dimension_field
    return total, field


def training_loop(
    num_epochs: int,
    T: float,
    num_landmarks: int,
    lambda_kl: float,
    lambda_drift: float,
    lambda_var: float,
    lambda_cov: float,
    batch_size: int = 1000,
    lr: float = 1e-3,
    *,
    train_loader: Any,
    device: Any | None = None,
    num_classes: int = 6,
    latent_dim: int = 32,
    matryoshka_dims: Sequence[int] | None = None,
):
    """Run separate two-stage training with a nested-prefix objective.

    The returned tuple intentionally has the same order and names as the
    reference notebook: model, total/reconstruction/KL/drift/variance/
    covariance histories, force history, and latent evolution snapshots.
    ``batch_size`` is also used for the positive evolution snapshot, matching
    the reference convention.
    """
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dimensions = validate_matryoshka_dims(
        matryoshka_dims,
        latent_dim,
        require_full=True,
    )

    vae_model = VAE(784, latent_dim, dimensions).to(target_device)
    vae_optimizer = optim.AdamW(vae_model.parameters(), lr=lr)
    train_losses, recon_losses, kl_losses = [], [], []
    drift_losses, var_losses, covar_losses = [], [], []
    average_V, evolution = [], []

    print("VAE training started")
    for epoch in range(num_epochs + 1):
        vae_model.train()
        items = {k: 0.0 for k in ("total", "recon", "kl", "var", "covar")}
        for images, _ in train_loader:
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            mu, logvar, z = vae_model.get_latent(x)
            parts = matryoshka_vae_losses(
                vae_model, x, mu, logvar, z, dimensions,
                lambda_kl, lambda_var, lambda_cov,
            )
            loss = parts["representation"]
            vae_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 5.0)
            vae_optimizer.step()
            items["total"] += loss.item()
            items["recon"] += parts["recon"].item()
            items["kl"] += parts["kl"].item()
            items["var"] += parts["var"].item()
            items["covar"] += parts["covar"].item()
        n = len(train_loader)
        for key in items:
            items[key] /= n
        train_losses.append(items["total"])
        recon_losses.append(items["recon"])
        kl_losses.append(items["kl"])
        drift_losses.append(0.0)
        var_losses.append(items["var"])
        covar_losses.append(items["covar"])
        average_V.append(0.0)
        evolution.append({"epoch": epoch, "pos": z.detach().cpu().numpy(),
                          "neg": z.detach().cpu().numpy()})
        if epoch < 20 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"-------- VAE epoch {epoch} --------")
            print(f"total={items['total']:.6f}")
            print(f"recon={items['recon']:.6f}")
            print(f"kl={items['kl']:.6f}")
            print(f"var={items['var']:.6f}")
            print("distances", torch.cdist(z[:50], z[:50]).mean().item())
            print(f"covar={items['covar']:.6f}")

    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad = False

    print("Building fixed conditional attraction caches...")
    t0 = time.perf_counter()
    cache, reference_bank = build_conditional_drift_cache(
        vae_model, train_loader, num_classes, num_landmarks, T, dimensions
    )
    print(f"Fixed attraction caches built in {(time.perf_counter() - t0) / 60:.2f} minutes")
    reference_snapshot = reference_bank[:min(batch_size, len(reference_bank))]

    generator = LatentGenerator(latent_dim, num_classes, matryoshka_dims=dimensions).to(target_device)
    generator_optimizer = optim.AdamW(generator.parameters(), lr=lr)
    print("Conditional generator training started")
    for epoch in range(num_epochs + 1):
        generator.train()
        items = {k: 0.0 for k in ("total", "drift", "var", "covar", "V")}
        for _, labels in train_loader:
            labels = labels.to(target_device, non_blocking=True)
            noise = torch.randn(labels.shape[0], latent_dim, device=target_device)
            z_neg = generator(noise, labels)
            L_drift, V = conditional_drift_loss(
                z_neg, labels, cache, T, dimensions
            )
            L_var = matryoshka_variance_loss(z_neg, dimensions)
            L_cov = matryoshka_covariance_loss(z_neg, dimensions)
            loss = lambda_drift * L_drift + lambda_var * L_var + lambda_cov * L_cov
            generator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            generator_optimizer.step()
            items["total"] += loss.item()
            items["drift"] += L_drift.item()
            items["var"] += L_var.item()
            items["covar"] += L_cov.item()
            items["V"] += V.norm(dim=1).mean().item()
            last_neg = z_neg.detach()
        n = len(train_loader)
        for key in items:
            items[key] /= n
        train_losses.append(items["total"])
        recon_losses.append(0.0)
        kl_losses.append(0.0)
        drift_losses.append(items["drift"])
        var_losses.append(items["var"])
        covar_losses.append(items["covar"])
        average_V.append(items["V"])
        evolution.append({"epoch": num_epochs + 1 + epoch,
                           "pos": reference_snapshot.cpu().numpy(),
                           "neg": last_neg.cpu().numpy()})
        if epoch < 20 or epoch % 10 == 0 or epoch == num_epochs:
            y_pos_latent = reference_snapshot
            print(f"-------- Generator epoch {epoch} --------")
            print(f"total={items['total']:.6f}")
            print(f"drift={items['drift']:.6f}    scaled={lambda_drift * items['drift']:.6f}")
            print(f"var={items['var']:.6f}    scaled={lambda_var * items['var']:.6f}")
            print("distances", torch.cdist(y_pos_latent[:50], y_pos_latent[:50]).mean().item())
            print(f"covar={items['covar']:.6f}    scaled={lambda_cov * items['covar']:.6f}")
            print(f"average |V|={items['V']:.6f}")
            print(f"latent std positive={reference_snapshot.std(dim=0).mean().item():.6f}")
            print(f"latent std negative={last_neg.std(dim=0).mean().item():.6f}")

    model = f(vae_model, generator, num_classes).to(target_device)
    print("Separate training finished")
    return (model, train_losses, recon_losses, kl_losses,
            drift_losses, var_losses, covar_losses, average_V, evolution)


def _vae_parameters(model: MatryoshkaModel) -> list[torch.Tensor]:
    return list(model.vae.parameters())


def _generator_parameters(model: MatryoshkaModel) -> list[torch.Tensor]:
    return list(model.generator_model.parameters())


def pretrain_vae(
    model: MatryoshkaModel,
    data_loader: Any,
    epochs: int,
    lambda_kl: float,
    lambda_var: float,
    lambda_cov: float,
    lr: float = 1e-3,
    *,
    device: Any | None = None,
    matryoshka_dims: Sequence[int] | None = None,
):
    """Pretrain the nested VAE while keeping the generator frozen."""
    target_device = next(model.parameters()).device if device is None else torch.device(device)
    dimensions = validate_matryoshka_dims(
        matryoshka_dims if matryoshka_dims is not None else model.matryoshka_dims,
        model.latent_dim,
        require_full=True,
    )
    vae_parameters = _vae_parameters(model)
    generator_parameters = _generator_parameters(model)
    for parameter in vae_parameters:
        parameter.requires_grad_(True)
    for parameter in generator_parameters:
        parameter.requires_grad_(False)

    optimizer = optim.AdamW(vae_parameters, lr=lr)
    history = {key: [] for key in ("total", "recon", "kl", "var", "covar")}
    print(f"Pretraining the VAE part for {epochs} epochs (generator frozen)...")

    for epoch in range(1, epochs + 1):
        model.train()
        items = {key: 0.0 for key in history}
        for images, _ in data_loader:
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            mu, logvar, z = model.get_latent(x)
            parts = matryoshka_vae_losses(
                model.vae, x, mu, logvar, z, dimensions,
                lambda_kl, lambda_var, lambda_cov,
            )
            loss = parts["representation"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_parameters, max_norm=5.0)
            optimizer.step()

            items["total"] += loss.item()
            items["recon"] += parts["recon"].item()
            items["kl"] += parts["kl"].item()
            items["var"] += parts["var"].item()
            items["covar"] += parts["covar"].item()

        for key in items:
            items[key] /= len(data_loader)
            history[key].append(items[key])
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"-------- VAE pretrain epoch {epoch}/{epochs} --------")
            print(format_epoch_losses(
                {key: items[key] for key in ("recon", "kl", "var", "covar")},
                {"kl": lambda_kl, "var": lambda_var, "covar": lambda_cov},
            ))

    for parameter in generator_parameters:
        parameter.requires_grad_(True)
    return history


def matryoshka_latent_match_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    dimensions: Sequence[int],
    weights: Mapping[int, float] | Sequence[float] | None = None,
) -> torch.Tensor:
    """Match generated and target latent prefixes at every nested dimension."""
    if generated.shape != target.shape:
        raise ValueError("generated and target must have the same shape.")
    dimensions = validate_matryoshka_dims(dimensions, generated.shape[-1], require_full=True)
    losses = {
        dimension: F.mse_loss(
            prefix_latent(generated, dimension),
            prefix_latent(target, dimension),
        )
        for dimension in dimensions
    }
    return _average_prefix_losses(losses, dimensions, weights)


def warmup_generator(
    model: MatryoshkaModel,
    data_loader: Any,
    epochs: int,
    lr: float = 1e-3,
    *,
    device: Any | None = None,
    matryoshka_dims: Sequence[int] | None = None,
):
    """Warm up the generator against frozen encoder means at every prefix."""
    target_device = next(model.parameters()).device if device is None else torch.device(device)
    dimensions = validate_matryoshka_dims(
        matryoshka_dims if matryoshka_dims is not None else model.matryoshka_dims,
        model.latent_dim,
        require_full=True,
    )
    vae_parameters = _vae_parameters(model)
    generator_parameters = _generator_parameters(model)
    for parameter in vae_parameters:
        parameter.requires_grad_(False)
    for parameter in generator_parameters:
        parameter.requires_grad_(True)

    optimizer = optim.AdamW(generator_parameters, lr=lr)
    history = {"total": [], "latent_match": []}
    print(f"Warming up the generator for {epochs} epochs (VAE frozen)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        latent_match = 0.0
        for images, class_ids in data_loader:
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            class_ids = class_ids.to(target_device, non_blocking=True)
            noise = torch.randn(x.shape[0], model.latent_dim, device=target_device)
            with torch.no_grad():
                target_mu = model.encode_mu(x)
            generated_latents, _ = model.generate(noise, class_ids)
            loss = matryoshka_latent_match_loss(generated_latents, target_mu, dimensions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, max_norm=5.0)
            optimizer.step()

            total += loss.item()
            latent_match += loss.item()

        total /= len(data_loader)
        latent_match /= len(data_loader)
        history["total"].append(total)
        history["latent_match"].append(latent_match)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"-------- Generator warmup epoch {epoch}/{epochs} --------")
            print(f"latent_match={latent_match:.6f}")

    for parameter in vae_parameters:
        parameter.requires_grad_(True)
    return history


def warmup_generator_drift(
    model: MatryoshkaModel,
    data_loader: Any,
    epochs: int,
    cache: Mapping[int, Mapping[int, tuple[torch.Tensor, ...]]],
    T: float,
    lambda_drift: float = 1.0,
    raw_drift_weight: float = 0.25,
    lr: float = 1e-4,
    *,
    device: Any | None = None,
    matryoshka_dims: Sequence[int] | None = None,
):
    """Warm up only the generator with raw and re-encoded prefix drift."""
    target_device = next(model.parameters()).device if device is None else torch.device(device)
    dimensions = validate_matryoshka_dims(
        matryoshka_dims if matryoshka_dims is not None else model.matryoshka_dims,
        model.latent_dim,
        require_full=True,
    )
    vae_parameters = _vae_parameters(model)
    generator_parameters = _generator_parameters(model)
    for parameter in vae_parameters:
        parameter.requires_grad_(False)
    for parameter in generator_parameters:
        parameter.requires_grad_(True)

    optimizer = optim.AdamW(generator_parameters, lr=lr)
    history = {key: [] for key in ("total", "drift", "drift_raw", "force")}
    print(f"Warming up the generator with DriftXpress for {epochs} epochs (VAE frozen)...")

    for epoch in range(1, epochs + 1):
        model.train()
        items = {key: 0.0 for key in history}
        for images, class_ids in data_loader:
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            class_ids = class_ids.to(target_device, non_blocking=True)
            noise = torch.randn(x.shape[0], model.latent_dim, device=target_device)
            z_raw, x_neg = model.generate(noise, class_ids)
            z_reencoded = model.encode_mu(x_neg)
            L_drift_reencoded, V = conditional_drift_loss(
                z_reencoded, class_ids, cache, T, dimensions
            )
            L_drift_raw, _ = conditional_drift_loss(
                z_raw, class_ids, cache, T, dimensions
            )
            L_drift = L_drift_reencoded + raw_drift_weight * L_drift_raw
            loss = lambda_drift * L_drift

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, max_norm=5.0)
            optimizer.step()

            items["total"] += loss.item()
            items["drift"] += L_drift_reencoded.item()
            items["drift_raw"] += L_drift_raw.item()
            items["force"] += V.norm(dim=1).mean().item()

        for key in items:
            items[key] /= len(data_loader)
            history[key].append(items[key])
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"-------- Generator DriftXpress warmup epoch {epoch}/{epochs} --------")
            print(f"drift={items['drift']:.6f} raw_drift={items['drift_raw']:.6f} total={items['total']:.6f} average |V|={items['force']:.6f}")

    for parameter in vae_parameters:
        parameter.requires_grad_(True)
    return history


def pretrained_full_joint_training_loop(
    num_epochs: int,
    T: float,
    num_landmarks: int,
    lambda_kl: float,
    lambda_drift: float,
    lambda_var: float,
    lambda_cov: float,
    lambda_representation: float = 1.0,
    loss_scale: float = 1.0,
    vae_pretrain_epochs: int = 50,
    generator_warmup_epochs: int = 30,
    generator_drift_warmup_epochs: int = 25,
    generator_lr: float = 1e-3,
    generator_drift_lr: float = 1e-4,
    joint_generator_lr: float = 1e-4,
    joint_vae_lr: float = 1e-5,
    raw_drift_weight: float = 0.25,
    lr: float = 1e-4,
    *,
    train_loader: Any,
    device: Any | None = None,
    num_classes: int = 6,
    latent_dim: int = 32,
    matryoshka_dims: Sequence[int] | None = None,
):
    """Run the pretrained fully-joint route with nested prefix objectives."""
    global DRIFT_CACHE
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dimensions = validate_matryoshka_dims(
        matryoshka_dims,
        latent_dim,
        require_full=True,
    )
    model = f(
        VAE(784, latent_dim, dimensions),
        LatentGenerator(latent_dim, num_classes, matryoshka_dims=dimensions),
        num_classes,
    ).to(target_device)
    pretrain_history = pretrain_vae(
        model,
        train_loader,
        vae_pretrain_epochs,
        lambda_kl=lambda_kl,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
        lr=lr,
        device=target_device,
        matryoshka_dims=dimensions,
    )

    print("Building the fixed conditional DriftXpress cache once, after pretraining...")
    cache_start = time.perf_counter()
    DRIFT_CACHE, _ = build_conditional_drift_cache(
        model,
        train_loader,
        num_classes,
        num_landmarks,
        T,
        dimensions,
        use_mean=True,
    )
    print(f"Fixed attraction caches built in {(time.perf_counter() - cache_start) / 60:.2f} minutes")

    generator_warmup_history = warmup_generator(
        model,
        train_loader,
        epochs=generator_warmup_epochs,
        lr=generator_lr,
        device=target_device,
        matryoshka_dims=dimensions,
    )
    generator_drift_history = warmup_generator_drift(
        model,
        train_loader,
        epochs=generator_drift_warmup_epochs,
        cache=DRIFT_CACHE,
        T=T,
        lambda_drift=lambda_drift,
        raw_drift_weight=raw_drift_weight,
        lr=generator_drift_lr,
        device=target_device,
        matryoshka_dims=dimensions,
    )

    vae_parameters = _vae_parameters(model)
    generator_parameters = _generator_parameters(model)
    optimizer = optim.AdamW([
        {"params": generator_parameters, "lr": joint_generator_lr},
        {"params": vae_parameters, "lr": joint_vae_lr},
    ])
    print(
        f"Joint learning rates: generator={joint_generator_lr:g}, "
        f"VAE/encoder-decoder={joint_vae_lr:g}"
    )
    history = {key: [] for key in ("total", "recon", "kl", "drift", "drift_raw", "var", "covar", "force")}
    evolution = []
    snapshots = set(range(0, num_epochs + 1, max(1, num_epochs // 5)))
    snapshots.add(num_epochs)
    train_start = time.perf_counter()
    print("Pretrained fully joint conditional training started")

    for epoch in range(num_epochs + 1):
        model.train()
        items = {key: 0.0 for key in history}
        last_pos, last_neg = None, None
        for images, class_ids in train_loader:
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            class_ids = class_ids.to(target_device, non_blocking=True)
            noise = torch.randn(x.shape[0], model.latent_dim, device=target_device)

            mu, logvar, z_pos = model.get_latent(x)
            z_raw, x_neg = model.generate(noise, class_ids)
            z_neg = model.encode_mu(x_neg)

            L_drift_reencoded, V = conditional_drift_loss(
                z_neg, class_ids, DRIFT_CACHE, T, dimensions
            )
            L_drift_raw, _ = conditional_drift_loss(
                z_raw, class_ids, DRIFT_CACHE, T, dimensions
            )
            L_drift = L_drift_reencoded + raw_drift_weight * L_drift_raw
            L_recon = matryoshka_reconstruction_loss(model.vae, z_pos, x, dimensions)
            L_kl = matryoshka_kl_loss(mu, logvar, dimensions)
            L_var = matryoshka_variance_loss(mu, dimensions) + matryoshka_variance_loss(z_neg, dimensions)
            L_cov = matryoshka_covariance_loss(mu, dimensions) + matryoshka_covariance_loss(z_neg, dimensions)
            loss, _ = compose_objective(
                {
                    "recon": L_recon,
                    "kl": L_kl,
                    "drift": L_drift,
                    "var": L_var,
                    "covar": L_cov,
                },
                {
                    "kl": lambda_kl,
                    "drift": lambda_drift,
                    "var": lambda_var,
                    "covar": lambda_cov,
                },
                lambda_representation=lambda_representation,
                loss_scale=loss_scale,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            items["total"] += loss.item()
            items["recon"] += L_recon.item()
            items["kl"] += L_kl.item()
            items["drift"] += L_drift.item()
            items["drift_raw"] += L_drift_raw.item()
            items["var"] += L_var.item()
            items["covar"] += L_cov.item()
            items["force"] += V.norm(dim=1).mean().item()
            last_pos, last_neg = z_pos.detach(), z_neg.detach()

        for key in items:
            items[key] /= len(train_loader)
            history[key].append(items[key])
        if epoch in snapshots:
            evolution.append({
                "epoch": epoch,
                "pos": last_pos.cpu().numpy(),
                "neg": last_neg.cpu().numpy(),
            })
        if epoch < 20 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"-------- Epoch {epoch} --------")
            print(format_epoch_losses(
                {key: items[key] for key in ("recon", "kl", "drift", "var", "covar")},
                {
                    "kl": lambda_kl,
                    "drift": lambda_drift,
                    "var": lambda_var,
                    "covar": lambda_cov,
                },
                lambda_representation=lambda_representation,
                loss_scale=loss_scale,
            ))
            print(
                f"reencoded drift={items['drift'] - raw_drift_weight * items['drift_raw']:.6f} "
                f"raw drift={items['drift_raw']:.6f}"
            )
            print("distances", torch.cdist(z_pos[:50], z_pos[:50]).mean().item())
            print(f"average |V|={items['force']:.6f}")
            print(f"latent std positive={last_pos.std(dim=0).mean().item():.6f}")
            print(f"latent std negative={last_neg.std(dim=0).mean().item():.6f}")

    elapsed = (time.perf_counter() - train_start) / 60
    print(f"Finished. Joint training took {elapsed:.2f} minutes")
    return (
        model,
        history["total"],
        history["recon"],
        history["kl"],
        history["drift"],
        history["var"],
        history["covar"],
        history["force"],
        evolution,
        pretrain_history,
        generator_warmup_history,
        generator_drift_history,
        history["drift_raw"],
    )


def matryoshka_latent_views(
    latents: Any,
    dimensions: Sequence[int],
) -> dict[int, Any]:
    """Return NumPy- or tensor-compatible prefix views for evaluation."""
    dimensions = validate_matryoshka_dims(dimensions, getattr(latents, "shape", [0, 0])[-1])
    return {dimension: latents[..., :dimension] for dimension in dimensions}


def evaluate_matryoshka_latents(
    real_latents: Any,
    generated_latents: Any,
    dimensions: Sequence[int],
    *,
    generated_labels: Any | None = None,
    n_classes: int | None = None,
    random_state: int | None = 0,
    n_projections: int = 128,
    include_mmd: bool = False,
    include_fid: bool = True,
) -> dict[str, Any]:
    """Compute the standard feature metrics independently at each prefix."""
    from .eval import evaluate_generation, latent_statistics

    real_views = matryoshka_latent_views(real_latents, dimensions)
    generated_views = matryoshka_latent_views(generated_latents, dimensions)
    metrics = {}
    statistics = {}
    for dimension in validate_matryoshka_dims(dimensions, real_views[max(real_views)].shape[-1]):
        metrics[dimension] = evaluate_generation(
            real_views[dimension],
            generated_views[dimension],
            generated_labels=generated_labels,
            n_classes=n_classes,
            random_state=random_state,
            n_projections=n_projections,
            include_mmd=include_mmd,
            include_fid=include_fid,
        )
        statistics[dimension] = latent_statistics(
            real_views[dimension], generated_views[dimension]
        )
    return {"dimensions": tuple(metrics), "metrics": metrics, "statistics": statistics}


def evaluate_matryoshka_reconstructions(
    model: Any,
    data_loader: Any,
    dimensions: Sequence[int] | None = None,
    *,
    device: Any | None = None,
    n_samples: int | None = None,
) -> dict[int, float]:
    """Measure reconstruction MSE from each nested mean-latent prefix."""
    if device is None:
        target_device = next(model.parameters()).device
    else:
        target_device = torch.device(device)
    latent_dim = int(model.latent_dim)
    dimensions = validate_matryoshka_dims(
        dimensions if dimensions is not None else getattr(model, "matryoshka_dims", None),
        latent_dim,
        require_full=True,
    )
    was_training = bool(getattr(model, "training", False))
    model.eval()
    totals = {dimension: 0.0 for dimension in dimensions}
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            if n_samples is not None and count >= n_samples:
                break
            x = images.to(target_device, non_blocking=True).flatten(start_dim=1)
            if n_samples is not None:
                x = x[:min(x.shape[0], n_samples - count)]
            if x.shape[0] == 0:
                break
            mu, _, _ = model.get_latent(x)
            for dimension in dimensions:
                reconstruction = model.decode_prefix(mu, dimension)
                totals[dimension] += F.mse_loss(reconstruction, x).item() * x.shape[0]
            count += x.shape[0]
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("data_loader did not yield any samples.")
    return {dimension: value / count for dimension, value in totals.items()}


def evaluate_matryoshka_model(
    model: Any,
    data_loader: Any,
    *,
    dimensions: Sequence[int] | None = None,
    device: Any | None = None,
    n_samples: int | None = None,
    n_classes: int | None = None,
    class_to_label: Any | None = None,
    random_state: int | None = 0,
    n_projections: int = 128,
    include_mmd: bool = False,
    include_fid: bool = True,
) -> dict[str, Any]:
    """Run full-space evaluation plus prefix metrics and reconstruction MSE."""
    from .eval import evaluate_latent_model

    result = evaluate_latent_model(
        model,
        data_loader,
        device=device,
        n_samples=n_samples,
        n_classes=n_classes,
        class_to_label=class_to_label,
        random_state=random_state,
        n_projections=n_projections,
        include_mmd=include_mmd,
        include_fid=include_fid,
    )
    samples = result["samples"]
    dimensions = validate_matryoshka_dims(
        dimensions if dimensions is not None else getattr(model, "matryoshka_dims", None),
        int(model.latent_dim),
        require_full=True,
    )
    prefix_result = evaluate_matryoshka_latents(
        samples["real_latents"],
        samples["generated_latents"],
        dimensions,
        generated_labels=samples["generated_class_ids"],
        n_classes=n_classes,
        random_state=random_state,
        n_projections=n_projections,
        include_mmd=include_mmd,
        include_fid=include_fid,
    )
    result["matryoshka_dimensions"] = dimensions
    result["matryoshka_metrics"] = prefix_result["metrics"]
    result["matryoshka_statistics"] = prefix_result["statistics"]
    result["matryoshka_reconstruction_mse"] = evaluate_matryoshka_reconstructions(
        model,
        data_loader,
        dimensions,
        device=device,
        n_samples=n_samples,
    )
    return result


def plot_matryoshka_diagnostics(
    metrics: Mapping[int, Mapping[str, float]],
    reconstruction_mse: Mapping[int, float] | None = None,
    *,
    metric_names: Sequence[str] = ("sliced_wasserstein", "fid"),
    figsize: tuple[float, float] = (10.0, 8.0),
):
    """Plot prefix metrics and reconstruction error against latent dimension."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_matryoshka_diagnostics requires matplotlib.") from exc

    if not metrics:
        raise ValueError("metrics must contain at least one dimension.")
    dimensions = tuple(sorted(int(dimension) for dimension in metrics))
    available_names = tuple(
        name for name in metric_names
        if any(name in metrics[dimension] for dimension in dimensions)
    )
    n_panels = 1 + int(reconstruction_mse is not None)
    figure, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    axes = axes[:, 0]
    axis = axes[0]
    for name in available_names:
        axis.plot(
            dimensions,
            [metrics[dimension].get(name, float("nan")) for dimension in dimensions],
            marker="o",
            label=name,
        )
    axis.set_ylabel("metric")
    axis.set_title("Matryoshka prefix evaluation")
    axis.grid(alpha=0.25)
    if available_names:
        axis.legend()

    if reconstruction_mse is not None:
        reconstruction_axis = axes[1]
        reconstruction_axis.plot(
            dimensions,
            [reconstruction_mse[dimension] for dimension in dimensions],
            marker="o",
            color="tab:green",
        )
        reconstruction_axis.set_ylabel("MSE")
        reconstruction_axis.set_title("Reconstruction from a zero-padded prefix")
        reconstruction_axis.grid(alpha=0.25)
        reconstruction_axis.set_xlabel("latent prefix dimension")
    else:
        axis.set_xlabel("latent prefix dimension")
    figure.tight_layout()
    return figure, axes


def plot_matryoshka_reconstructions(
    model: Any,
    data_loader: Any,
    dimensions: Sequence[int] | None = None,
    *,
    device: Any | None = None,
    samples: int = 8,
):
    """Show reconstructions made from each nested prefix on separate rows."""
    if samples < 1:
        raise ValueError("samples must be positive.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_matryoshka_reconstructions requires matplotlib.") from exc

    target_device = next(model.parameters()).device if device is None else torch.device(device)
    latent_dim = int(model.latent_dim)
    dimensions = validate_matryoshka_dims(
        dimensions if dimensions is not None else getattr(model, "matryoshka_dims", None),
        latent_dim,
        require_full=True,
    )
    images, _ = next(iter(data_loader))
    images = images[:samples].to(target_device, non_blocking=True).flatten(start_dim=1)
    if images.shape[0] == 0:
        raise ValueError("data_loader yielded an empty batch.")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode_mu"):
            mu = model.encode_mu(images)
        else:
            mu, _, _ = model.get_latent(images)
        reconstructions = {
            dimension: model.decode_prefix(mu, dimension)
            for dimension in dimensions
        }
    if was_training:
        model.train()

    count = images.shape[0]
    figure, axes = plt.subplots(
        len(dimensions) + 1,
        count,
        figsize=(1.5 * count, 1.6 * (len(dimensions) + 1)),
        squeeze=False,
    )
    rows = [("real", images)] + [
        (f"d={dimension}", reconstructions[dimension])
        for dimension in dimensions
    ]
    for row, (label, values) in enumerate(rows):
        for column in range(count):
            axes[row, column].imshow(
                values[column].detach().cpu().reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(label)
    figure.suptitle("Reconstructions from nested latent prefixes")
    figure.tight_layout()
    return figure, axes


__all__ = [
    "default_matryoshka_dims",
    "validate_matryoshka_dims",
    "prefix_latent",
    "matryoshka_component_losses",
    "matryoshka_kl_loss",
    "matryoshka_reconstruction_loss",
    "matryoshka_variance_loss",
    "matryoshka_covariance_loss",
    "MatryoshkaVAE",
    "MatryoshkaLatentGenerator",
    "MatryoshkaModel",
    "VAE",
    "LatentGenerator",
    "f",
    "matryoshka_vae_losses",
    "matryoshka_latent_match_loss",
    "build_conditional_drift_cache",
    "conditional_drift_loss",
    "training_loop",
    "pretrain_vae",
    "warmup_generator",
    "warmup_generator_drift",
    "pretrained_full_joint_training_loop",
    "matryoshka_latent_views",
    "evaluate_matryoshka_latents",
    "evaluate_matryoshka_reconstructions",
    "evaluate_matryoshka_model",
    "plot_matryoshka_diagnostics",
    "plot_matryoshka_reconstructions",
]
