"""Shared 500-dimensional toy experiments for Algorithms 1--5."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from torch.utils.data import DataLoader, TensorDataset

from src.driftXpress import (
    build_nystrom_cache,
    compute_V,
    covariance_loss,
    drift_loss,
    kl_loss,
    pre_compute_summaries,
    recon_loss,
    select_landmarks,
    variance_loss,
)
from src.eval import (
    class_entropy,
    mutual_information_gap,
    plot_latent_correlation,
    plot_latent_projections,
    plot_loss_history,
    sliced_wasserstein_distance,
    total_correlation,
)

HIGH_DIM = 500
LATENT_DIM = 32
TOY_NAMES = ("blobs", "checkerboard", "circles")
TOY_CLASSES = {"blobs": 3, "checkerboard": 2, "circles": 2}
PROJECTION_SEEDS = {"blobs": 101, "checkerboard": 102, "circles": 103}

torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_checkerboard(n_samples, seed, squares=6, extent=3.0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-extent, extent, size=(n_samples, 2))
    scaled = (points + extent) / (2.0 * extent) * squares
    labels = (np.floor(scaled[:, 0]) + np.floor(scaled[:, 1])).astype(int) % 2
    return points, labels


def make_toy(name, n_samples, seed, high_dim=HIGH_DIM):
    """Create a labelled 2-D toy distribution embedded in 500 dimensions."""
    if name == "blobs":
        points, labels = make_blobs(
            n_samples=n_samples,
            centers=3,
            n_features=2,
            cluster_std=0.75,
            random_state=seed,
        )
    elif name == "checkerboard":
        points, labels = make_checkerboard(n_samples, seed)
    elif name == "circles":
        points, labels = make_circles(
            n_samples=n_samples,
            factor=0.45,
            noise=0.06,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown toy: {name}")

    points = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-8)
    projection_rng = np.random.default_rng(PROJECTION_SEEDS[name])
    projection, _ = np.linalg.qr(
        projection_rng.normal(size=(high_dim, 2)),
        mode="reduced",
    )
    noise_rng = np.random.default_rng(seed + 9000)
    high_dimensional = points @ projection.T
    high_dimensional += 0.02 * noise_rng.normal(
        size=(n_samples, high_dim)
    )
    return (
        high_dimensional.astype(np.float32),
        labels.astype(np.int64),
        points.astype(np.float32),
    )


class ToyJointModel(nn.Module):
    """Conditional VAE-generator model used by Algorithms 1, 3, 4, and 5."""

    def __init__(
        self,
        input_dim=HIGH_DIM,
        latent_dim=LATENT_DIM,
        num_classes=2,
        label_dim=16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim),
        )
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def get_latent(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar, self.reparameterize(mu, logvar)

    def encode_mu(self, x):
        return self.fc_mu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def generate(self, noise, labels):
        label_features = self.label_embedding(labels.long())
        z = self.generator(torch.cat([noise, label_features], dim=1))
        return z, self.decode(z)


class ToyVAE(nn.Module):
    def __init__(self, input_dim=HIGH_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim),
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def get_latent(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar, self.reparameterize(mu, logvar)

    def encode_mu(self, x):
        return self.fc_mu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)


class ToyLatentGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=LATENT_DIM,
        num_classes=2,
        label_dim=16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_classes, label_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, noise, labels):
        label_features = self.embedding(labels.long())
        return self.generator(torch.cat([noise, label_features], dim=1))


class ToySeparateModel(nn.Module):
    """A frozen VAE plus a separately trained conditional latent generator."""

    def __init__(self, vae, generator, num_classes):
        super().__init__()
        self.vae = vae
        self.generator_model = generator
        self.latent_dim = vae.latent_dim
        self.num_classes = num_classes

    def get_latent(self, x):
        return self.vae.get_latent(x)

    def encode_mu(self, x):
        return self.vae.encode_mu(x)

    def decode(self, z):
        return self.vae.decode(z)

    def generate(self, noise, labels):
        z = self.generator_model(noise, labels)
        return z, self.decode(z)


class ToyLatentDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim=LATENT_DIM,
        num_classes=2,
        label_dim=16,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, latents, labels):
        label_features = self.label_embedding(labels.long())
        return self.network(
            torch.cat([latents, label_features], dim=1)
        ).squeeze(-1)


def _history(extra=()):
    keys = ["total", "recon", "kl", "drift", "var", "covar", "force"]
    keys.extend(extra)
    return {key: [] for key in keys}


def _record(history, values):
    for key in history:
        history[key].append(float(values.get(key, 0.0)))


def _loader(x_numpy, y_numpy, batch_size):
    x = torch.as_tensor(x_numpy, dtype=torch.float32)
    y = torch.as_tensor(y_numpy, dtype=torch.long)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def build_conditional_drift_cache(
    model,
    x,
    labels,
    num_classes,
    num_landmarks,
    T,
    deterministic=False,
):
    """Build one fixed class-conditional cache for the full toy dataset."""
    model.eval()
    with torch.no_grad():
        if deterministic:
            support_latents = model.encode_mu(x)
        else:
            _, _, support_latents = model.get_latent(x)

    cache = {}
    for class_id in range(num_classes):
        support = support_latents[labels == class_id].detach()
        if support.shape[0] == 0:
            raise ValueError(f"No support samples for class {class_id}.")
        landmarks = select_landmarks(support, num_landmarks)
        K_inv_sqrt = build_nystrom_cache(landmarks, T)
        A, b = pre_compute_summaries(
            support,
            landmarks,
            K_inv_sqrt,
            T,
        )
        cache[class_id] = (landmarks, K_inv_sqrt, A, b)
    return cache


def conditional_drift_loss(generated, labels, cache, T):
    total = generated.new_zeros(())
    field = torch.zeros_like(generated)
    batch_size = generated.shape[0]

    for class_tensor in torch.unique(labels):
        class_id = int(class_tensor.item())
        mask = labels == class_id
        landmarks, K_inv_sqrt, A, b = cache[class_id]
        class_loss, class_field = drift_loss(
            generated[mask],
            landmarks,
            K_inv_sqrt,
            A,
            b,
            T,
        )
        total = total + mask.float().sum() / batch_size * class_loss
        field[mask] = class_field
    return total, field


def _batch_values(total, recon, kl, drift, var, covar, field, extra=None):
    values = {
        "total": total.item(),
        "recon": recon.item(),
        "kl": kl.item(),
        "drift": drift.item(),
        "var": var.item(),
        "covar": covar.item(),
        "force": field.norm(dim=1).mean().item(),
    }
    if extra:
        values.update(extra)
    return values


def _print_epoch(label, epoch, history):
    if epoch < 3 or (epoch + 1) % 10 == 0:
        print(
            f"{label} epoch {epoch + 1:03d}: "
            f"total={history['total'][-1]:.5f} "
            f"recon={history['recon'][-1]:.5f} "
            f"drift={history['drift'][-1]:.5f}"
        )


def _prepare_inputs(x_numpy, y_numpy):
    x = torch.as_tensor(x_numpy, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_numpy, dtype=torch.long, device=device)
    return x, y


def train_algorithm1(
    x_numpy,
    y_numpy,
    num_classes,
    num_epochs=50,
    batch_size=256,
    latent_dim=LATENT_DIM,
    num_landmarks=128,
    T=1.0,
    lambda_kl=1e-5,
    lambda_drift=30.0,
    lambda_var=0.1,
    lambda_cov=0.3,
    lr=1e-3,
):
    """Algorithm 1: joint VAE-generator training with raw-latent drift."""
    x, labels = _prepare_inputs(x_numpy, y_numpy)
    loader = _loader(x_numpy, y_numpy, batch_size)
    model = ToyJointModel(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)
    cache = build_conditional_drift_cache(
        model,
        x,
        labels,
        num_classes,
        num_landmarks,
        T,
        deterministic=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    history = _history()
    start = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        totals = {key: 0.0 for key in history}
        for batch_x, batch_labels in loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            noise = torch.randn(
                batch_x.shape[0],
                latent_dim,
                device=device,
            )
            mu, logvar, z_pos = model.get_latent(batch_x)
            x_recon = model.decode(z_pos)
            z_neg, _ = model.generate(noise, batch_labels)
            L_drift, V = conditional_drift_loss(
                z_neg,
                batch_labels,
                cache,
                T,
            )
            L_recon = recon_loss(x_recon, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(z_pos) + variance_loss(z_neg)
            L_cov = covariance_loss(z_pos) + covariance_loss(z_neg)
            loss = (
                L_recon
                + lambda_kl * L_kl
                + lambda_drift * L_drift
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            values = _batch_values(
                loss,
                L_recon,
                L_kl,
                L_drift,
                L_var,
                L_cov,
                V,
            )
            for key, value in values.items():
                totals[key] += value

        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 1", epoch, history)

    print(f"Algorithm 1 finished in {(time.perf_counter() - start) / 60:.2f} minutes")
    return model, history


def train_algorithm2(
    x_numpy,
    y_numpy,
    num_classes,
    num_epochs=50,
    batch_size=256,
    latent_dim=LATENT_DIM,
    num_landmarks=128,
    T=1.0,
    lambda_kl=1e-5,
    lambda_drift=30.0,
    lambda_var=0.1,
    lambda_cov=0.3,
    lr=1e-3,
):
    """Algorithm 2: separately trained VAE followed by generator training."""
    x, labels = _prepare_inputs(x_numpy, y_numpy)
    loader = _loader(x_numpy, y_numpy, batch_size)
    vae = ToyVAE(input_dim=x.shape[1], latent_dim=latent_dim).to(device)
    vae_optimizer = optim.AdamW(vae.parameters(), lr=lr)
    history = _history()

    print("Algorithm 2: VAE phase")
    for epoch in range(num_epochs):
        vae.train()
        totals = {key: 0.0 for key in history}
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            mu, logvar, z = vae.get_latent(batch_x)
            reconstruction = vae.decode(z)
            L_recon = recon_loss(reconstruction, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(mu)
            L_cov = covariance_loss(mu)
            loss = (
                L_recon
                + lambda_kl * L_kl
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            vae_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            vae_optimizer.step()
            values = _batch_values(
                loss,
                L_recon,
                L_kl,
                torch.zeros_like(L_recon),
                L_var,
                L_cov,
                torch.zeros_like(mu),
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 2 VAE", epoch, history)

    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    print("Algorithm 2: generator phase")
    cache = build_conditional_drift_cache(
        vae,
        x,
        labels,
        num_classes,
        num_landmarks,
        T,
        deterministic=False,
    )
    generator = ToyLatentGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)
    generator_optimizer = optim.AdamW(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        generator.train()
        totals = {key: 0.0 for key in history}
        for _, batch_labels in loader:
            batch_labels = batch_labels.to(device)
            noise = torch.randn(
                batch_labels.shape[0],
                latent_dim,
                device=device,
            )
            z_neg = generator(noise, batch_labels)
            L_drift, V = conditional_drift_loss(
                z_neg,
                batch_labels,
                cache,
                T,
            )
            L_var = variance_loss(z_neg)
            L_cov = covariance_loss(z_neg)
            loss = (
                lambda_drift * L_drift
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            generator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            generator_optimizer.step()
            values = _batch_values(
                loss,
                torch.zeros_like(L_drift),
                torch.zeros_like(L_drift),
                L_drift,
                L_var,
                L_cov,
                V,
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 2 generator", epoch, history)

    return ToySeparateModel(vae, generator, num_classes), history


def train_algorithm3(
    x_numpy,
    y_numpy,
    num_classes,
    num_epochs=50,
    batch_size=256,
    latent_dim=LATENT_DIM,
    num_landmarks=128,
    T=1.0,
    lambda_kl=1e-5,
    lambda_drift=30.0,
    lambda_var=0.1,
    lambda_cov=0.3,
    lr=1e-3,
):
    """Algorithm 3: joint training with drift on re-encoded generated data."""
    x, labels = _prepare_inputs(x_numpy, y_numpy)
    loader = _loader(x_numpy, y_numpy, batch_size)
    model = ToyJointModel(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)
    cache = build_conditional_drift_cache(
        model,
        x,
        labels,
        num_classes,
        num_landmarks,
        T,
        deterministic=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    history = _history()
    start = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        totals = {key: 0.0 for key in history}
        for batch_x, batch_labels in loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            noise = torch.randn(
                batch_x.shape[0],
                latent_dim,
                device=device,
            )
            mu, logvar, z_pos = model.get_latent(batch_x)
            x_recon = model.decode(z_pos)
            z_raw, x_neg = model.generate(noise, batch_labels)
            z_neg = model.encode_mu(x_neg)
            L_drift, V = conditional_drift_loss(
                z_neg,
                batch_labels,
                cache,
                T,
            )
            L_recon = recon_loss(x_recon, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(z_pos) + variance_loss(z_neg)
            L_cov = covariance_loss(z_pos) + covariance_loss(z_neg)
            loss = (
                L_recon
                + lambda_kl * L_kl
                + lambda_drift * L_drift
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values = _batch_values(
                loss,
                L_recon,
                L_kl,
                L_drift,
                L_var,
                L_cov,
                V,
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 3", epoch, history)

    print(f"Algorithm 3 finished in {(time.perf_counter() - start) / 60:.2f} minutes")
    return model, history


def _set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def train_algorithm4(
    x_numpy,
    y_numpy,
    num_classes,
    num_epochs=50,
    batch_size=256,
    latent_dim=LATENT_DIM,
    num_landmarks=128,
    T=1.0,
    lambda_kl=1e-5,
    lambda_drift=30.0,
    lambda_var=0.1,
    lambda_cov=0.3,
    lambda_adv=1.0,
    lr=1e-3,
    disc_lr=1e-3,
):
    """Algorithm 4: joint training with a latent adversarial discriminator."""
    x, labels = _prepare_inputs(x_numpy, y_numpy)
    loader = _loader(x_numpy, y_numpy, batch_size)
    model = ToyJointModel(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)
    discriminator = ToyLatentDiscriminator(
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    discriminator_optimizer = optim.AdamW(
        discriminator.parameters(),
        lr=disc_lr,
    )
    bce = nn.BCEWithLogitsLoss()
    cache = build_conditional_drift_cache(
        model,
        x,
        labels,
        num_classes,
        num_landmarks,
        T,
        deterministic=False,
    )
    history = _history(extra=("adv", "disc"))
    start = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        totals = {key: 0.0 for key in history}
        for batch_x, batch_labels in loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            noise = torch.randn(
                batch_x.shape[0],
                latent_dim,
                device=device,
            )

            with torch.no_grad():
                _, _, real_disc_latent = model.get_latent(batch_x)
                fake_disc_latent, _ = model.generate(noise, batch_labels)
            real_logits = discriminator(real_disc_latent, batch_labels)
            fake_logits = discriminator(fake_disc_latent, batch_labels)
            L_disc = 0.5 * (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            L_disc.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                5.0,
            )
            discriminator_optimizer.step()

            mu, logvar, z_pos = model.get_latent(batch_x)
            x_recon = model.decode(z_pos)
            z_neg, _ = model.generate(noise, batch_labels)
            L_drift, V = conditional_drift_loss(
                z_neg,
                batch_labels,
                cache,
                T,
            )
            L_recon = recon_loss(x_recon, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(z_pos) + variance_loss(z_neg)
            L_cov = covariance_loss(z_pos) + covariance_loss(z_neg)

            _set_requires_grad(discriminator, False)
            generator_logits = discriminator(z_neg, batch_labels)
            L_adv = bce(
                generator_logits,
                torch.ones_like(generator_logits),
            )
            _set_requires_grad(discriminator, True)

            loss = (
                L_recon
                + lambda_kl * L_kl
                + lambda_drift * L_drift
                + lambda_var * L_var
                + lambda_cov * L_cov
                + lambda_adv * L_adv
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            values = _batch_values(
                loss,
                L_recon,
                L_kl,
                L_drift,
                L_var,
                L_cov,
                V,
                extra={
                    "adv": L_adv.item(),
                    "disc": L_disc.item(),
                },
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 4", epoch, history)

    print(f"Algorithm 4 finished in {(time.perf_counter() - start) / 60:.2f} minutes")
    return model, history, discriminator


def _pretrain_vae(
    model,
    loader,
    epochs,
    lambda_kl,
    lambda_var,
    lambda_cov,
    lr,
):
    vae_parameters = list(model.encoder.parameters())
    vae_parameters += list(model.fc_mu.parameters())
    vae_parameters += list(model.fc_var.parameters())
    vae_parameters += list(model.decoder.parameters())
    optimizer = optim.AdamW(vae_parameters, lr=lr)
    history = _history()

    for epoch in range(epochs):
        model.train()
        totals = {key: 0.0 for key in history}
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            mu, logvar, z = model.get_latent(batch_x)
            reconstruction = model.decode(z)
            L_recon = recon_loss(reconstruction, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(mu)
            L_cov = covariance_loss(mu)
            loss = (
                L_recon
                + lambda_kl * L_kl
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_parameters, 5.0)
            optimizer.step()
            values = _batch_values(
                loss,
                L_recon,
                L_kl,
                torch.zeros_like(L_recon),
                L_var,
                L_cov,
                torch.zeros_like(mu),
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 5 VAE pretrain", epoch, history)
    return history


def train_algorithm5(
    x_numpy,
    y_numpy,
    num_classes,
    num_epochs=50,
    batch_size=256,
    latent_dim=LATENT_DIM,
    num_landmarks=128,
    T=1.0,
    lambda_kl=1e-5,
    lambda_drift=30.0,
    lambda_var=0.1,
    lambda_cov=0.3,
    lambda_l1=0.0,
    vae_pretrain_epochs=25,
    lr=1e-3,
):
    """Algorithm 5: pretraining, deterministic cache, and two drift paths."""
    x, labels = _prepare_inputs(x_numpy, y_numpy)
    loader = _loader(x_numpy, y_numpy, batch_size)
    model = ToyJointModel(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)

    print(f"Algorithm 5: VAE pretraining for {vae_pretrain_epochs} epochs")
    pretrain_history = _pretrain_vae(
        model,
        loader,
        vae_pretrain_epochs,
        lambda_kl,
        lambda_var,
        lambda_cov,
        lr,
    )

    print("Algorithm 5: deterministic cache construction")
    cache_start = time.perf_counter()
    cache = build_conditional_drift_cache(
        model,
        x,
        labels,
        num_classes,
        num_landmarks,
        T,
        deterministic=True,
    )
    print(
        "Deterministic cache built in "
        f"{(time.perf_counter() - cache_start) / 60:.2f} minutes"
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    history = _history(extra=("drift_reencoded", "drift_raw"))
    for key in history:
        history[key].extend(
            pretrain_history.get(
                key,
                [0.0] * len(pretrain_history["total"]),
            )
        )

    for epoch in range(num_epochs):
        model.train()
        totals = {key: 0.0 for key in history}
        for batch_x, batch_labels in loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            noise = torch.randn(
                batch_x.shape[0],
                latent_dim,
                device=device,
            )
            mu, logvar, z_pos = model.get_latent(batch_x)
            x_recon = model.decode(z_pos)
            z_raw, x_neg = model.generate(noise, batch_labels)
            z_neg = model.encode_mu(x_neg)

            L_reencoded, V_reencoded = conditional_drift_loss(
                z_neg,
                batch_labels,
                cache,
                T,
            )
            L_raw, V_raw = conditional_drift_loss(
                z_raw,
                batch_labels,
                cache,
                T,
            )
            L_drift = 0.5 * (L_reencoded + L_raw)
            V = 0.5 * (V_reencoded + V_raw)
            L_recon = recon_loss(x_recon, batch_x)
            L_kl = kl_loss(mu, logvar)
            L_var = variance_loss(mu) + variance_loss(z_neg)
            L_cov = covariance_loss(mu) + covariance_loss(z_neg)
            L1 = F.l1_loss(x_recon, batch_x)
            L_recon_total = L_recon + lambda_l1 * L1
            loss = (
                L_recon_total
                + lambda_kl * L_kl
                + lambda_drift * L_drift
                + lambda_var * L_var
                + lambda_cov * L_cov
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            values = _batch_values(
                loss,
                L_recon_total,
                L_kl,
                L_drift,
                L_var,
                L_cov,
                V,
                extra={
                    "drift_reencoded": L_reencoded.item(),
                    "drift_raw": L_raw.item(),
                },
            )
            for key, value in values.items():
                totals[key] += value
        _record(
            history,
            {key: value / len(loader) for key, value in totals.items()},
        )
        _print_epoch("Algorithm 5", epoch, history)

    return model, history


def sample_model(model, n_samples, n_classes, seed=123):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    labels = torch.arange(
        n_classes,
        device=device,
    ).repeat((n_samples + n_classes - 1) // n_classes)[:n_samples]
    noise = torch.randn(
        n_samples,
        model.latent_dim,
        generator=generator,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        generated_latents, generated = model.generate(noise, labels)
    return (
        generated_latents.cpu().numpy(),
        generated.cpu().numpy(),
        labels.cpu().numpy(),
    )


def evaluate_toy_model(model, real_x, real_y, n_classes, seed=123):
    real_x_tensor = torch.as_tensor(
        real_x,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        real_latents = model.encode_mu(real_x_tensor).cpu().numpy()
    generated_latents, generated_x, generated_y = sample_model(
        model,
        len(real_x),
        n_classes,
        seed=seed,
    )

    metrics = {
        "swd_data": float(
            sliced_wasserstein_distance(
                real_x,
                generated_x,
                n_projections=64,
                random_state=seed,
            )
        ),
        "swd_latent": float(
            sliced_wasserstein_distance(
                real_latents,
                generated_latents,
                n_projections=64,
                random_state=seed,
            )
        ),
        "real_latent_std": float(real_latents.std(axis=0).mean()),
        "generated_latent_std": float(generated_latents.std(axis=0).mean()),
        "real_latent_tc": float(total_correlation(real_latents)),
        "generated_latent_tc": float(total_correlation(generated_latents)),
        "real_latent_mig": float(
            mutual_information_gap(real_latents, real_y)
        ),
        "generated_latent_mig": float(
            mutual_information_gap(generated_latents, generated_y)
        ),
        "generated_class_entropy": float(
            class_entropy(generated_y, n_classes=n_classes)
        ),
    }
    return {
        "metrics": metrics,
        "real_x": np.asarray(real_x),
        "real_y": np.asarray(real_y),
        "real_latents": real_latents,
        "generated_x": generated_x,
        "generated_y": generated_y,
        "generated_latents": generated_latents,
    }


def plot_toy_losses(histories, algorithm_name):
    for toy_name, history in histories.items():
        keys = [
            key
            for key in (
                "total",
                "recon",
                "kl",
                "drift",
                "drift_reencoded",
                "drift_raw",
                "adv",
                "disc",
                "var",
                "covar",
                "force",
            )
            if key in history
        ]
        figure, _ = plot_loss_history(
            history,
            keys=tuple(keys),
            log_scale=True,
        )
        figure.suptitle(f"{algorithm_name} losses: {toy_name}")
        figure.tight_layout()
        plt.show()


def _projection_plot(values, labels, title, plot_samples, seed):
    rng = np.random.default_rng(seed)
    count = min(plot_samples, len(values))
    indices = rng.choice(len(values), size=count, replace=False)
    try:
        figure, _ = plot_latent_projections(
            values[indices],
            labels=labels[indices],
            methods=("pca", "umap"),
        )
    except ImportError:
        print("UMAP unavailable; showing PCA only.")
        figure, _ = plot_latent_projections(
            values[indices],
            labels=labels[indices],
            methods=("pca",),
        )
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


def plot_toy_analysis(evaluation, toy_name, plot_samples=1000):
    """Show real/generated and class-labelled PCA/UMAP in data and latent space."""
    real_x = evaluation["real_x"]
    generated_x = evaluation["generated_x"]
    real_latents = evaluation["real_latents"]
    generated_latents = evaluation["generated_latents"]
    real_y = evaluation["real_y"]
    generated_y = evaluation["generated_y"]

    for space_name, real_values, generated_values in (
        ("data space", real_x, generated_x),
        ("latent space", real_latents, generated_latents),
    ):
        values = np.concatenate([real_values, generated_values])
        source_labels = np.concatenate([
            np.zeros(len(real_values), dtype=np.int64),
            np.ones(len(generated_values), dtype=np.int64),
        ])
        class_labels = np.concatenate([real_y, generated_y])
        _projection_plot(
            values,
            source_labels,
            f"{toy_name} {space_name}: real (0) vs generated (1)",
            plot_samples * 2,
            101,
        )
        _projection_plot(
            values,
            class_labels,
            f"{toy_name} {space_name}: class labels",
            plot_samples * 2,
            102,
        )

    for name, latents in (
        ("real", real_latents),
        ("generated", generated_latents),
    ):
        figure, _, _ = plot_latent_correlation(
            latents,
            annotate=False,
        )
        figure.suptitle(f"{toy_name} {name} latent correlation")
        figure.tight_layout()
        plt.show()


def print_toy_metrics(evaluations):
    for toy_name, evaluation in evaluations.items():
        print(f"\n===== {toy_name} =====")
        for name, value in evaluation["metrics"].items():
            print(f"{name}: {value:.6f}")


__all__ = [
    "HIGH_DIM",
    "LATENT_DIM",
    "TOY_NAMES",
    "TOY_CLASSES",
    "PROJECTION_SEEDS",
    "device",
    "make_toy",
    "ToyJointModel",
    "ToyVAE",
    "ToyLatentGenerator",
    "ToySeparateModel",
    "ToyLatentDiscriminator",
    "build_conditional_drift_cache",
    "conditional_drift_loss",
    "train_algorithm1",
    "train_algorithm2",
    "train_algorithm3",
    "train_algorithm4",
    "train_algorithm5",
    "evaluate_toy_model",
    "plot_toy_losses",
    "plot_toy_analysis",
    "print_toy_metrics",
]
