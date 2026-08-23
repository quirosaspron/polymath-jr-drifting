"""PyTorch helpers for the landmark/Nyström drift approximation."""

import torch
import torch.nn.functional as F


def select_landmarks(y_pos, num_landmarks=128):
    num_landmarks = min(num_landmarks, y_pos.shape[0])
    indices = torch.randperm(y_pos.shape[0], device=y_pos.device)[:num_landmarks]
    return y_pos[indices]


def kzu(z, landmarks, T):
    return torch.exp(-torch.cdist(z, landmarks) / T)


def build_nystrom_cache(landmarks, T, jitter=1e-5):
    K_uu = kzu(landmarks, landmarks, T)
    K_uu = K_uu + jitter * torch.eye(len(landmarks), device=landmarks.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(K_uu)
    eigenvalues = torch.clamp(eigenvalues, min=jitter)
    return eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.T


def nystrom_map(z, landmarks, K_uu_inv_sqrt, T):
    # The factor is cached: no eigendecomposition occurs during training.
    return (kzu(z, landmarks, T) @ K_uu_inv_sqrt).T


def pre_compute_summaries(y_pos, landmarks, K_uu_inv_sqrt, T):
    phi_y = nystrom_map(y_pos, landmarks, K_uu_inv_sqrt, T)
    return phi_y @ y_pos, phi_y.sum(dim=1, keepdim=True)


def attraction(y_neg, landmarks, K_uu_inv_sqrt, Ap, bp, T):
    phi = nystrom_map(y_neg, landmarks, K_uu_inv_sqrt, T)
    numerator = (Ap.T @ phi).T
    denominator = phi.T @ bp + 1e-8
    return numerator / denominator


def repulsion(y_neg, T):
    N = y_neg.shape[0]
    dist_neg = torch.cdist(y_neg, y_neg)

    # Remove self-interaction.
    dist_neg += torch.eye(N, device=dist_neg.device) * 1e6

    K = torch.exp(-dist_neg / T)
    denominator = K.sum(dim=1, keepdim=True) + 1e-8
    return (K @ y_neg) / denominator


def compute_V(y_neg, landmarks, K_uu_inv_sqrt, Ap, bp, T=1.0):
    V_pos = attraction(y_neg, landmarks, K_uu_inv_sqrt, Ap, bp, T)
    V_neg = repulsion(y_neg, T)
    return V_pos - V_neg


def drift_loss(generated, landmarks, K_uu_inv_sqrt, Ap, bp, T):
    with torch.no_grad():
        V = compute_V(generated, landmarks, K_uu_inv_sqrt, Ap, bp, T)
        frozen_targets = (generated + V).detach()
    return F.mse_loss(generated, frozen_targets), V


def recon_loss(y_pos_recon, y_pos):
    return F.mse_loss(y_pos_recon, y_pos, reduction="mean")


def kl_loss(mu, logvar):
    kl_per_sample = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=-1)
    return kl_per_sample.mean()


def variance_loss(mu, gamma=1.5, eps=1e-4):
    std = torch.sqrt(mu.var(dim=0, unbiased=True) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(latents):
    batch_size, dim = latents.shape
    latents = latents - latents.mean(dim=0, keepdim=True)
    cov = (latents.T @ latents) / (batch_size - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / dim


def total_loss(
    y_pos_recon,
    y_pos,
    y_pos_mu,
    y_pos_logvar,
    y_neg_latent,
    y_pos_latent,
    temp,
    landmarks,
    K_uu_inv_sqrt,
    Ap,
    bp,
    lambda_kl=0.01,
    lambda_drift=1.0,
    lambda_var=0.1,
    lambda_cov=0.1,
    lambda_representation=1.0,
):
    L_recon = recon_loss(y_pos_recon, y_pos)
    L_kl = kl_loss(y_pos_mu, y_pos_logvar)
    L_drift, V = drift_loss(
        y_neg_latent, landmarks, K_uu_inv_sqrt, Ap, bp, temp
    )
    L_var =  variance_loss(y_pos_mu) + variance_loss(y_neg_latent)
    L_cov = covariance_loss(y_pos_mu) + covariance_loss(y_neg_latent)
    representation = (
        L_recon
        + lambda_kl * L_kl
        + lambda_var * L_var
        + lambda_cov * L_cov
    )
    total = lambda_representation * representation + lambda_drift * L_drift
    loss_items = {
        "recon": L_recon.detach(),
        "kl": L_kl.detach(),
        "drift": L_drift.detach(),
        "V": V.detach(),
        "var": L_var.detach(),
        "covar": L_cov.detach(),
        "representation": representation.detach(),
        "representation_scaled": (lambda_representation * representation).detach(),
    }
    return total, loss_items
