"""PyTorch loss helpers for the drift-based VAE experiments."""

import torch
import torch.nn.functional as F


def softmax(logits, dim):
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=dim, keepdim=True)

def compute_V(y_neg, y_pos, T, eps=1e-8):
    N = y_neg.shape[0]
    points = torch.cat([y_neg, y_pos],dim=0).to(torch.float32)

    dist = torch.cdist(y_neg, points)
    dist[:, :N] += torch.eye(N, device=dist.device) * 1e6

    logits = -dist/T # this is log(k(x,y)) for every pair
    A_row = softmax(logits, dim=1)
    A_col = softmax(logits, dim=0)
    A = torch.sqrt(A_row * A_col)
    # A = A_row
    
    A_neg = A[:, :N]
    A_pos = A[:, N:]

    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    V = drift_pos - drift_neg
    
    return V

def drift_loss(y_neg, y_pos, temp):
    with torch.no_grad():
        V = compute_V(y_neg, y_pos, temp)
        frozen_targets = (y_neg + V).detach().clone()
    return F.mse_loss(y_neg, frozen_targets), V


def recon_loss(y_pos_recon, y_pos):
    return F.mse_loss(y_pos_recon, y_pos, reduction='mean')


def kl_loss(mu, logvar): # normalized
    kl_per_sample = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=-1)

    return kl_per_sample.mean()


# see VICReg
def variance_loss(mu, gamma=1.5, eps=1e-4):
    std = torch.sqrt(mu.var(dim=0, unbiased=True) + eps)
    return torch.mean(F.relu(gamma - std))

def covariance_loss(latents):
    batch_size, dim = latents.shape
    latents = latents - latents.mean(dim=0, keepdim=True)
    cov = (latents.T @ latents) / (batch_size - 1)   # bug fix: was `@ z`
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / dim

def total_loss(y_pos_recon, y_pos, y_pos_mu, y_pos_logvar, y_neg_latent, y_pos_latent,temp,
                lambda_kl=0.01, lambda_drift=1.0, lambda_var=0.1, lambda_cov=0.1,
                lambda_representation=1.0):
    L_recon = recon_loss(y_pos_recon, y_pos)
    L_kl    = kl_loss(y_pos_mu, y_pos_logvar)
    L_drift, V = drift_loss(y_neg_latent, y_pos_latent, temp)
    L_var   = variance_loss(y_pos_mu) + variance_loss(y_neg_latent)
    L_cov = covariance_loss(y_pos_mu) + covariance_loss(y_neg_latent)
    representation = L_recon + lambda_kl*L_kl + lambda_var*L_var + lambda_cov*L_cov
    total = lambda_representation * representation + lambda_drift * L_drift
    loss_items = {"recon": L_recon.detach(), "kl": L_kl.detach(),
                    "drift": L_drift.detach(), "V": V.detach(),"var": L_var.detach(), "covar":L_cov.detach(),
                    "representation": representation.detach(),
                    "representation_scaled": (lambda_representation * representation).detach()}
    return total, loss_items



    
