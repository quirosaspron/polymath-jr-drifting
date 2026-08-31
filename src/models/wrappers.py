"""Composition helpers that make data flow explicit."""

from __future__ import annotations

import torch
from torch import nn


class LatentDriftModel(nn.Module):
    """Bundle a VAE, latent generator, and phi without conflating them."""

    def __init__(self, *, vae: nn.Module, generator: nn.Module, phi: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.generator = generator
        self.phi = phi

    def encode_real(self, images: torch.Tensor) -> torch.Tensor:
        """Encode real images to original VAE latents."""

        if hasattr(self.vae, "encode_mu"):
            return self.vae.encode_mu(images)
        latent = self.vae.get_latent(images)
        return latent[0] if isinstance(latent, tuple) else latent

    def generate_latent(self, noise: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Generate original VAE latents."""

        return self.generator(noise, condition)

    def drift_features(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply phi for kernel/drift calculations only."""

        return self.phi(latents)

    def generate_image(self, noise: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode generated original latents without routing them through phi."""

        latent = self.generate_latent(noise, condition)
        return self.vae.decode(latent)
