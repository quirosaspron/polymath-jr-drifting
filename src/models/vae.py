"""Ordinary MLP VAE extracted from the stable separate-training notebook."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn


class VAE(nn.Module):
    """VAE with an explicit latent interface for downstream drift experiments."""

    def __init__(
        self,
        *,
        input_shape: Sequence[int] = (784,),
        latent_dim: int = 16,
        hidden_dims: Sequence[int] = (192, 96),
        conditional_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.conditional_dim = conditional_dim
        input_dim = math.prod(self.input_shape)
        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim + conditional_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([nn.Linear(previous_dim, hidden_dim), nn.LeakyReLU(0.2)])
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(previous_dim, latent_dim)
        self.fc_logvar = nn.Linear(previous_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim + conditional_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([nn.Linear(previous_dim, hidden_dim), nn.LeakyReLU(0.2)])
            previous_dim = hidden_dim
        decoder_layers.extend([nn.Linear(previous_dim, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)

    def _condition(self, condition: Any, *, batch_size: int, device: Any) -> torch.Tensor | None:
        if self.conditional_dim == 0:
            return None
        if condition is None:
            raise ValueError("condition is required when conditional_dim is non-zero")
        condition_tensor = torch.as_tensor(condition, device=device, dtype=torch.float32)
        if condition_tensor.shape != (batch_size, self.conditional_dim):
            raise ValueError("condition must have shape (batch, conditional_dim)")
        return condition_tensor

    def encode(self, x: torch.Tensor, condition: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log variance."""

        flattened = x.reshape(x.shape[0], -1)
        condition_tensor = self._condition(
            condition, batch_size=x.shape[0], device=x.device
        )
        if condition_tensor is not None:
            flattened = torch.cat([flattened, condition_tensor], dim=1)
        hidden = self.encoder(flattened)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def encode_mu(self, x: torch.Tensor, condition: Any = None) -> torch.Tensor:
        """Return deterministic latent coordinates used as real drift samples."""

        return self.encode(x, condition)[0]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent with the reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def get_latent(
        self, x: torch.Tensor, condition: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Notebook-compatible alias returning mean, log variance, and sample."""

        mu, logvar = self.encode(x, condition)
        return mu, logvar, self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor, condition: Any = None) -> torch.Tensor:
        """Decode an original VAE latent, never a metric-head feature."""

        condition_tensor = self._condition(
            condition, batch_size=z.shape[0], device=z.device
        )
        decoder_input = z
        if condition_tensor is not None:
            decoder_input = torch.cat([decoder_input, condition_tensor], dim=1)
        reconstruction = self.decoder(decoder_input)
        if len(self.input_shape) > 1:
            reconstruction = reconstruction.reshape(z.shape[0], *self.input_shape)
        return reconstruction

    def forward(self, x: torch.Tensor, condition: Any = None) -> dict[str, torch.Tensor]:
        """Return reconstruction and posterior tensors needed by the VAE loss."""

        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decode(z, condition),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
