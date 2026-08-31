"""The learned representation used inside the drift kernel."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class IdentityRepresentation(nn.Module):
    """Raw baseline: kernel distances use the original latent coordinates."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class PhiNetwork(nn.Module):
    """Learnable representation ``phi(z)`` used to calculate kernel distances.

    Phi does not replace the VAE latent sent to the decoder. It changes only the
    geometry used by the drift kernel. Dimensionality reduction is optional;
    ``feature_dim=None`` keeps the original latent dimension.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        feature_dim: int | None = None,
        hidden_dims: Sequence[int] = (128, 128),
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = latent_dim if feature_dim is None else feature_dim
        self.hidden_dims = tuple(hidden_dims)
        self.residual = residual
        if self.latent_dim < 1 or self.feature_dim < 1:
            raise ValueError("latent_dim and feature_dim must be positive")
        if any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")

        layers: list[nn.Module] = []
        input_dim = self.latent_dim
        for width in self.hidden_dims:
            layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
            input_dim = width
        layers.append(nn.Linear(input_dim, self.feature_dim))
        self.network = nn.Sequential(*layers)

        # A residual shortcut is safe only when the representation keeps the
        # same number of coordinates. Dimensionality-reducing phi networks use
        # the MLP alone, which is the path used by the first toy experiment.
        self.skip = (
            nn.Identity()
            if self.residual and self.latent_dim == self.feature_dim
            else None
        )
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return phi(z), the features used by the drift kernel."""

        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected input shape (batch, {self.latent_dim}), got {tuple(z.shape)}"
            )
        features = self.network(z)
        if self.skip is not None:
            features = features + self.skip(z)
        return features
