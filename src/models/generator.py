"""Conditional latent generator extracted from the separate-training notebook."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class LatentGenerator(nn.Module):
    """Map noise, and optionally a condition, to a VAE-compatible latent."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        noise_dim: int | None = None,
        hidden_dims: Sequence[int] = (48, 48, 48),
        num_classes: int | None = 10,
        label_dim: int = 16,
    ) -> None:
        super().__init__()
        self.noise_dim = latent_dim if noise_dim is None else noise_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.num_classes = num_classes
        self.label_dim = label_dim if num_classes is not None else 0
        self.embedding = (
            None if num_classes is None else nn.Embedding(num_classes, label_dim)
        )
        layers: list[nn.Module] = []
        previous_dim = self.noise_dim + self.label_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.SiLU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, latent_dim))
        self.generator = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a latent that can be passed directly to ``VAE.decode``."""

        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError("noise must have shape (batch, noise_dim)")
        generator_input = noise
        if self.embedding is not None:
            if condition is None:
                raise ValueError("class labels are required for a conditional generator")
            labels = condition.to(device=noise.device, dtype=torch.long).reshape(-1)
            if labels.shape[0] != noise.shape[0]:
                raise ValueError("noise and labels must have the same batch size")
            generator_input = torch.cat([noise, self.embedding(labels)], dim=1)
        elif condition is not None:
            raise ValueError("condition was supplied to an unconditional generator")
        return self.generator(generator_input)
