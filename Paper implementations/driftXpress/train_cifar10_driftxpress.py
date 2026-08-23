# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pillow==12.3.0",
#     "timm==1.0.28",
#     "torch==2.13.0",
#     "torchvision==0.28.0",
#     "tqdm==4.70.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import sys, subprocess

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-U",
        "timm>=1.0.20",
        "torchmetrics[image]",
        "torch-fidelity",
        "tqdm",
    ])
    return


@app.cell(hide_code=True)
def _():
    """Train a scalable one-step CIFAR-10 DriftXpress generator.

    This is a direct image-space adaptation of ``Paper implementations/Drifting/
    image_gen.ipynb``.  It intentionally does *not* use a VAE: Gaussian image
    noise is mapped to an RGB image in one generator evaluation.

    The implementation follows the two important pieces of DriftXpress:

    1. a frozen learned image encoder defines the space in which the drift acts;
    2. attraction to the data distribution is cached with a Nyström approximation,
       while repulsion among current generated samples remains exact.

    The default is an unconditional CIFAR-10 run.  That is the appropriate mode
    for a comparable FID.  ``--class-index 4`` is available for the earlier
    deer-only experiment, but its FID must not be compared to full-CIFAR FID.

    Blackwell / multi-GPU example (eight GPUs):

        torchrun --standalone --nproc_per_node=8 train_cifar10_driftxpress.py \
            --output-dir runs/cifar10_dinoxpress --batch-size 128 --steps 100000

    Dependencies (install in the training environment):

        pip install torch torchvision timm torchmetrics[image] tqdm

    The first run downloads CIFAR-10, the frozen DINOv3 ViT-B/16 weights, and the
    Inception weights used by torchmetrics FID.  The Nyström cache is saved in the
    run directory and reused by later resumed runs.
    """
    from __future__ import annotations
    import argparse
    import copy
    import json
    import math
    import os
    import random
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Iterable, Sequence

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from tqdm.auto import tqdm
    print(torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    IMAGE_SIZE = 32
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)
    CACHE_VERSION = 1


    # -----------------------------------------------------------------------------
    # Distributed utilities
    # -----------------------------------------------------------------------------


    @dataclass(frozen=True)
    class DistributedContext:
        rank: int
        local_rank: int
        world_size: int
        device: torch.device

        @property
        def is_distributed(self) -> bool:
            return self.world_size > 1

        @property
        def is_main(self) -> bool:
            return self.rank == 0


    def init_distributed() -> DistributedContext:
        """Support both ``python`` and ``torchrun`` launches."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA GPUs.")
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return DistributedContext(rank, local_rank, world_size, torch.device("cuda", local_rank))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(0, 0, 1, device)


    def barrier(ctx: DistributedContext) -> None:
        if ctx.is_distributed:
            dist.barrier()


    def rank_print(ctx: DistributedContext, *values: object, **kwargs: object) -> None:
        if ctx.is_main:
            print(*values, **kwargs)


    def unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model


    def seed_everything(seed: int, rank: int) -> None:
        # Each rank gets distinct training noise, while landmark selection below
        # uses its own fixed CPU generator and is therefore identical on all ranks.
        seed = seed + rank
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


    # -----------------------------------------------------------------------------
    # CIFAR-10 data
    # -----------------------------------------------------------------------------


    def cifar10_dataset(
        root: str | Path,
        *,
        train: bool,
        download: bool,
        class_index: int,
    ) -> Dataset:
        """Return CIFAR-10 in [0, 1], optionally restricted to one class."""
        base = datasets.CIFAR10(
            root=str(root),
            train=train,
            transform=transforms.ToTensor(),
            download=download,
        )
        if class_index < 0:
            return base
        if not 0 <= class_index < 10:
            raise ValueError("class_index must be -1 (all classes) or an integer in [0, 9].")
        indices = [i for i, label in enumerate(base.targets) if label == class_index]
        return Subset(base, indices)


    def loader_kwargs(args: argparse.Namespace, ctx: DistributedContext) -> dict:
        workers = int(args.num_workers)
        return {
            "num_workers": workers,
            "pin_memory": ctx.device.type == "cuda",
            "persistent_workers": workers > 0,
            "prefetch_factor": 4 if workers > 0 else None,
        }


    # -----------------------------------------------------------------------------
    # Roughly 30M-parameter one-step image generator
    # -----------------------------------------------------------------------------


    def group_count(channels: int) -> int:
        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1


    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.skip = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv1(F.silu(self.norm1(x)))
            h = self.conv2(F.silu(self.norm2(h)))
            # A mild residual rescaling is helpful because this network has many
            # blocks but no diffusion timestep conditioning.
            return (self.skip(x) + h) * (2.0 ** -0.5)


    class SpatialAttention(nn.Module):
        """Small full attention block, used only at 8x8 and 4x4 resolutions."""

        def __init__(self, channels: int, heads: int = 8):
            super().__init__()
            if channels % heads != 0:
                raise ValueError(f"channels={channels} must divide heads={heads}")
            self.channels = channels
            self.heads = heads
            self.norm = nn.GroupNorm(group_count(channels), channels)
            self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, channels, height, width = x.shape
            h = self.qkv(self.norm(x))
            q, k, v = h.chunk(3, dim=1)
            head_dim = channels // self.heads
            q = q.reshape(batch, self.heads, head_dim, height * width).transpose(-1, -2)
            k = k.reshape(batch, self.heads, head_dim, height * width).transpose(-1, -2)
            v = v.reshape(batch, self.heads, head_dim, height * width).transpose(-1, -2)
            h = F.scaled_dot_product_attention(q, k, v)
            h = h.transpose(-1, -2).reshape(batch, channels, height, width)
            return x + self.proj(h)


    class EncoderStage(nn.Module):
        def __init__(self, channels: int, blocks: int, use_attention: bool):
            super().__init__()
            self.blocks = nn.Sequential(*[ResidualBlock(channels, channels) for _ in range(blocks)])
            self.attention = SpatialAttention(channels) if use_attention else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attention(self.blocks(x))


    class DecoderStage(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int, use_attention: bool):
            super().__init__()
            self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            body: list[nn.Module] = [ResidualBlock(out_channels + skip_channels, out_channels)]
            body.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
            self.blocks = nn.Sequential(*body)
            self.attention = SpatialAttention(out_channels) if use_attention else nn.Identity()

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.pre(x)
            x = torch.cat((x, skip), dim=1)
            return self.attention(self.blocks(x))


    class DriftUNet(nn.Module):
        """Direct Gaussian-image-noise -> RGB generator (about 30M parameters)."""

        def __init__(self, base_channels: int = 88, blocks_per_stage: int = 3):
            super().__init__()
            c0, c1, c2, c3 = (
                base_channels,
                base_channels * 2,
                base_channels * 3,
                base_channels * 4,
            )
            self.in_conv = nn.Conv2d(3, c0, kernel_size=3, padding=1)
            self.enc0 = EncoderStage(c0, blocks_per_stage, use_attention=False)  # 32x32
            self.down0 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1)
            self.enc1 = EncoderStage(c1, blocks_per_stage, use_attention=False)  # 16x16
            self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
            self.enc2 = EncoderStage(c2, blocks_per_stage, use_attention=True)   # 8x8
            self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
            self.enc3 = EncoderStage(c3, blocks_per_stage, use_attention=True)   # 4x4

            self.mid = nn.Sequential(
                ResidualBlock(c3, c3), SpatialAttention(c3), ResidualBlock(c3, c3), ResidualBlock(c3, c3)
            )

            self.dec2 = DecoderStage(c3, c2, c2, blocks_per_stage, use_attention=True)   # 8x8
            self.dec1 = DecoderStage(c2, c1, c1, blocks_per_stage, use_attention=False)  # 16x16
            self.dec0 = DecoderStage(c1, c0, c0, blocks_per_stage, use_attention=False)  # 32x32
            self.out = nn.Sequential(
                nn.GroupNorm(group_count(c0), c0),
                nn.SiLU(),
                nn.Conv2d(c0, 3, kernel_size=3, padding=1),
                nn.Tanh(),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, noise: torch.Tensor) -> torch.Tensor:
            x0 = self.enc0(self.in_conv(noise))
            x1 = self.enc1(self.down0(x0))
            x2 = self.enc2(self.down1(x1))
            x3 = self.enc3(self.down2(x2))
            h = self.mid(x3)
            h = self.dec2(h, x2)
            h = self.dec1(h, x1)
            h = self.dec0(h, x0)
            return self.out(h)


    def parameter_count(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())


    # -----------------------------------------------------------------------------
    # Frozen DINOv3 image feature encoder
    # -----------------------------------------------------------------------------


    class DINOv3FeatureEncoder(nn.Module):
        """Frozen multi-layer DINOv3 ViT-B/16 features used by the drift field.

        Parameters are frozen, but fake-image gradients are intentionally retained:
        the generator needs a gradient through this encoder.  Real-image features
        are evaluated under ``torch.no_grad`` by the cache builder/training loop.
        """

        def __init__(
            self,
            model_name: str,
            input_size: int,
            pool_size: int,
            extra_statistics: bool,
        ):
            super().__init__()
            try:
                import timm
            except ImportError as error:
                raise ImportError("Install timm to use the frozen DINOv3 encoder: pip install timm") from error

            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

            self.input_size = input_size
            self.pool_size = pool_size
            self.extra_statistics = extra_statistics
            self.patch_size = 16
            self.layer_indices = (2, 5, 8, 11)
            self.register_buffer("mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(DINO_STD).view(1, 3, 1, 1))

        def train(self, mode: bool = True) -> "DINOv3FeatureEncoder":
            # Keep BatchNorm/dropout-like behavior (if present) frozen and stable.
            super().train(False)
            self.model.eval()
            return self

        def _as_spatial_map(self, feature: torch.Tensor, target_size: int) -> torch.Tensor:
            if feature.ndim == 4:
                return feature
            if feature.ndim != 3:
                raise ValueError(f"Unexpected DINO intermediate feature shape: {tuple(feature.shape)}")
            grid = target_size // self.patch_size
            spatial_tokens = grid * grid
            # Some timm variants include a prefix token; retaining the final patch
            # tokens is robust to either representation.
            feature = feature[:, -spatial_tokens:, :]
            return feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[2], grid, grid)

        def _intermediate_layers(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
            output = self.model.forward_intermediates(x, indices=self.layer_indices)
            if isinstance(output, tuple):
                output = output[-1]
            if torch.is_tensor(output):
                return (output,)
            return output

        def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
            # The generator emits [-1, 1].  DINO expects ImageNet-normalized [0, 1].
            rgb = (images + 1.0) * 0.5
            x = (rgb - self.mean) / self.std
            target_size = (self.input_size // self.patch_size) * self.patch_size
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
            intermediates = self._intermediate_layers(x)

            groups: list[torch.Tensor] = []
            if self.extra_statistics:
                # Direct color second-moment feature; this helps the drift control
                # color/contrast without introducing a pixel-space reconstruction loss.
                groups.append(rgb.square().mean(dim=(2, 3), keepdim=False).unsqueeze(1))

            for feature in intermediates:
                fmap = self._as_spatial_map(feature, target_size)
                pooled = F.adaptive_avg_pool2d(fmap, (self.pool_size, self.pool_size))
                groups.append(pooled.flatten(2).transpose(1, 2))
                if self.extra_statistics:
                    groups.append(fmap.mean(dim=(2, 3)).unsqueeze(1))
                    groups.append(fmap.flatten(2).std(dim=2, unbiased=False).unsqueeze(1))
                    for patch in (2, 4):
                        mean = F.avg_pool2d(pooled, kernel_size=patch, stride=patch)
                        second_moment = F.avg_pool2d(pooled.square(), kernel_size=patch, stride=patch)
                        std = (second_moment - mean.square()).clamp_min(1e-8).sqrt()
                        groups.append(mean.flatten(2).transpose(1, 2))
                        groups.append(std.flatten(2).transpose(1, 2))
            return groups


    # -----------------------------------------------------------------------------
    # Projected RKHS / Nyström DriftXpress cache
    # -----------------------------------------------------------------------------


    @dataclass
    class NystromStats:
        # All tensors use [locations, landmarks, feature_dimension] conventions.
        landmarks: torch.Tensor
        inverse_sqrt_kernel: torch.Tensor
        positive_totals: torch.Tensor
        positive_weighted_points: torch.Tensor
        temperature: float

        def to(self, device: torch.device) -> "NystromStats":
            return NystromStats(
                landmarks=self.landmarks.to(device=device, dtype=torch.float32, non_blocking=True),
                inverse_sqrt_kernel=self.inverse_sqrt_kernel.to(device=device, dtype=torch.float32, non_blocking=True),
                positive_totals=self.positive_totals.to(device=device, dtype=torch.float32, non_blocking=True),
                positive_weighted_points=self.positive_weighted_points.to(device=device, dtype=torch.float32, non_blocking=True),
                temperature=float(self.temperature),
            )

        def cpu_dict(self) -> dict:
            return {
                "landmarks": self.landmarks.detach().cpu(),
                "inverse_sqrt_kernel": self.inverse_sqrt_kernel.detach().cpu(),
                "positive_totals": self.positive_totals.detach().cpu(),
                "positive_weighted_points": self.positive_weighted_points.detach().cpu(),
                "temperature": self.temperature,
            }

        @classmethod
        def from_dict(cls, value: dict) -> "NystromStats":
            return cls(
                landmarks=value["landmarks"],
                inverse_sqrt_kernel=value["inverse_sqrt_kernel"],
                positive_totals=value["positive_totals"],
                positive_weighted_points=value["positive_weighted_points"],
                temperature=float(value["temperature"]),
            )


    def laplacian_kernel(x: torch.Tensor, y: torch.Tensor, temperature: float, eps: float = 1e-8) -> torch.Tensor:
        """Reference DriftXpress Laplacian kernel for batched feature locations.

        x and y have shape [locations, points, dimension].  Scaling the bandwidth
        by dimension is the reference implementation's convention.
        """
        dimension = x.shape[-1]
        bandwidth = max(float(temperature) * float(dimension), eps)
        return torch.exp(-torch.cdist(x.float(), y.float(), p=2) / bandwidth)


    def inverse_sqrt_psd(matrices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices.float())
        inverse_eigenvalues = eigenvalues.clamp_min(eps).rsqrt()
        return (eigenvectors * inverse_eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


    def nystrom_features(points: torch.Tensor, stats: NystromStats) -> torch.Tensor:
        """Explicit features phi(x) = K(x,U) (W + ridge I)^(-1/2)."""
        kernel = laplacian_kernel(points, stats.landmarks, stats.temperature)
        return torch.bmm(kernel, stats.inverse_sqrt_kernel)


    def prepare_stats_for_landmarks(
        landmark_group: torch.Tensor,
        temperature: float,
        ridge: float,
        device: torch.device,
    ) -> NystromStats:
        # [M, L, D] -> [L, M, D]
        landmarks = landmark_group.transpose(0, 1).contiguous().to(device=device, dtype=torch.float32)
        locations, landmark_count, dimension = landmarks.shape
        kernel = laplacian_kernel(landmarks, landmarks, temperature)
        eye = torch.eye(landmark_count, device=device, dtype=kernel.dtype).expand(locations, -1, -1)
        inverse_sqrt = inverse_sqrt_psd(kernel + ridge * eye)
        return NystromStats(
            landmarks=landmarks,
            inverse_sqrt_kernel=inverse_sqrt,
            positive_totals=torch.zeros(locations, landmark_count, device=device, dtype=torch.float32),
            positive_weighted_points=torch.zeros(locations, landmark_count, dimension, device=device, dtype=torch.float32),
            temperature=float(temperature),
        )


    def cache_filename(args: argparse.Namespace) -> str:
        scope = "all" if args.class_index < 0 else f"class{args.class_index}"
        model_name = args.encoder_model.replace("/", "_").replace(".", "_")
        return f"nystrom_v{CACHE_VERSION}_{scope}_{model_name}_m{args.landmarks}_t{args.temperature:.4f}.pt"


    def _load_tensor_file(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before ``weights_only`` existed.
            return torch.load(path, map_location="cpu")


    @torch.no_grad()
    def encode_landmarks(
        dataset: Dataset,
        indices: torch.Tensor,
        encoder: DINOv3FeatureEncoder,
        args: argparse.Namespace,
        ctx: DistributedContext,
    ) -> list[torch.Tensor]:
        subset = Subset(dataset, indices.tolist())
        loader = DataLoader(
            subset,
            batch_size=args.cache_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs(args, ctx),
        )
        groups: list[list[torch.Tensor]] | None = None
        for images, _ in loader:
            images = images.to(ctx.device, non_blocking=True).mul(2.0).sub(1.0)
            batch_groups = encoder(images)
            if groups is None:
                groups = [[] for _ in batch_groups]
            for bucket, values in zip(groups, batch_groups):
                bucket.append(values.float().cpu())
        if groups is None:
            raise RuntimeError("No landmark images were encoded.")
        return [torch.cat(bucket, dim=0) for bucket in groups]


    @torch.no_grad()
    def build_nystrom_cache(
        dataset: Dataset,
        encoder: DINOv3FeatureEncoder,
        args: argparse.Namespace,
        ctx: DistributedContext,
        progress: bool = True,
    ) -> list[NystromStats]:
        """Build the global data-attraction cache in distributed shards.

        Each GPU encodes one shard of the real dataset.  The sufficient statistics
        are then all-reduced, yielding one identical cache on every rank without
        ever gathering all 50k DINO features in host memory.
        """
        landmark_count = min(args.landmarks, len(dataset))
        landmark_rng = torch.Generator(device="cpu").manual_seed(args.seed + 173)
        landmark_indices = torch.randperm(len(dataset), generator=landmark_rng)[:landmark_count]
        rank_print(ctx, f"Preparing {landmark_count} Nyström landmarks and real-data summaries …")
        landmark_groups = encode_landmarks(dataset, landmark_indices, encoder, args, ctx)
        stats_groups = [
            prepare_stats_for_landmarks(group, args.temperature, args.ridge, ctx.device)
            for group in landmark_groups
        ]
        del landmark_groups
        torch.cuda.empty_cache()

        local_indices = list(range(ctx.rank, len(dataset), ctx.world_size))
        local_dataset = Subset(dataset, local_indices)
        real_loader = DataLoader(
            local_dataset,
            batch_size=args.cache_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs(args, ctx),
        )
        iterator: Iterable = real_loader
        if ctx.is_main and progress:
            iterator = tqdm(real_loader, desc="Caching real DINO features", unit="batch", mininterval=1.0)

        for images, _ in iterator:
            images = images.to(ctx.device, non_blocking=True).mul(2.0).sub(1.0)
            groups = encoder(images)
            if len(groups) != len(stats_groups):
                raise RuntimeError("DINO feature-group count changed while building the cache.")
            for group, stats in zip(groups, stats_groups):
                points = group.float().transpose(0, 1).contiguous()  # [L, B, D]
                phi = nystrom_features(points, stats)
                stats.positive_totals.add_(phi.sum(dim=1))
                stats.positive_weighted_points.add_(torch.bmm(phi.transpose(1, 2), points))

        if ctx.is_distributed:
            for stats in stats_groups:
                dist.all_reduce(stats.positive_totals, op=dist.ReduceOp.SUM)
                dist.all_reduce(stats.positive_weighted_points, op=dist.ReduceOp.SUM)

        return stats_groups


    def load_or_create_nystrom_cache(
        dataset: Dataset,
        encoder: DINOv3FeatureEncoder,
        args: argparse.Namespace,
        ctx: DistributedContext,
    ) -> list[NystromStats]:
        cache_path = Path(args.output_dir) / cache_filename(args)
        payload: dict | None = None
        if cache_path.exists() and not args.rebuild_cache:
            try:
                payload = _load_tensor_file(cache_path)
                expected = {
                    "version": CACHE_VERSION,
                    "landmarks": args.landmarks,
                    "temperature": args.temperature,
                    "ridge": args.ridge,
                    "dataset_size": len(dataset),
                    "extra_statistics": args.extra_feature_statistics,
                }
                if any(payload.get(key) != value for key, value in expected.items()):
                    rank_print(ctx, "Existing Nyström cache configuration differs; rebuilding it.")
                    payload = None
            except Exception as error:  # A partial/interrupted cache should not block a new run.
                rank_print(ctx, f"Could not read Nyström cache ({error}); rebuilding it.")
                payload = None

        if payload is not None:
            rank_print(ctx, f"Loading Nyström attraction cache: {cache_path}")
            return [NystromStats.from_dict(item).to(ctx.device) for item in payload["stats"]]

        stats = build_nystrom_cache(dataset, encoder, args, ctx)
        if ctx.is_main:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": CACHE_VERSION,
                    "landmarks": args.landmarks,
                    "temperature": args.temperature,
                    "ridge": args.ridge,
                    "dataset_size": len(dataset),
                    "extra_statistics": args.extra_feature_statistics,
                    "stats": [item.cpu_dict() for item in stats],
                },
                cache_path,
            )
            print(f"Saved Nyström attraction cache: {cache_path}")
        barrier(ctx)
        return stats


    # -----------------------------------------------------------------------------
    # Stop-gradient DriftXpress loss
    # -----------------------------------------------------------------------------


    def gather_global_detached(features: torch.Tensor, ctx: DistributedContext) -> tuple[torch.Tensor, slice]:
        """Gather current generated features for exact global-batch repulsion.

        The drift field is explicitly stop-gradient, so gathering detached features
        is correct: no cross-rank autograd graph is needed or desired.
        """
        local_batch = features.shape[0]
        if not ctx.is_distributed:
            return features.detach(), slice(0, local_batch)
        gathered = [torch.empty_like(features) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, features.detach().contiguous())
        start = ctx.rank * local_batch
        return torch.cat(gathered, dim=0), slice(start, start + local_batch)


    def normalize_drift(drift: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize each feature location so multiscale groups contribute evenly."""
        dimension = drift.shape[-1]
        scale = torch.sqrt((drift.float().square().sum(dim=-1) / dimension).mean(dim=-1)).detach()
        return drift / (scale[:, None, None] + eps)


    @torch.no_grad()
    def nystrom_drift_target(
        global_features: torch.Tensor,
        stats: NystromStats,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return one full stop-gradient drift step for every generated feature.

        ``global_features`` is [B_global, locations, D].  Attraction uses the
        precomputed whole-dataset summaries; repulsion is exact over the generated
        global batch, with self-interaction removed.
        """
        points = global_features.float().transpose(0, 1).contiguous()  # [L, B, D]
        phi = nystrom_features(points, stats)
        positive_num = torch.bmm(phi, stats.positive_weighted_points)
        positive_den = (phi * stats.positive_totals[:, None, :]).sum(dim=-1, keepdim=True).clamp_min(eps)
        positive_barycenter = positive_num / positive_den

        distances = torch.cdist(points, points, p=2)
        sample_count = points.shape[1]
        distances = distances + torch.eye(sample_count, device=points.device, dtype=distances.dtype)[None] * 1e6
        bandwidth = max(stats.temperature * points.shape[-1], eps)
        weights = torch.exp(-distances / bandwidth)
        negative_barycenter = torch.bmm(weights, points) / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

        drift = normalize_drift(positive_barycenter - negative_barycenter, eps=eps)
        return global_features.float() + drift.transpose(0, 1)


    def drifting_loss(
        fake_groups: Sequence[torch.Tensor],
        stats_groups: Sequence[NystromStats],
        ctx: DistributedContext,
    ) -> torch.Tensor:
        if len(fake_groups) != len(stats_groups):
            raise RuntimeError("Generated and cached feature groups do not match.")
        total = fake_groups[0].new_zeros((), dtype=torch.float32)
        for fake_group, stats in zip(fake_groups, stats_groups):
            global_group, local_slice = gather_global_detached(fake_group, ctx)
            target = nystrom_drift_target(global_group, stats)[local_slice]
            # ``target`` is completely detached.  This is Eq. (5)-style regression
            # to one frozen drift step, not a differentiable field objective.
            total = total + F.mse_loss(fake_group.float(), target)
        return total / len(fake_groups)


    # -----------------------------------------------------------------------------
    # EMA, sampling, FID, checkpoints
    # -----------------------------------------------------------------------------


    class EMA:
        def __init__(self, model: nn.Module, decay: float):
            self.decay = decay
            self.model = copy.deepcopy(model).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        @torch.no_grad()
        def update(self, model: nn.Module) -> None:
            for ema_value, current_value in zip(self.model.state_dict().values(), model.state_dict().values()):
                if ema_value.is_floating_point():
                    ema_value.lerp_(current_value.detach(), 1.0 - self.decay)
                else:
                    ema_value.copy_(current_value)


    @torch.no_grad()
    def save_samples(model: nn.Module, path: Path, device: torch.device, count: int = 64) -> None:
        model.eval()
        noise = torch.randn(count, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        images = model(noise).add(1.0).mul(0.5).clamp(0, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(images, path, nrow=int(math.sqrt(count)))


    @torch.no_grad()
    def evaluate_fid(
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        samples: int,
        batch_size: int,
        workers: int,
    ) -> float:
        """Compute a 10k/50k-sample Inception FID on one rank.

        The result is only benchmark-comparable when using the same dataset split
        and sample count as the comparison.  Full CIFAR-10's 10k test split is the
        sensible default; class-restricted runs are diagnostic only.
        """
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError as error:
            raise ImportError("FID requires: pip install 'torchmetrics[image]'") from error

        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        real_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        seen = 0
        for real, _ in tqdm(real_loader, desc="FID real", leave=False, mininterval=1.0):
            if seen >= samples:
                break
            real = real[: samples - seen].to(device, non_blocking=True)
            metric.update(real, real=True)
            seen += real.shape[0]

        model.eval()
        generated = 0
        while generated < samples:
            n = min(batch_size, samples - generated)
            noise = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            fake = model(noise).add(1.0).mul(0.5).clamp(0, 1)
            metric.update(fake, real=False)
            generated += n
        return float(metric.compute().item())


    def checkpoint_payload(
        model: nn.Module,
        ema: EMA,
        optimizer: torch.optim.Optimizer,
        step: int,
        args: argparse.Namespace,
    ) -> dict:
        return {
            "step": step,
            "model": unwrap(model).state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }


    def cosine_lr(step: int, args: argparse.Namespace) -> float:
        if step <= args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1.0 + math.cos(math.pi * progress))


    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------


    def train(args: argparse.Namespace) -> None:
        ctx = init_distributed()
        try:
            seed_everything(args.seed, ctx.rank)
            output_dir = Path(args.output_dir)
            if ctx.is_main:
                (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                (output_dir / "samples").mkdir(parents=True, exist_ok=True)
                with (output_dir / "config.json").open("w", encoding="utf-8") as file:
                    json.dump(vars(args), file, indent=2, sort_keys=True)
            barrier(ctx)

            train_data = cifar10_dataset(
                args.data_root,
                train=True,
                download=args.download,
                class_index=args.class_index,
            )
            fid_data = cifar10_dataset(
                args.data_root,
                train=False,
                download=args.download,
                class_index=args.class_index,
            )
            rank_print(ctx, f"Training images: {len(train_data):,}; FID reference images: {len(fid_data):,}")

            generator = DriftUNet(base_channels=args.base_channels, blocks_per_stage=args.blocks_per_stage).to(ctx.device)
            parameters = parameter_count(generator)
            rank_print(ctx, f"Generator parameters: {parameters / 1e6:.2f}M")
            if not 25_000_000 <= parameters <= 40_000_000:
                rank_print(ctx, "Warning: generator parameter count is outside the intended roughly-30M range.")

            ema = EMA(generator, args.ema_decay)
            optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
            start_step = 0
            if args.resume:
                checkpoint = _load_tensor_file(Path(args.resume))
                generator.load_state_dict(checkpoint["model"])
                ema.model.load_state_dict(checkpoint["ema"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_step = int(checkpoint["step"])
                rank_print(ctx, f"Resumed checkpoint {args.resume} from step {start_step:,}")

            encoder = DINOv3FeatureEncoder(
                model_name=args.encoder_model,
                input_size=args.encoder_size,
                pool_size=args.pool_size,
                extra_statistics=args.extra_feature_statistics,
            ).to(ctx.device)
            encoder.eval()
            stats_groups = load_or_create_nystrom_cache(train_data, encoder, args, ctx)
            rank_print(ctx, f"Drift uses {len(stats_groups)} frozen DINO feature groups.")

            if ctx.is_distributed:
                generator = DDP(
                    generator,
                    device_ids=[ctx.local_rank],
                    output_device=ctx.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )

            sampler = DistributedSampler(
                train_data,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                drop_last=True,
            ) if ctx.is_distributed else None
            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=sampler is None,
                drop_last=True,
                **loader_kwargs(args, ctx),
            )
            if len(train_loader) == 0:
                raise RuntimeError("Training loader is empty; reduce --batch-size or use all CIFAR-10.")

            global_batch = args.batch_size * ctx.world_size
            nominal_epochs = args.steps * global_batch / len(train_data)
            rank_print(
                ctx,
                f"Training {args.steps:,} steps = roughly {nominal_epochs:,.0f} dataset epochs "
                f"at global batch {global_batch}.  Precision: {'bf16' if args.amp else 'fp32'}.",
            )

            loader_epoch = 0
            if sampler is not None:
                sampler.set_epoch(loader_epoch)
            loader_iterator = iter(train_loader)
            metric_file = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if ctx.is_main else None
            progress = tqdm(
                total=args.steps,
                initial=start_step,
                disable=not ctx.is_main,
                desc="DriftXpress",
                unit="step",
                mininterval=1.0,
            )
            start_time = time.time()

            for step in range(start_step + 1, args.steps + 1):
                try:
                    real, _ = next(loader_iterator)
                except StopIteration:
                    loader_epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(loader_epoch)
                    loader_iterator = iter(train_loader)
                    real, _ = next(loader_iterator)
                batch_size = real.shape[0]
                lr = cosine_lr(step, args)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                noise = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=ctx.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and ctx.device.type == "cuda"):
                    fake = generator(noise)
                    # The encoder's weights are frozen, but this forward *must* keep
                    # gradients from all features back to the generator output.
                    fake_groups = encoder(fake)
                loss = drifting_loss(fake_groups, stats_groups, ctx)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite drift loss at step {step}: {loss.item()}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(unwrap(generator))

                log_loss = loss.detach().float().clone()
                if ctx.is_distributed:
                    dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
                if ctx.is_main:
                    progress.update(1)
                    progress.set_postfix(loss=f"{log_loss.item():.5f}", lr=f"{lr:.2e}")

                should_log = step % args.log_every == 0 or step == 1 or step == args.steps
                if should_log and ctx.is_main:
                    elapsed = time.time() - start_time
                    record = {
                        "step": step,
                        "drift_loss": log_loss.item(),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "images_per_second": global_batch * step / max(elapsed, 1e-6),
                    }
                    print(json.dumps(record))
                    assert metric_file is not None
                    metric_file.write(json.dumps(record) + "\n")
                    metric_file.flush()

                if ctx.is_main and (step % args.sample_every == 0 or step == args.steps):
                    save_samples(ema.model, output_dir / "samples" / f"ema_step_{step:07d}.png", ctx.device)

                if ctx.is_main and (step % args.save_every == 0 or step == args.steps):
                    payload = checkpoint_payload(generator, ema, optimizer, step, args)
                    latest_path = output_dir / "checkpoints" / "last.pt"
                    torch.save(payload, latest_path)
                    if step % args.keep_every == 0 or step == args.steps:
                        torch.save(payload, output_dir / "checkpoints" / f"step_{step:07d}.pt")

                if ctx.is_main and args.fid_every > 0 and (step % args.fid_every == 0 or step == args.steps):
                    fid = evaluate_fid(
                        ema.model,
                        fid_data,
                        ctx.device,
                        samples=args.fid_samples,
                        batch_size=args.fid_batch_size,
                        workers=args.num_workers,
                    )
                    record = {"step": step, "ema_fid": fid, "fid_samples": args.fid_samples}
                    print(json.dumps(record))
                    assert metric_file is not None
                    metric_file.write(json.dumps(record) + "\n")
                    metric_file.flush()

            if ctx.is_main:
                progress.close()
                if metric_file is not None:
                    metric_file.close()
                print(f"Finished. Checkpoints: {output_dir / 'checkpoints'}")
        finally:
            if ctx.is_distributed and dist.is_initialized():
                dist.destroy_process_group()


    # -----------------------------------------------------------------------------
    # Command line
    # -----------------------------------------------------------------------------


    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--data-root", type=str, default="./data", help="Directory holding torchvision's CIFAR-10 files.")
        parser.add_argument("--output-dir", type=str, required=True, help="Run directory for cache, samples, metrics, and checkpoints.")
        parser.add_argument("--class-index", type=int, default=-1, help="-1 trains all CIFAR-10 classes; 4 is deer-only.")
        parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument("--steps", type=int, default=100_000, help="50k matches the reference release; 100k is the long default.")
        parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size. Eight GPUs yield global batch 1024.")
        parser.add_argument("--num-workers", type=int, default=8)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--lr-final", type=float, default=2e-5)
        parser.add_argument("--warmup-steps", type=int, default=2_000)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--grad-clip", type=float, default=1.0)
        parser.add_argument("--ema-decay", type=float, default=0.9999)
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 autocast on CUDA.")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--resume", type=str, default=None)

        parser.add_argument("--base-channels", type=int, default=88, help="Default model is about 30M parameters.")
        parser.add_argument("--blocks-per-stage", type=int, default=3)
        parser.add_argument("--encoder-model", type=str, default="vit_base_patch16_dinov3.lvd1689m")
        parser.add_argument("--encoder-size", type=int, default=112)
        parser.add_argument("--pool-size", type=int, default=4)
        parser.add_argument("--extra-feature-statistics", action=argparse.BooleanOptionalAction, default=True)

        parser.add_argument("--temperature", type=float, default=0.05, help="Reference CIFAR DriftXpress Laplacian-kernel temperature.")
        parser.add_argument("--landmarks", type=int, default=512, help="Nyström landmark count. 256 is a cheaper debugging setting.")
        parser.add_argument("--ridge", type=float, default=1e-4)
        parser.add_argument("--cache-batch-size", type=int, default=256)
        parser.add_argument("--rebuild-cache", action="store_true")

        parser.add_argument("--log-every", type=int, default=100)
        parser.add_argument("--sample-every", type=int, default=2_000)
        parser.add_argument("--save-every", type=int, default=2_000)
        parser.add_argument("--keep-every", type=int, default=10_000)
        parser.add_argument("--fid-every", type=int, default=10_000, help="0 disables FID during training.")
        parser.add_argument("--fid-samples", type=int, default=10_000)
        parser.add_argument("--fid-batch-size", type=int, default=256)
        return parser


    if __name__ == "__main__":
        parser = build_parser()
        args = build_parser().parse_args([
        "--output-dir", "runs/cifar10_molab",
        "--steps", "20000",
        "--batch-size", "256",
        "--landmarks", "512",
        "--save-every", "1000",
        "--keep-every", "5000",
        "--num-workers", "0",
        "--sample-every", "1000",
        "--fid-every", "5000",
    ])
        train(args)
    return (
        DINOv3FeatureEncoder,
        DataLoader,
        DriftUNet,
        F,
        Path,
        args,
        datasets,
        evaluate_fid,
        json,
        os,
        torch,
        transforms,
    )


@app.cell
def _():
    import subprocess, sys

    subprocess.run([
        sys.executable, "train_cifar10_driftxpress.py",
        "--output-dir", "runs/cifar10_smoke",
        "--steps", "1000",
        "--batch-size", "64",
        "--landmarks", "128",
        "--fid-every", "0",
    ], check=True)
    return


@app.cell
def _(Path, args, os):
    print("Current folder:", os.getcwd())
    print("Training output folder:", args.output_dir)

    print("Contents:")
    for path in sorted(Path(args.output_dir).rglob("*")):
        print(path)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from PIL import Image

    image_path = "runs/cifar10_molab/samples/ema_step_0020000.png"

    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title("EMA samples after 500 steps")
    plt.show()
    return (plt,)


@app.cell
def _(DriftUNet, Path, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    run_dir = Path("runs/cifar10_molab")
    checkpoint = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location=device,
        weights_only=False,
    )

    model_args = checkpoint["args"]
    ema_model = DriftUNet(
        base_channels=model_args["base_channels"],
        blocks_per_stage=model_args["blocks_per_stage"],
    ).to(device)
    ema_model.load_state_dict(checkpoint["ema"])
    ema_model.eval()

    print("Loaded checkpoint at step:", checkpoint["step"])
    return checkpoint, device, ema_model, run_dir


@app.cell
def _(
    DINOv3FeatureEncoder,
    DataLoader,
    F,
    checkpoint,
    datasets,
    device,
    ema_model,
    evaluate_fid,
    torch,
    transforms,
):

    # --- FID: 10k generated EMA samples vs the 10k CIFAR-10 test images ---
    test_data = datasets.CIFAR10(
        root=checkpoint["args"]["data_root"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    fid_10k = evaluate_fid(
        ema_model,
        test_data,
        device=device,
        samples=10_000,
        batch_size=256,
        workers=0,  # Important in Marimo
    )
    print(f"EMA FID (10k generated vs CIFAR-10 test): {fid_10k:.2f}")


    # --- Diversity: DINO feature distances, generated versus real ---
    probe = DINOv3FeatureEncoder(
        model_name=checkpoint["args"]["encoder_model"],
        input_size=checkpoint["args"]["encoder_size"],
        pool_size=checkpoint["args"]["pool_size"],
        extra_statistics=checkpoint["args"]["extra_feature_statistics"],
    ).to(device).eval()

    @torch.no_grad()
    def dino_embedding(images):
        groups = probe(images)
        # One global vector per image, using the same multiscale DINO groups as training.
        vector = torch.cat([group.float().mean(dim=1) for group in groups], dim=1)
        return F.normalize(vector, dim=1)

    @torch.no_grad()
    def generated_embeddings(model, count=1000, batch_size=128):
        embeddings = []
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            fake = model(torch.randn(n, 3, 32, 32, device=device))
            embeddings.append(dino_embedding(fake).cpu())
        return torch.cat(embeddings)

    @torch.no_grad()
    def real_embeddings(dataset, count=1000, batch_size=128):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        embeddings = []
        collected = 0
        for images, _ in loader:
            images = images[:count - collected].to(device).mul(2).sub(1)  # [0,1] → [-1,1]
            embeddings.append(dino_embedding(images).cpu())
            collected += len(images)
            if collected >= count:
                break
        return torch.cat(embeddings)

    def diversity_metrics(embeddings):
        similarities = embeddings @ embeddings.T
        similarities.fill_diagonal_(-float("inf"))
        nearest_neighbor_distance = 1 - similarities.max(dim=1).values

        upper = torch.triu_indices(len(embeddings), len(embeddings), offset=1)
        pairwise_distance = 1 - similarities[upper[0], upper[1]]

        return {
            "mean_pairwise_distance": pairwise_distance.mean().item(),
            "median_nearest_neighbor_distance": nearest_neighbor_distance.median().item(),
        }

    fake_emb = generated_embeddings(ema_model)
    real_emb = real_embeddings(test_data)

    fake_diversity = diversity_metrics(fake_emb)
    real_diversity = diversity_metrics(real_emb)

    print("\nDINO diversity (higher distances = more varied images)")
    print("Generated:", fake_diversity)
    print("Real CIFAR-10:", real_diversity)
    print(
        "Generated / real mean-distance ratio:",
        fake_diversity["mean_pairwise_distance"] / real_diversity["mean_pairwise_distance"],
    )
    return


@app.cell
def _(json, plt, run_dir):

    records = [
        json.loads(line)
        for line in open(run_dir / "metrics.jsonl")
    ]

    trin = [r for r in records if "drift_loss" in r]
    fid = [r for r in records if "ema_fid" in r]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot([r["step"] for r in trin], [r["drift_loss"] for r in trin])
    axes[0].set(title="Drift loss", xlabel="Step", ylabel="Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot([r["step"] for r in trin], [r["grad_norm"] for r in trin])
    axes[1].set(title="Gradient norm", xlabel="Step", ylabel="Norm")
    axes[1].grid(alpha=0.3)

    if fid:
        axes[2].plot(
            [r["step"] for r in fid],
            [r["ema_fid"] for r in fid],
            marker="o",
        )
    axes[2].set(title="EMA FID — lower is better", xlabel="Step", ylabel="FID")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import shutil

    backup_path = shutil.make_archive(
        "/marimo/cifar10_molab_backup",
        "zip",
        root_dir="/marimo/runs/cifar10_molab",
    )
    print("Backup created:", backup_path)
    return


@app.cell
def _(Path):
    import marimo as mo

    archive = Path("/marimo/cifar10_molab_backup.zip")

    print("Exists:", archive.exists())
    if archive.exists():
        print(f"Size: {archive.stat().st_size / 1e9:.2f} GB")

    mo.download(
        data=lambda: archive.read_bytes(),
        filename=archive.name,
        label="Download CIFAR-10 DriftXpress backup",
    )
    return


if __name__ == "__main__":
    app.run()
