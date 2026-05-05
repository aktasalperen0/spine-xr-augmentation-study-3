"""WGAN Generator + Critic for 256x256 grayscale spine X-rays (paper §4.6.2).

Architecture follows the DCGAN/WGAN convention:
- Generator: latent z (120) -> Linear -> 4x4x(16C) -> 6 ConvTranspose blocks (4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256).
- Critic: mirror — 6 strided Conv blocks (256 -> ... -> 4) -> Linear -> scalar.

Critic deliberately has no BatchNorm (incompatible with both weight clipping and GP).
- For weight_clip path, we use no normalization in the critic (paper-faithful).
- For GP path, we use InstanceNorm (Gulrajani et al. 2017 standard).

The critic's normalization choice is set externally by the trainer (`set_critic_norm`).
This keeps the same architecture file usable for both loss paths without duplicating code.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _g_block(in_c: int, out_c: int, k: int = 4, s: int = 2, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """latent (B, latent_dim) -> image (B, out_channels, image_size, image_size) in [-1, 1]."""

    def __init__(
        self,
        latent_dim: int = 120,
        base_channels: int = 64,
        out_channels: int = 1,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise NotImplementedError("Generator currently fixed to 256x256 output")
        self.latent_dim = latent_dim
        self.image_size = image_size

        c = base_channels
        # Project latent to 4x4 feature map with 16*c channels.
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 16 * c * 4 * 4),
            nn.BatchNorm1d(16 * c * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256. Six up-blocks.
        self.up = nn.Sequential(
            _g_block(16 * c, 8 * c),   # 4 -> 8
            _g_block(8 * c, 4 * c),    # 8 -> 16
            _g_block(4 * c, 2 * c),    # 16 -> 32
            _g_block(2 * c, c),        # 32 -> 64
            _g_block(c, c // 2 if c >= 2 else c),  # 64 -> 128
        )
        last_in = c // 2 if c >= 2 else c
        # Final transposed conv to image_size with Tanh.
        self.to_img = nn.Sequential(
            nn.ConvTranspose2d(last_in, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = self.project(z).view(z.size(0), -1, 4, 4)
        c = self.up(c)
        return self.to_img(c)


class Critic(nn.Module):
    """Image -> scalar score. No final activation (Wasserstein critic outputs are unbounded reals)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        image_size: int = 256,
        norm: str = "none",
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise NotImplementedError("Critic currently fixed to 256x256 input")
        self.norm = norm

        def block(in_c: int, out_c: int, use_norm: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=(norm != "batch")),
            ]
            if use_norm:
                if norm == "instance":
                    layers.append(nn.InstanceNorm2d(out_c, affine=True))
                elif norm == "layer":
                    # LayerNorm over (C, H, W) — needs spatial size; use GroupNorm with G=1 as a proxy.
                    layers.append(nn.GroupNorm(1, out_c))
                # 'none' => no norm (paper-faithful for weight_clip)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        c = base_channels
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4. Six down-blocks.
        # First block intentionally has no norm (standard).
        self.down = nn.Sequential(
            block(in_channels, c, use_norm=False),
            block(c, 2 * c, use_norm=(norm != "none")),
            block(2 * c, 4 * c, use_norm=(norm != "none")),
            block(4 * c, 8 * c, use_norm=(norm != "none")),
            block(8 * c, 16 * c, use_norm=(norm != "none")),
            block(16 * c, 16 * c, use_norm=(norm != "none")),
        )
        self.head = nn.Linear(16 * c * 4 * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None and m.weight is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        return self.head(h.flatten(1)).squeeze(1)


def build_critic_for_loss(loss_kind: str, **kwargs) -> Critic:
    """Helper: pick the right normalization for the chosen loss path."""
    if loss_kind == "weight_clip":
        return Critic(norm="none", **kwargs)
    if loss_kind == "gp":
        return Critic(norm="instance", **kwargs)
    raise ValueError(f"unknown loss_kind {loss_kind!r}")
