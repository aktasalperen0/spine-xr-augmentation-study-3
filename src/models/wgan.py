"""WGAN Generator + Critic for grayscale lesion patches — paper-faithful, multi-resolution.

Paper §4.6.2 + Figure 6 caption: "Wasserstein loss with weight clipping", DCGAN base
(§4.6.1: discriminator = convolutional + batch normalization + Leaky ReLU). The critic
therefore INHERITS BatchNorm (we are reproducing the paper's DCGAN base, not Arjovsky 2017's
"no-BN-in-critic" recommendation — the latter is what made our earlier 256x256 critic explode
to -40M loss).

Generalised over image_size in {64, 128, 256}: number of up/down blocks = log2(image_size/4).
The ROI-patch pivot uses 128x128 (lesion patches are small; 128 is faster and more stable than
the 256 we used for whole images).

Critic init: uniform_(-clip, +clip) so the critic does not start saturated.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_VALID_SIZES = (64, 128, 256)


def _num_blocks(image_size: int) -> int:
    if image_size not in _VALID_SIZES:
        raise ValueError(f"image_size must be one of {_VALID_SIZES}, got {image_size}")
    return int(round(math.log2(image_size // 4)))  # 64->4, 128->5, 256->6


def _g_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    """latent (B, latent_dim) -> image (B, out_channels, image_size, image_size) in [-1, 1]."""

    def __init__(self, latent_dim=120, base_channels=64, out_channels=1, image_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        n = _num_blocks(image_size)
        c = base_channels

        # Channel multipliers from the 4x4 tier down to 1, capped at 16.
        mults = [min(2 ** k, 16) for k in range(n - 1, -1, -1)]  # e.g. n=5 -> [16,8,4,2,1]
        m0 = mults[0]
        self.project = nn.Sequential(
            nn.Linear(latent_dim, m0 * c * 4 * 4),
            nn.BatchNorm1d(m0 * c * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self._proj_c = m0 * c

        up = []
        for i in range(n - 1):  # n-1 hidden up-blocks; final handled by to_img
            up.append(_g_block(mults[i] * c, mults[i + 1] * c))
        self.up = nn.Sequential(*up)
        last_c = mults[-1] * c  # == c
        self.to_img = nn.Sequential(
            nn.ConvTranspose2d(last_c, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.project(z).view(z.size(0), self._proj_c, 4, 4)
        x = self.up(x)
        return self.to_img(x)


class Critic(nn.Module):
    """Image -> scalar Wasserstein score (no final activation)."""

    def __init__(self, in_channels=1, base_channels=64, image_size=128, weight_clip_value=0.01):
        super().__init__()
        self.weight_clip_value = float(weight_clip_value)
        n = _num_blocks(image_size)
        c = base_channels

        def block(in_c, out_c, use_bn):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Ascending channel multipliers, capped at 16. First block has no BN (DCGAN convention).
        mults = [min(2 ** k, 16) for k in range(n)]  # n=5 -> [1,2,4,8,16]
        down = [block(in_channels, mults[0] * c, use_bn=False)]
        for i in range(1, n):
            down.append(block(mults[i - 1] * c, mults[i] * c, use_bn=True))
        self.down = nn.Sequential(*down)
        self.head = nn.Linear(mults[-1] * c * 4 * 4, 1)
        self._init_weights()

    def _init_weights(self):
        clip = self.weight_clip_value
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(m.weight, -clip, clip)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.down(x)
        return self.head(h.flatten(1)).squeeze(1)
