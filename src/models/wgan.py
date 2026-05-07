"""WGAN Generator + Critic for 256x256 grayscale spine X-rays — paper-faithful rewrite.

Paper §4.6.2: "WGAN was introduced... with the basic architecture of a DCGAN and utilizing
the Wasserstein loss... weight clipping" (Figure 6 caption: "Wasserstein loss with weight
clipping").

Paper §4.6.1 (DCGAN base): "the downsampling block consists of convolutional, batch
normalization, and Leaky ReLU activation function." So the critic INHERITS BatchNorm from
the DCGAN base — we are reproducing that, not the original Arjovsky 2017 "no BN in critic"
recommendation.

Why this matters (from our −40 M loss diagnosis):
  Without BN, a 6-layer critic at 256² with weights clipped to ±0.01 amplifies activations
  layer-to-layer (≈0.16 → 1.6 → 33 → 1.3 K → 110 K → 18 M → 2.9 B at the head). With BN,
  each layer renormalises to mean 0 / variance 1, so outputs and losses stay O(1).

Critic init: uniform_(-clip, +clip). Avoids the "init at std=0.02 → immediately clipped to
±0.01 → critic starts saturated" failure mode the deviated weight_clip path hit in the smoke.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------- Generator (unchanged DCGAN-style) ---------

def _g_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
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

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 — six up-blocks.
        last_in = c // 2 if c >= 2 else c
        self.up = nn.Sequential(
            _g_block(16 * c, 8 * c),   # 4 -> 8
            _g_block(8 * c, 4 * c),    # 8 -> 16
            _g_block(4 * c, 2 * c),    # 16 -> 32
            _g_block(2 * c, c),        # 32 -> 64
            _g_block(c, last_in),      # 64 -> 128
        )
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
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = self.project(z).view(z.size(0), -1, 4, 4)
        c = self.up(c)
        return self.to_img(c)


# --------- Critic (paper-faithful: DCGAN downsampling = Conv + BN + LeakyReLU) ---------

class Critic(nn.Module):
    """Image -> scalar Wasserstein score. No final activation (real-valued by design).

    Per paper §4.6.1, the discriminator/critic uses BatchNorm in middle blocks. The first
    block conventionally has no BN (DCGAN convention; BN of a single-channel input is a no-op
    anyway). The classification head is a Linear taking the flattened 4×4 feature map.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        image_size: int = 256,
        weight_clip_value: float = 0.01,
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise NotImplementedError("Critic currently fixed to 256x256 input")
        self.weight_clip_value = float(weight_clip_value)

        def block(in_c: int, out_c: int, use_bn: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_bn),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        c = base_channels
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4. Six down-blocks.
        # Block 0: no BN (DCGAN convention). Blocks 1-5: BN (paper §4.6.1).
        self.down = nn.Sequential(
            block(in_channels, c,    use_bn=False),
            block(c,            2 * c, use_bn=True),
            block(2 * c,        4 * c, use_bn=True),
            block(4 * c,        8 * c, use_bn=True),
            block(8 * c,       16 * c, use_bn=True),
            block(16 * c,      16 * c, use_bn=True),
        )
        self.head = nn.Linear(16 * c * 4 * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Uniform init within the clip range so the critic does not start saturated.

        Background: with normal_(0, 0.02) and clip=0.01, ~62% of weights are |w|>0.01 and
        get clipped immediately, putting the critic at saturation from iter 0. Initialising
        within the clip range keeps the network operating in its dynamic range from the
        start.
        """
        clip = self.weight_clip_value
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.uniform_(m.weight, -clip, clip)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # BN affine: keep gamma=1, beta=0 (standard). BN params are NOT clipped.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        return self.head(h.flatten(1)).squeeze(1)
