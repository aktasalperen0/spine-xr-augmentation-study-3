"""Progressive Growing GAN (Karras et al. 2018) for 128x128 grayscale ROI patches.

Core ProGAN ingredients (the ones that make it stable where plain GANs failed):
- Equalized learning-rate conv/linear (runtime weight scaling).
- PixelNorm in the generator.
- Minibatch-stddev layer in the discriminator (fights mode collapse).
- Progressive growing 4->8->16->32->64->128 with smooth fade-in of each new block.

Loss is WGAN-GP (standard for ProGAN), handled in the trainer.

Resolutions: [4,8,16,32,64,128] => 6 stages (depth 0..5).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channels per stage (depth 0=4x4 ... 5=128x128). Capped for memory.
CHANNELS = [256, 256, 256, 128, 64, 32]
RESOLUTIONS = [4, 8, 16, 32, 64, 128]


class EqLR:
    """Mixin helper: scale weights at runtime by He constant (equalized learning rate)."""


class EqConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.scale = math.sqrt(2.0 / (in_c * k * k))
        self.s, self.p = s, p

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, stride=self.s, padding=self.p)


class EqLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.scale = math.sqrt(2.0 / in_f)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


def _lrelu():
    return nn.LeakyReLU(0.2)


class GBlock(nn.Module):
    """Upsample (except first) + 2 conv + pixelnorm."""
    def __init__(self, in_c, out_c, first=False):
        super().__init__()
        self.first = first
        if first:
            self.c1 = EqConv2d(in_c, out_c, 4, 1, 3)   # 1x1 -> 4x4 via padding
        else:
            self.c1 = EqConv2d(in_c, out_c, 3, 1, 1)
        self.c2 = EqConv2d(out_c, out_c, 3, 1, 1)
        self.pn = PixelNorm()
        self.act = _lrelu()

    def forward(self, x):
        if not self.first:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.pn(self.act(self.c1(x)))
        x = self.pn(self.act(self.c2(x)))
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1, max_depth=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_depth = max_depth
        self.pn0 = PixelNorm()
        self.blocks = nn.ModuleList()
        self.to_gray = nn.ModuleList()
        in_c = latent_dim
        for d in range(max_depth + 1):
            out_c = CHANNELS[d]
            self.blocks.append(GBlock(in_c, out_c, first=(d == 0)))
            self.to_gray.append(EqConv2d(out_c, out_channels, 1, 1, 0))
            in_c = out_c

    def forward(self, z, depth, alpha):
        x = self.pn0(z).view(z.size(0), self.latent_dim, 1, 1)
        x = self.blocks[0](x)
        for d in range(1, depth + 1):
            prev = x
            x = self.blocks[d](x)
        if depth == 0:
            return torch.tanh(self.to_gray[0](x))
        out = self.to_gray[depth](x)
        if alpha < 1.0:
            up_prev = F.interpolate(prev, scale_factor=2, mode="nearest")
            out = alpha * out + (1 - alpha) * self.to_gray[depth - 1](up_prev)
        return torch.tanh(out)


class MinibatchStd(nn.Module):
    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8).mean()
        return torch.cat([x, std.expand(x.size(0), 1, x.size(2), x.size(3))], dim=1)


class DBlock(nn.Module):
    """2 conv + downsample (except last)."""
    def __init__(self, in_c, out_c, last=False):
        super().__init__()
        self.last = last
        if last:
            self.mbstd = MinibatchStd()
            self.c1 = EqConv2d(in_c + 1, out_c, 3, 1, 1)
            self.c2 = EqConv2d(out_c, out_c, 4, 1, 0)   # 4x4 -> 1x1
            self.fc = EqLinear(out_c, 1)
        else:
            self.c1 = EqConv2d(in_c, in_c, 3, 1, 1)
            self.c2 = EqConv2d(in_c, out_c, 3, 1, 1)
        self.act = _lrelu()

    def forward(self, x):
        if self.last:
            x = self.mbstd(x)
            x = self.act(self.c1(x))
            x = self.act(self.c2(x))
            return self.fc(x.flatten(1)).squeeze(1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return F.avg_pool2d(x, 2)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, max_depth=5):
        super().__init__()
        self.max_depth = max_depth
        self.from_gray = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for d in range(max_depth + 1):
            c = CHANNELS[d]
            self.from_gray.append(EqConv2d(in_channels, c, 1, 1, 0))
            nxt = CHANNELS[d - 1] if d > 0 else CHANNELS[0]
            self.blocks.append(DBlock(c, nxt, last=(d == 0)))
        self.act = _lrelu()

    def forward(self, img, depth, alpha):
        x = self.act(self.from_gray[depth](img))
        x = self.blocks[depth](x)
        if depth > 0 and alpha < 1.0:
            down = F.avg_pool2d(img, 2)
            skip = self.act(self.from_gray[depth - 1](down))
            x = alpha * x + (1 - alpha) * skip
        for d in range(depth - 1, -1, -1):
            x = self.blocks[d](x)
        return x
