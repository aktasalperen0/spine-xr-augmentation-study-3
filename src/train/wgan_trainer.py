"""WGAN training loop — paper-faithful: Wasserstein loss + weight clipping (paper §4.6.2).

One iteration = n_critic critic updates + 1 generator update.

Paper-faithful path (the only path):
- Critic loss   : -E[D(real)] + E[D(G(z))]   (we minimise this)
- Generator loss: -E[D(G(z))]
- After each critic step, clamp critic conv/linear weights to [-c, +c]. BatchNorm params
  are NOT clipped (they're scale/shift, not part of the Lipschitz-bounded function).
- Optimizer    : RMSProp (paper / Arjovsky 2017 standard).
- No AMP (mixed precision interacts badly with weight clipping).

Snapshots: at every `snapshot_every` generator iterations after `first_snapshot_at`, write
G/D state_dicts and a 4×4 sample grid PNG. Phase 06 (FID) ranks snapshots; Phase 07 generates
from the chosen one.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader, RandomSampler

from src.models.wgan import Critic, Generator


@dataclass
class WGANConfig:
    class_name: str
    class_slug: str
    case: str
    pool_csv: str
    out_dir: Path

    latent_dim: int = 120
    image_size: int = 256
    g_base_channels: int = 64
    d_base_channels: int = 64
    out_channels: int = 1

    batch_size: int = 64
    n_critic: int = 5
    weight_clip_value: float = 0.01

    optim: dict = field(default_factory=lambda: {"name": "rmsprop", "lr": 5e-5})

    total_iterations: int = 20000
    log_every: int = 100
    snapshot_every: int = 1500
    first_snapshot_at: int = 5000
    sample_every: int = 1500
    num_sample_images: int = 16

    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    seed: int = 42


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_optim(params, spec: dict):
    name = spec["name"].lower()
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=float(spec["lr"]))
    raise ValueError(
        f"WGAN-clip is paper-faithful only with RMSProp; got {name!r}. "
        "Adam is intentionally not supported in this code path."
    )


def _save_sample_grid(G: Generator, fixed_z: torch.Tensor, out_path: Path) -> None:
    G.eval()
    with torch.no_grad():
        x = G(fixed_z).clamp(-1, 1)
    G.train()
    # Map [-1, 1] -> [0, 1] for the PNG grid.
    x = (x + 1.0) / 2.0
    grid = vutils.make_grid(x.cpu(), nrow=int(math.sqrt(x.size(0))), padding=2, value_range=(0.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(out_path))


def _clip_critic(D: nn.Module, clip_value: float) -> None:
    """Clip ONLY conv & linear weights — not BatchNorm scale/shift."""
    with torch.no_grad():
        for m in D.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.clamp_(-clip_value, clip_value)
                if m.bias is not None:
                    m.bias.clamp_(-clip_value, clip_value)


def train_wgan(cfg: WGANConfig, dataset, log) -> dict:
    device = _select_device()
    log.info(
        f"[wgan/{cfg.class_slug}] device={device.type}  loss=weight_clip  "
        f"n_pool={len(dataset)}  clip=±{cfg.weight_clip_value}  n_critic={cfg.n_critic}"
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Replacement sampler so small classes (16-300 imgs) still produce many batches.
    sampler = RandomSampler(dataset, replacement=True, num_samples=10**9)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    G = Generator(
        latent_dim=cfg.latent_dim, base_channels=cfg.g_base_channels,
        out_channels=cfg.out_channels, image_size=cfg.image_size,
    ).to(device)
    D = Critic(
        in_channels=cfg.out_channels,
        base_channels=cfg.d_base_channels,
        image_size=cfg.image_size,
        weight_clip_value=cfg.weight_clip_value,
    ).to(device)

    opt_g = _make_optim(G.parameters(), cfg.optim)
    opt_d = _make_optim(D.parameters(), cfg.optim)

    out = Path(cfg.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log_csv = out / "log.csv"
    log_fh = open(log_csv, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["g_iter", "d_loss", "g_loss", "d_real_mean", "d_fake_mean"])

    fixed_z = torch.randn(cfg.num_sample_images, cfg.latent_dim, device=device)

    data_iter = iter(loader)

    def next_real() -> torch.Tensor:
        nonlocal data_iter
        try:
            return next(data_iter).to(device, non_blocking=True)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter).to(device, non_blocking=True)

    G.train()
    D.train()
    snapshots: list[int] = []

    for g_iter in range(1, cfg.total_iterations + 1):
        # ----- Critic updates -----
        d_loss_acc = 0.0
        d_real_acc = 0.0
        d_fake_acc = 0.0
        for _ in range(cfg.n_critic):
            real = next_real()
            bs = real.size(0)
            z = torch.randn(bs, cfg.latent_dim, device=device)

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = G(z)
            d_real = D(real)
            d_fake = D(fake)
            d_loss = -d_real.mean() + d_fake.mean()
            d_loss.backward()
            opt_d.step()

            _clip_critic(D, cfg.weight_clip_value)

            d_loss_acc += float(d_loss.item())
            d_real_acc += float(d_real.mean().item())
            d_fake_acc += float(d_fake.mean().item())

        d_loss_avg = d_loss_acc / cfg.n_critic
        d_real_avg = d_real_acc / cfg.n_critic
        d_fake_avg = d_fake_acc / cfg.n_critic

        # ----- Generator update -----
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
        opt_g.zero_grad(set_to_none=True)
        fake = G(z)
        g_loss = -D(fake).mean()
        g_loss.backward()
        opt_g.step()

        writer.writerow([g_iter,
                         f"{d_loss_avg:.6f}",
                         f"{float(g_loss.item()):.6f}",
                         f"{d_real_avg:.6f}",
                         f"{d_fake_avg:.6f}"])

        if g_iter % cfg.log_every == 0:
            log.info(
                f"[wgan/{cfg.class_slug}] iter {g_iter}/{cfg.total_iterations} "
                f"d_loss={d_loss_avg:.4f} g_loss={float(g_loss.item()):.4f} "
                f"d_real={d_real_avg:.4f} d_fake={d_fake_avg:.4f}"
            )

        if g_iter % cfg.sample_every == 0 or g_iter == cfg.total_iterations:
            _save_sample_grid(G, fixed_z, out / "samples" / f"iter_{g_iter:06d}.png")

        if g_iter >= cfg.first_snapshot_at and (
            g_iter % cfg.snapshot_every == 0 or g_iter == cfg.total_iterations
        ):
            ckpt_path = out / "checkpoints" / f"iter_{g_iter:06d}.pt"
            torch.save({
                "g_iter": g_iter,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "config": {
                    "latent_dim": cfg.latent_dim,
                    "image_size": cfg.image_size,
                    "out_channels": cfg.out_channels,
                    "g_base_channels": cfg.g_base_channels,
                    "d_base_channels": cfg.d_base_channels,
                    "loss": "weight_clip",
                    "weight_clip_value": cfg.weight_clip_value,
                    "class_name": cfg.class_name,
                    "class_slug": cfg.class_slug,
                },
            }, ckpt_path)
            snapshots.append(g_iter)
            log.info(f"[wgan/{cfg.class_slug}] saved checkpoint {ckpt_path.name}")

    log_fh.close()
    return {
        "class_name": cfg.class_name,
        "class_slug": cfg.class_slug,
        "case": cfg.case,
        "n_pool": int(len(dataset)),
        "loss": "weight_clip",
        "total_iterations": cfg.total_iterations,
        "snapshots": snapshots,
        "out_dir": str(out),
    }
