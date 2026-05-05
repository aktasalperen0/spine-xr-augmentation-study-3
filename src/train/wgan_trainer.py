"""WGAN training loop — weight-clipping (paper-faithful) by default; WGAN-GP optional.

One iteration = n_critic critic updates + 1 generator update.

Paper-faithful path (loss == 'weight_clip'):
- Critic loss   : -E[D(real)] + E[D(G(z))]   (we minimise this)
- Generator loss: -E[D(G(z))]
- After each critic step, clamp weights to [-c, +c].
- Optimizer    : RMSProp (paper) on both.

WGAN-GP escape hatch (loss == 'gp'):
- Add lambda * E[(||grad D(x_hat)||_2 - 1)^2] to the critic loss.
- x_hat = eps*real + (1-eps)*fake, eps ~ U(0, 1) per sample.
- No weight clipping; critic uses InstanceNorm.
- Optimizer    : Adam (betas (0.0, 0.9)).
- AMP can be enabled (mixed precision) without weight-clip pathologies.

Snapshots: at every `snapshot_every` generator iterations after `first_snapshot_at`, write
G/D state_dicts and a 4x4 sample grid PNG. Phase 06 (next milestone) will compute FID per
snapshot to pick the deployable checkpoint.
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

from src.models.wgan import Generator, build_critic_for_loss


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
    loss: str = "weight_clip"          # 'weight_clip' | 'gp'
    weight_clip_value: float = 0.01
    gp_lambda: float = 10.0

    optim_clip: dict = field(default_factory=lambda: {"name": "rmsprop", "lr": 5e-5})
    optim_gp: dict = field(default_factory=lambda: {"name": "adam", "lr": 1e-4, "betas": (0.0, 0.9)})

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

    amp: bool = False
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
    if name == "adam":
        betas = spec.get("betas", (0.0, 0.9))
        return torch.optim.Adam(params, lr=float(spec["lr"]), betas=tuple(betas))
    raise ValueError(f"unknown optimizer {name!r}")


def _gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device) -> torch.Tensor:
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=device)
    x_hat = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    score = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=score.sum(), inputs=x_hat,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


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


def train_wgan(cfg: WGANConfig, dataset, log) -> dict:
    device = _select_device()
    log.info(f"[wgan/{cfg.class_slug}] device={device.type}  loss={cfg.loss}  N_pool={len(dataset)}")

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
    D = build_critic_for_loss(
        cfg.loss,
        in_channels=cfg.out_channels,
        base_channels=cfg.d_base_channels,
        image_size=cfg.image_size,
    ).to(device)

    if cfg.loss == "weight_clip":
        opt_g = _make_optim(G.parameters(), cfg.optim_clip)
        opt_d = _make_optim(D.parameters(), cfg.optim_clip)
        amp_enabled = False  # AMP + weight clipping is brittle.
    elif cfg.loss == "gp":
        opt_g = _make_optim(G.parameters(), cfg.optim_gp)
        opt_d = _make_optim(D.parameters(), cfg.optim_gp)
        amp_enabled = bool(cfg.amp) and device.type == "cuda"
    else:
        raise ValueError(cfg.loss)

    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    out = Path(cfg.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log_csv = out / "log.csv"
    log_fh = open(log_csv, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["g_iter", "d_loss", "g_loss", "d_real_mean", "d_fake_mean", "gp"])

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
        gp_acc = 0.0
        for _ in range(cfg.n_critic):
            real = next_real()
            bs = real.size(0)
            z = torch.randn(bs, cfg.latent_dim, device=device)

            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                with torch.no_grad():
                    fake = G(z)
                d_real = D(real)
                d_fake = D(fake)
                d_loss = -d_real.mean() + d_fake.mean()
                if cfg.loss == "gp":
                    gp = _gradient_penalty(D, real, fake.detach(), device)
                    d_loss = d_loss + cfg.gp_lambda * gp
                else:
                    gp = torch.tensor(0.0, device=device)

            if amp_enabled:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss.backward()
                opt_d.step()

            if cfg.loss == "weight_clip":
                with torch.no_grad():
                    for p in D.parameters():
                        p.clamp_(-cfg.weight_clip_value, cfg.weight_clip_value)

            d_loss_acc += float(d_loss.item())
            d_real_acc += float(d_real.mean().item())
            d_fake_acc += float(d_fake.mean().item())
            gp_acc += float(gp.item())

        d_loss_avg = d_loss_acc / cfg.n_critic
        d_real_avg = d_real_acc / cfg.n_critic
        d_fake_avg = d_fake_acc / cfg.n_critic
        gp_avg = gp_acc / cfg.n_critic

        # ----- Generator update -----
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            fake = G(z)
            g_loss = -D(fake).mean()
        if amp_enabled:
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            g_loss.backward()
            opt_g.step()

        writer.writerow([g_iter,
                         f"{d_loss_avg:.6f}",
                         f"{float(g_loss.item()):.6f}",
                         f"{d_real_avg:.6f}",
                         f"{d_fake_avg:.6f}",
                         f"{gp_avg:.6f}"])

        if g_iter % cfg.log_every == 0:
            log.info(
                f"[wgan/{cfg.class_slug}] iter {g_iter}/{cfg.total_iterations} "
                f"d_loss={d_loss_avg:.4f} g_loss={float(g_loss.item()):.4f} "
                f"d_real={d_real_avg:.4f} d_fake={d_fake_avg:.4f}"
                + (f" gp={gp_avg:.4f}" if cfg.loss == "gp" else "")
            )

        if g_iter % cfg.sample_every == 0 or g_iter == cfg.total_iterations:
            _save_sample_grid(G, fixed_z, out / "samples" / f"iter_{g_iter:06d}.png")

        if g_iter >= cfg.first_snapshot_at and (g_iter % cfg.snapshot_every == 0 or g_iter == cfg.total_iterations):
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
                    "loss": cfg.loss,
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
        "loss": cfg.loss,
        "total_iterations": cfg.total_iterations,
        "snapshots": snapshots,
        "out_dir": str(out),
    }
