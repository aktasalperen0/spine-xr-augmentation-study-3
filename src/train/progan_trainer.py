"""Progressive-growing GAN trainer (WGAN-GP loss) for ROI patches.

Grows 4->8->...->128. Each new stage fades in over `fade_iters`, then stabilises for
`stab_iters`. Real images are downsampled on the fly to the current stage resolution.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, RandomSampler

from src.models.progan import RESOLUTIONS, Discriminator, Generator


@dataclass
class ProGANConfig:
    class_name: str
    class_slug: str
    case: str
    pool_csv: str
    out_dir: Path
    latent_dim: int = 128
    out_channels: int = 1
    max_depth: int = 5                       # 128x128
    fade_iters: int = 4000
    stab_iters: int = 4000
    gp_lambda: float = 10.0
    drift_eps: float = 0.001
    lr: float = 0.001
    betas: tuple = (0.0, 0.99)
    batch_per_res: dict = field(default_factory=lambda: {4: 128, 8: 128, 16: 64, 32: 32, 64: 16, 128: 16})
    log_every: int = 100
    sample_every: int = 1000
    num_sample_images: int = 16
    num_workers: int = 4
    seed: int = 42


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def _gp(D, real, fake, depth, alpha, device):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=device)
    xh = (eps * real + (1 - eps) * fake).requires_grad_(True)
    s = D(xh, depth, alpha)
    g = torch.autograd.grad(s.sum(), xh, create_graph=True)[0].view(bs, -1)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def _grid(x, path: Path):
    x = (x.clamp(-1, 1) + 1) / 2
    g = vutils.make_grid(x.cpu(), nrow=int(math.sqrt(x.size(0))), padding=2, value_range=(0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(g, str(path))


def train_progan(cfg: ProGANConfig, dataset, log) -> dict:
    device = _device()
    log.info(f"[progan/{cfg.class_slug}] device={device.type} N_pool={len(dataset)}")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    G = Generator(cfg.latent_dim, cfg.out_channels, cfg.max_depth).to(device)
    D = Discriminator(cfg.out_channels, cfg.max_depth).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    out = Path(cfg.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    fh = open(out / "log.csv", "w", newline=""); wr = csv.writer(fh)
    wr.writerow(["global_iter", "depth", "alpha", "d_loss", "g_loss"])
    fixed_z = torch.randn(cfg.num_sample_images, cfg.latent_dim, device=device)

    def make_loader(res):
        bs = cfg.batch_per_res.get(res, 16)
        sampler = RandomSampler(dataset, replacement=True, num_samples=10**9)
        return DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True)

    gi = 0
    snapshots = []
    for depth in range(cfg.max_depth + 1):
        res = RESOLUTIONS[depth]
        loader = make_loader(res); it = iter(loader)
        phases = [("stab", cfg.stab_iters)] if depth == 0 else [("fade", cfg.fade_iters), ("stab", cfg.stab_iters)]
        for phase, n_iters in phases:
            for i in range(n_iters):
                gi += 1
                alpha = (i / max(1, n_iters)) if phase == "fade" else 1.0
                try:
                    real = next(it)
                except StopIteration:
                    it = iter(loader); real = next(it)
                real = real.to(device, non_blocking=True)
                if res != real.size(-1):
                    real = F.interpolate(real, size=res, mode="area")

                # --- D step ---
                optD.zero_grad(set_to_none=True)
                z = torch.randn(real.size(0), cfg.latent_dim, device=device)
                fake = G(z, depth, alpha).detach()
                d_real = D(real, depth, alpha)
                d_fake = D(fake, depth, alpha)
                gp = _gp(D, real, fake, depth, alpha, device)
                d_loss = d_fake.mean() - d_real.mean() + cfg.gp_lambda * gp + cfg.drift_eps * (d_real ** 2).mean()
                d_loss.backward(); optD.step()

                # --- G step ---
                optG.zero_grad(set_to_none=True)
                z = torch.randn(real.size(0), cfg.latent_dim, device=device)
                g_loss = -D(G(z, depth, alpha), depth, alpha).mean()
                g_loss.backward(); optG.step()

                wr.writerow([gi, depth, f"{alpha:.3f}", f"{float(d_loss.item()):.4f}", f"{float(g_loss.item()):.4f}"])
                if gi % cfg.log_every == 0:
                    log.info(f"[progan/{cfg.class_slug}] gi={gi} res={res} {phase} a={alpha:.2f} "
                             f"d={d_loss.item():.3f} g={g_loss.item():.3f}")
                if gi % cfg.sample_every == 0:
                    with torch.no_grad():
                        _grid(G(fixed_z, depth, alpha), out / "samples" / f"res{res:03d}_gi{gi:07d}.png")
        # checkpoint at end of each stabilised stage
        ck = out / "checkpoints" / f"depth{depth}_res{res:03d}.pt"
        torch.save({"depth": depth, "G_state": G.state_dict(), "D_state": D.state_dict(),
                    "config": {"latent_dim": cfg.latent_dim, "out_channels": cfg.out_channels,
                               "max_depth": cfg.max_depth, "class_name": cfg.class_name,
                               "class_slug": cfg.class_slug}}, ck)
        snapshots.append(res)
        log.info(f"[progan/{cfg.class_slug}] saved {ck.name}")
    fh.close()
    return {"class_name": cfg.class_name, "class_slug": cfg.class_slug, "case": cfg.case,
            "n_pool": len(dataset), "snapshots": snapshots, "out_dir": str(out)}
