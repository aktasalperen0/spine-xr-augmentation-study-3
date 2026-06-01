"""MONAI pixel-space DDPM trainer for ROI lesion patches.

Uses MONAI's DiffusionModelUNet + DDPMScheduler. Import path differs by MONAI version:
  - newer monai: monai.networks.nets.DiffusionModelUNet, monai.networks.schedulers.DDPMScheduler
  - monai-generative: generative.networks.nets / generative.networks.schedulers
`_import_monai_generative()` tries both.

Training objective: predict the noise epsilon added at a random timestep (standard DDPM MSE).
Sampling: iterative denoise from N(0,1) through the scheduler.
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


def import_monai_generative():
    """Return (DiffusionModelUNet, DDPMScheduler) across MONAI variants."""
    try:
        from monai.networks.nets import DiffusionModelUNet
        from monai.networks.schedulers import DDPMScheduler
        return DiffusionModelUNet, DDPMScheduler
    except Exception:
        from generative.networks.nets import DiffusionModelUNet
        from generative.networks.schedulers import DDPMScheduler
        return DiffusionModelUNet, DDPMScheduler


@dataclass
class DiffusionConfig:
    class_name: str
    class_slug: str
    case: str
    pool_csv: str
    out_dir: Path
    image_size: int = 128
    in_channels: int = 1
    channels: tuple = (64, 128, 128, 256)
    attention_levels: tuple = (False, False, True, True)
    num_res_blocks: int = 2
    num_head_channels: int = 64
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    batch_size: int = 32
    lr: float = 2.5e-4
    total_iterations: int = 8000
    log_every: int = 100
    sample_every: int = 1000
    ckpt_every: int = 1000
    first_ckpt_at: int = 2000
    num_sample_images: int = 16
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    extra: dict = field(default_factory=dict)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_unet_scheduler(cfg: DiffusionConfig, device):
    DiffusionModelUNet, DDPMScheduler = import_monai_generative()
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=cfg.in_channels,
        out_channels=cfg.in_channels,
        channels=tuple(cfg.channels),
        attention_levels=tuple(cfg.attention_levels),
        num_res_blocks=cfg.num_res_blocks,
        num_head_channels=cfg.num_head_channels,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        schedule="linear_beta",
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    )
    return unet, scheduler


@torch.no_grad()
def sample_images(unet, scheduler, n, image_size, in_channels, device):
    unet.eval()
    img = torch.randn(n, in_channels, image_size, image_size, device=device)
    scheduler.set_timesteps(scheduler.num_train_timesteps)
    for t in scheduler.timesteps:
        model_out = unet(img, timesteps=torch.full((n,), int(t), device=device, dtype=torch.long))
        img, _ = scheduler.step(model_out, int(t), img)
    unet.train()
    return img.clamp(-1, 1)


def _save_grid(imgs, out_path: Path):
    x = (imgs + 1.0) / 2.0
    grid = vutils.make_grid(x.cpu(), nrow=int(math.sqrt(x.size(0))), padding=2, value_range=(0.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(out_path))


def train_diffusion(cfg: DiffusionConfig, dataset, log) -> dict:
    device = _device()
    log.info(f"[ddpm/{cfg.class_slug}] device={device.type} N_pool={len(dataset)} size={cfg.image_size}")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    sampler = RandomSampler(dataset, replacement=True, num_samples=10**9)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    unet, scheduler = build_unet_scheduler(cfg, device)
    opt = torch.optim.Adam(unet.parameters(), lr=cfg.lr)
    amp_on = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    out = Path(cfg.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    fh = open(out / "log.csv", "w", newline=""); wr = csv.writer(fh); wr.writerow(["iter", "loss"])

    data_iter = iter(loader)
    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader); return next(data_iter)

    unet.train()
    snapshots = []
    for it in range(1, cfg.total_iterations + 1):
        x = next_batch().to(device, non_blocking=True)
        noise = torch.randn_like(x)
        t = torch.randint(0, cfg.num_train_timesteps, (x.size(0),), device=device).long()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            noisy = scheduler.add_noise(original_samples=x, noise=noise, timesteps=t)
            pred = unet(noisy, timesteps=t)
            loss = F.mse_loss(pred.float(), noise.float())
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        wr.writerow([it, f"{float(loss.item()):.6f}"])

        if it % cfg.log_every == 0:
            log.info(f"[ddpm/{cfg.class_slug}] iter {it}/{cfg.total_iterations} loss={loss.item():.4f}")
        if it % cfg.sample_every == 0 or it == cfg.total_iterations:
            imgs = sample_images(unet, scheduler, cfg.num_sample_images, cfg.image_size, cfg.in_channels, device)
            _save_grid(imgs, out / "samples" / f"iter_{it:06d}.png")
        if it >= cfg.first_ckpt_at and (it % cfg.ckpt_every == 0 or it == cfg.total_iterations):
            ck = out / "checkpoints" / f"iter_{it:06d}.pt"
            torch.save({"iter": it, "unet_state": unet.state_dict(),
                        "config": {k: getattr(cfg, k) for k in
                                   ["image_size", "in_channels", "channels", "attention_levels",
                                    "num_res_blocks", "num_head_channels", "num_train_timesteps",
                                    "beta_start", "beta_end", "class_name", "class_slug"]}}, ck)
            snapshots.append(it)
            log.info(f"[ddpm/{cfg.class_slug}] saved {ck.name}")
    fh.close()
    return {"class_name": cfg.class_name, "class_slug": cfg.class_slug, "case": cfg.case,
            "n_pool": len(dataset), "total_iterations": cfg.total_iterations, "snapshots": snapshots,
            "out_dir": str(out)}
