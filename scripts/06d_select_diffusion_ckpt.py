"""Phase 06d: pick the best DDPM checkpoint per class by FID (Q1).

For each class and each checkpoint under outputs/05d_diffusion/<case>/<slug>/checkpoints/,
sample `--fid-samples` patches, compute FID against that class's real ROI patches, and select
the argmin-FID checkpoint. Writes outputs/06d_ckpt_select/<slug>.json (best ckpt + FID table).

This operationalises the user's observation that the best samples are NOT at the final iter
(8k over-whitens) — we choose the checkpoint with the lowest FID instead of a static iter.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from src.eval.fid import compute_fid, real_paths_for_class
from src.train.diffusion_trainer import import_monai_generative, sample_images
from src.utils.config import load_config
from src.utils.logging import get_logger


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def _build_from_ckpt(ckpt, device):
    DiffusionModelUNet, DDPMScheduler = import_monai_generative()
    c = ckpt["config"]
    unet = DiffusionModelUNet(
        spatial_dims=2, in_channels=int(c["in_channels"]), out_channels=int(c["in_channels"]),
        channels=tuple(c["channels"]), attention_levels=tuple(c["attention_levels"]),
        num_res_blocks=int(c["num_res_blocks"]), num_head_channels=int(c["num_head_channels"]),
    ).to(device)
    unet.load_state_dict(ckpt["unet_state"]); unet.eval()
    sched = DDPMScheduler(num_train_timesteps=int(c["num_train_timesteps"]), schedule="linear_beta",
                          beta_start=float(c["beta_start"]), beta_end=float(c["beta_end"]))
    return unet, sched, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--diffusion", default="configs/diffusion.yaml")
    ap.add_argument("--classes-filter", nargs="*", default=None)
    ap.add_argument("--diffusion-tag", default="05d_diffusion")
    ap.add_argument("--fid-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-tag", default="06d_ckpt_select")
    args = ap.parse_args()

    base = load_config(args.config)
    dcfg = load_config(args.diffusion)
    log = get_logger()
    device = _device()
    droot = Path(base["paths"]["outputs_root"]) / args.diffusion_tag
    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag
    out_root.mkdir(parents=True, exist_ok=True)

    for entry in dcfg["classes"]:
        if args.classes_filter and entry["name"] not in args.classes_filter:
            continue
        slug, case, name = entry["slug"], entry["case"], entry["name"]
        ck_dir = droot / case / slug / "checkpoints"
        ckpts = sorted(ck_dir.glob("*.pt"))
        if not ckpts:
            log.warning(f"[{slug}] no checkpoints in {ck_dir}"); continue
        real = real_paths_for_class(f"{base['paths']['outputs_root']}/02b_roi/{case}/train.csv", name)
        log.info(f"[{slug}] {len(ckpts)} checkpoints, {len(real)} real patches")

        table = []
        for cp in ckpts:
            ckpt = torch.load(cp, map_location=device, weights_only=False)
            unet, sched, c = _build_from_ckpt(ckpt, device)
            with tempfile.TemporaryDirectory() as td:
                written = 0
                while written < args.fid_samples:
                    bs = min(args.batch_size, args.fid_samples - written)
                    imgs = sample_images(unet, sched, bs, int(c["image_size"]), int(c["in_channels"]), device)
                    x = imgs[:, 0] if imgs.size(1) == 1 else imgs.mean(1)
                    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    for i in range(bs):
                        cv2.imwrite(str(Path(td) / f"{written+i:06d}.png"), x[i])
                    written += bs
                fake = [str(p) for p in Path(td).glob("*.png")]
                try:
                    fid = compute_fid(real, fake, device=("cuda" if device.type == "cuda" else "cpu"))
                except Exception as e:
                    fid = float("nan"); log.warning(f"[{slug}] FID err at {cp.name}: {e}")
            table.append({"checkpoint": str(cp), "iter": int(ckpt.get("iter", 0)), "fid": fid})
            log.info(f"[{slug}] {cp.name} FID={fid:.2f}")

        valid = [t for t in table if t["fid"] == t["fid"]]
        best = min(valid, key=lambda t: t["fid"]) if valid else table[-1]
        (out_root / f"{slug}.json").write_text(json.dumps(
            {"class_name": name, "class_slug": slug, "case": case,
             "best_checkpoint": best["checkpoint"], "best_iter": best["iter"], "best_fid": best["fid"],
             "table": table}, indent=2))
        log.info(f"[{slug}] BEST = {Path(best['checkpoint']).name} (FID={best['fid']:.2f})")


if __name__ == "__main__":
    main()
