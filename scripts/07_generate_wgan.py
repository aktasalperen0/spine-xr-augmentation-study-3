"""Phase 07: generate synthetic images from a trained WGAN checkpoint.

Loads a checkpoint produced by `scripts/05_train_wgan.py`, reconstructs the Generator
exactly (loss kind, channels, latent dim, image size are all stored in the checkpoint),
then samples N synthetic images and writes them as PNGs.

A label-table CSV (matching the audit/02_splits column convention) is also written, with
the target class set to 1 and all other classes set to 0. Phase 08 (hybrid set construction)
will read this CSV and append rows into a case's train CSV — rebinding only the columns
it needs.

Usage:
    python scripts/07_generate_wgan.py \\
        --checkpoint outputs/05_wgan/case_1/disc_space_narrowing/checkpoints/iter_006000.pt \\
        --n-samples 800 \\
        --batch-size 32

    # Or override the target class name (otherwise read from the checkpoint).
    python scripts/07_generate_wgan.py --checkpoint <path> --n-samples 600 --class-name "Disc space narrowing"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from src.models.wgan import Generator
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


# Match the global class set in configs/base.yaml so the manifest is directly compatible
# with outputs/01_audit/*_labels.csv layout.
ALL_CLASSES = [
    "Osteophytes",
    "Disc space narrowing",
    "Other lesions",
    "Foraminal stenosis",
    "Surgical implant",
    "Spondylolysthesis",
    "Vertebral collapse",
    "No finding",
]


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True, help="Path to a Phase-05 checkpoint .pt file")
    ap.add_argument("--n-samples", type=int, default=800, help="Number of synthetic images to generate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--class-name", default=None,
                    help="Target class name. If omitted, taken from the checkpoint's embedded config.")
    ap.add_argument("--class-slug", default=None,
                    help="Output sub-dir slug. If omitted, derived from class name.")
    ap.add_argument("--out-tag", default="07_wgan_generated",
                    help="Sub-dir under outputs/. Default 07_wgan_generated.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    base = load_config(args.config)
    log = get_logger()
    set_seed(args.seed)
    device = _select_device()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    log.info(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    class_name = args.class_name or cfg["class_name"]
    if class_name not in ALL_CLASSES:
        raise ValueError(f"class_name {class_name!r} not in known classes {ALL_CLASSES}")
    class_slug = args.class_slug or cfg.get("class_slug") or _slugify(class_name)
    log.info(
        f"checkpoint meta: class={class_name!r} slug={class_slug!r} "
        f"loss={cfg['loss']} latent_dim={cfg['latent_dim']} image_size={cfg['image_size']} "
        f"out_channels={cfg['out_channels']} g_iter={ckpt.get('g_iter', '?')}"
    )

    G = Generator(
        latent_dim=int(cfg["latent_dim"]),
        base_channels=int(cfg["g_base_channels"]),
        out_channels=int(cfg["out_channels"]),
        image_size=int(cfg["image_size"]),
    ).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag / class_slug
    pngs_dir = out_root / "pngs"
    pngs_dir.mkdir(parents=True, exist_ok=True)

    iter_tag = f"iter_{int(ckpt.get('g_iter', 0)):06d}"
    rows: list[dict] = []
    n = int(args.n_samples)
    bs = int(args.batch_size)

    log.info(f"generating {n} samples (batch_size={bs}, device={device.type}) -> {pngs_dir}")
    written = 0
    rng = np.random.default_rng(args.seed)
    while written < n:
        cur_bs = min(bs, n - written)
        z = torch.randn(cur_bs, int(cfg["latent_dim"]), device=device)
        with torch.no_grad():
            x = G(z).clamp(-1.0, 1.0)
        # x: (B, C, H, W) in [-1, 1]
        # -> uint8 grayscale [0, 255]. If C==3 we average to grayscale to match dataset convention.
        if x.size(1) == 1:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

        for i in range(cur_bs):
            image_id = f"wgan_{class_slug}_{iter_tag}_{written + i:06d}"
            png_path = pngs_dir / f"{image_id}.png"
            cv2.imwrite(str(png_path), x[i])
            row = {
                "image_id": image_id,
                "study_id": f"wgan_synth_{class_slug}",
                "path": str(png_path),
                "source": "wgan_synth",
            }
            for c in ALL_CLASSES:
                row[c] = 1 if c == class_name else 0
            row["transform"] = "wgan_real"  # tag for traceability; aug copies will be added in Phase 08
            row["wgan_iter"] = int(ckpt.get("g_iter", 0))
            rows.append(row)
        written += cur_bs
        log.info(f"  {written}/{n}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifest.csv", index=False)
    (out_root / "manifest.json").write_text(json.dumps({
        "class_name": class_name,
        "class_slug": class_slug,
        "checkpoint": str(ckpt_path),
        "checkpoint_iter": int(ckpt.get("g_iter", 0)),
        "n_samples": n,
        "loss": cfg["loss"],
        "latent_dim": int(cfg["latent_dim"]),
        "image_size": int(cfg["image_size"]),
        "out_channels": int(cfg["out_channels"]),
        "seed": args.seed,
    }, indent=2))
    log.info(f"wrote {n} pngs to {pngs_dir}")
    log.info(f"wrote manifest: {out_root/'manifest.csv'}")


if __name__ == "__main__":
    main()
