"""Phase 07p: generate synthetic ROI patches from a trained ProGAN checkpoint (full 128 depth).

Writes PNGs + manifest.csv (07 schema) for the CV driver's --synth-manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
import torch

from src.models.progan import Generator
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

ALL_CLASSES = [
    "Osteophytes", "Disc space narrowing", "Other lesions", "Foraminal stenosis",
    "Surgical implant", "Spondylolysthesis", "Vertebral collapse", "No finding",
]


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--out-tag", default="07p_progan_generated")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    base = load_config(args.config)
    log = get_logger(); set_seed(args.seed)
    device = _device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    class_name = args.class_name or cfg["class_name"]
    if class_name not in ALL_CLASSES:
        raise ValueError(f"{class_name!r} not in {ALL_CLASSES}")
    slug = cfg.get("class_slug") or class_name.lower().replace(" ", "_")

    G = Generator(int(cfg["latent_dim"]), int(cfg["out_channels"]), int(cfg["max_depth"])).to(device)
    G.load_state_dict(ckpt["G_state"]); G.eval()
    depth = int(cfg["max_depth"])

    out_root = Path(base["paths"]["outputs_root"]) / args.out_tag / slug
    pngs = out_root / "pngs"; pngs.mkdir(parents=True, exist_ok=True)
    rows = []; written = 0
    log.info(f"generating {args.n_samples} ProGAN samples for {class_name!r}")
    while written < args.n_samples:
        bs = min(args.batch_size, args.n_samples - written)
        z = torch.randn(bs, int(cfg["latent_dim"]), device=device)
        with torch.no_grad():
            x = G(z, depth, 1.0).clamp(-1, 1)
        x = x[:, 0] if x.size(1) == 1 else x.mean(1)
        x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for i in range(bs):
            iid = f"progan_{slug}_{written+i:06d}"
            p = pngs / f"{iid}.png"; cv2.imwrite(str(p), x[i])
            row = {"image_id": iid, "study_id": f"progan_synth_{slug}", "path": str(p), "source": "progan_synth"}
            for c in ALL_CLASSES:
                row[c] = 1 if c == class_name else 0
            row["transform"] = "synth_real"
            rows.append(row)
        written += bs; log.info(f"  {written}/{args.n_samples}")
    pd.DataFrame(rows).to_csv(out_root / "manifest.csv", index=False)
    (out_root / "manifest.json").write_text(json.dumps(
        {"class_name": class_name, "class_slug": slug, "checkpoint": args.checkpoint,
         "n_samples": args.n_samples}, indent=2))
    log.info(f"wrote manifest {out_root/'manifest.csv'}")


if __name__ == "__main__":
    main()
