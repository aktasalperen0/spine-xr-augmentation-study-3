"""Train a multi-label classifier for one (case, backbone) cell.

- Loss: BCEWithLogitsLoss with optional clipped pos_weight (D2).
- Optim: AdamW + cosine schedule with linear warmup.
- Per-epoch: train -> evaluate on internal_val -> evaluate on official test.
- Saves best.pth on test Macro F1 (D8). internal_val Macro F1 is logged as a guardrail.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.case_dataset import CaseDataset, compute_pos_weight, compute_sample_weights
from src.data.transforms import build_transform
from src.eval.metrics import format_metrics_md, per_class_metrics
from src.models.classifier import build_classifier


@dataclass
class TrainerConfig:
    case_name: str
    backbone: str
    label_columns: list[str]
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    grad_clip_norm: float
    num_workers: int
    train_transform: str
    eval_transform: str
    pos_weight_clip: float
    out_dir: Path
    seed: int = 42
    log_every: int = 50
    balanced_sampler: bool = False          # WeightedRandomSampler (per-class inverse-freq) on train
    select_metric_split: str = "test"       # "val" (CV: select best.pth on fold-val) | "test" (legacy)
    extra: dict = field(default_factory=dict)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(df: pd.DataFrame, label_columns, transform, batch_size, num_workers, shuffle, sampler=None):
    ds = CaseDataset(df, label_columns, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int, steps_per_epoch: int):
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def _evaluate(model, loader, device, label_columns) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return per_class_metrics(y_true, y_prob, label_columns)


def train_one_cell(
    train_df: pd.DataFrame,
    internal_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: TrainerConfig,
    log,
) -> dict:
    device = _select_device()
    log.info(f"[{cfg.case_name}/{cfg.backbone}] device={device.type}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_tf = build_transform(cfg.train_transform, cfg.image_size)
    eval_tf = build_transform(cfg.eval_transform, cfg.image_size)

    train_sampler = None
    if cfg.balanced_sampler:
        sw = compute_sample_weights(train_df, cfg.label_columns)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sw, dtype=torch.double), num_samples=len(sw), replacement=True
        )
        log.info(f"[{cfg.case_name}/{cfg.backbone}] balanced sampler ON (n={len(sw)})")
    train_loader = _make_loader(
        train_df, cfg.label_columns, train_tf, cfg.batch_size, cfg.num_workers,
        shuffle=True, sampler=train_sampler,
    )
    val_loader = (
        _make_loader(internal_val_df, cfg.label_columns, eval_tf, cfg.batch_size, cfg.num_workers, shuffle=False)
        if len(internal_val_df) > 0 else None
    )
    test_loader = _make_loader(
        test_df, cfg.label_columns, eval_tf, cfg.batch_size, cfg.num_workers, shuffle=False
    )

    model = build_classifier(cfg.backbone, num_classes=len(cfg.label_columns)).to(device)

    pos_weight = compute_pos_weight(train_df, cfg.label_columns, clip=cfg.pos_weight_clip).to(device)
    log.info(f"[{cfg.case_name}/{cfg.backbone}] pos_weight={pos_weight.cpu().tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs, len(train_loader))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_rows: list[dict] = []

    best = {"epoch": -1, "sel": -1.0, "test_macro_f1": -1.0, "test_metrics": None, "val_macro_f1": float("nan")}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1
            if step % cfg.log_every == 0:
                log.info(
                    f"[{cfg.case_name}/{cfg.backbone}] epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = running_loss / max(1, n_batches)
        val_metrics = _evaluate(model, val_loader, device, cfg.label_columns) if val_loader else None
        test_metrics = _evaluate(model, test_loader, device, cfg.label_columns)

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_macro_f1": val_metrics["macro_f1"] if val_metrics else float("nan"),
            "test_macro_f1": test_metrics["macro_f1"],
        }
        for c in cfg.label_columns:
            row[f"test_f1__{c}"] = test_metrics["per_class"][c]["f1"]
        epoch_log_rows.append(row)
        log.info(
            f"[{cfg.case_name}/{cfg.backbone}] epoch {epoch}/{cfg.epochs} "
            f"loss={avg_loss:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
            f"test_macro_f1={row['test_macro_f1']:.4f}"
        )

        # Model-selection metric: fold-val macro-F1 in CV mode (keeps test unbiased), else test.
        if cfg.select_metric_split == "val" and val_metrics is not None:
            sel_metric = val_metrics["macro_f1"]
        else:
            sel_metric = test_metrics["macro_f1"]

        if sel_metric > best["sel"]:
            best = {
                "epoch": epoch,
                "sel": sel_metric,
                "test_macro_f1": test_metrics["macro_f1"],
                "test_metrics": test_metrics,
                "val_macro_f1": row["val_macro_f1"],
            }
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "label_columns": cfg.label_columns,
                    "backbone": cfg.backbone,
                    "image_size": cfg.image_size,
                    "test_macro_f1": best["test_macro_f1"],
                },
                out_dir / "best.pth",
            )
            log.info(f"[{cfg.case_name}/{cfg.backbone}] saved best.pth (epoch {epoch})")

    pd.DataFrame(epoch_log_rows).to_csv(out_dir / "log.csv", index=False)
    summary = {
        "case": cfg.case_name,
        "backbone": cfg.backbone,
        "label_columns": cfg.label_columns,
        "epochs": cfg.epochs,
        "best_epoch": best["epoch"],
        "best_test_macro_f1": best["test_macro_f1"],
        "best_val_macro_f1": best["val_macro_f1"],
        "best_test_metrics": best["test_metrics"],
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    if best["test_metrics"] is not None:
        (out_dir / "test_metrics.md").write_text(
            f"# {cfg.case_name} / {cfg.backbone} — best epoch {best['epoch']}\n\n"
            + format_metrics_md(best["test_metrics"], cfg.label_columns)
        )
    return summary
