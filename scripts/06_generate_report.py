"""Phase 06: comprehensive thesis report generator.

Auto-scans outputs/ and produces publication-quality figures (300dpi PNG+PDF), Markdown+LaTeX
tables, and an auto-written insights/discussion text. Degrades gracefully when a condition or
artifact is missing (e.g. run it before ProGAN finishes).

Layers:
  L1 aggregate  : read cv_summary.json, fold metrics.json, log.csv, acceptance reports, FID jsons
  L2 reinfer    : ensemble the 3 fold best.pth on the real held-out test -> probs+labels
                  (the ONLY way to get confusion matrices / ROC / PR curves; they aren't persisted)
  L3 figures    : CMs, ROC/PR, bar charts, learning curves, class-distribution, win-heatmap,
                  fold-variance boxplots, radar, real-vs-synth montages
  L4 tables/text: macro & per-class mean±std (MD+LaTeX), synthetic-quality table,
                  ablation_insights.txt, report.md

Outputs to outputs/06_reports/{figures,tables,*.txt,*.md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

from src.utils.config import load_config
from src.utils.logging import get_logger

NO_FINDING = "No finding"
CONDITIONS = [
    ("03cv_baseline", "Baseline"),
    ("04cv_traditional", "Traditional"),
    ("05cv_diffusion", "DDPM"),
    ("06cv_progan", "ProGAN"),
]
# slug -> (class_name, case)
SYNTH_CLASSES = {
    "disc_space_narrowing": ("Disc space narrowing", "case_1"),
    "vertebral_collapse": ("Vertebral collapse", "case_1"),
    "foraminal_stenosis": ("Foraminal stenosis", "case_2"),
    "spondylolysthesis": ("Spondylolysthesis", "case_2"),
    "surgical_implant": ("Surgical implant", "case_3"),
    "other_lesions": ("Other lesions", "case_4"),
}
PALETTE = {"Baseline": "#9aa7c7", "Traditional": "#4c72b0", "DDPM": "#55a868", "ProGAN": "#c44e52"}


# ---------------------------------------------------------------- helpers

def case_label_columns(case_body: dict) -> list[str]:
    return list(case_body["positives"]) + [NO_FINDING]


def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def present_conditions(outputs_root: Path) -> list[tuple[str, str]]:
    return [(tag, lbl) for tag, lbl in CONDITIONS if (outputs_root / tag / "cv_summary.json").exists()]


# ---------------------------------------------------------------- L1 aggregate

def load_cv_summaries(outputs_root: Path, conds) -> pd.DataFrame:
    rows = []
    for tag, lbl in conds:
        for s in json.loads((outputs_root / tag / "cv_summary.json").read_text()):
            row = {"condition": lbl, "tag": tag, "case": s["case"],
                   "macro_mean": s["test_macro_f1_mean"], "macro_std": s["test_macro_f1_std"],
                   "folds": s["per_fold_test_macro_f1"]}
            for c, v in s["per_class_test_f1_mean"].items():
                row[f"f1__{c}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def load_fold_perclass(outputs_root: Path, tag: str, case: str, backbone="densenet121"):
    """Per-class precision/recall/f1 across folds -> dict[class] -> dict[metric] -> (mean,std)."""
    base = outputs_root / tag / case / backbone
    folds = []
    for fd in sorted(base.glob("fold_*")):
        mp = fd / "metrics.json"
        if mp.exists():
            folds.append(json.loads(mp.read_text())["best_test_metrics"]["per_class"])
    if not folds:
        return {}
    classes = list(folds[0].keys())
    out = {}
    for c in classes:
        out[c] = {}
        for m in ("precision", "recall", "f1", "auroc", "ap"):
            vals = [f[c][m] for f in folds if c in f and f[c].get(m) == f[c].get(m)]
            out[c][m] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)
        out[c]["n_pos"] = int(folds[0][c].get("n_pos", 0))
    return out


# ---------------------------------------------------------------- L2 re-inference

def reinfer(outputs_root: Path, tag: str, case: str, label_cols, backbone, image_size, device, log):
    """Ensemble the 3 fold best.pth on the real test set. Returns (y_true[N,C], y_prob[N,C]) or None."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.case_dataset import CaseDataset
    from src.data.transforms import build_transform
    from src.models.classifier import build_classifier

    base = outputs_root / tag / case / backbone
    ckpts = [fd / "best.pth" for fd in sorted(base.glob("fold_*")) if (fd / "best.pth").exists()]
    test_csv = outputs_root / "02b_roi" / case / "test.csv"
    if not ckpts or not test_csv.exists():
        return None
    df = pd.read_csv(test_csv)
    tf = build_transform("val", image_size)
    loader = DataLoader(CaseDataset(df, label_cols, tf), batch_size=128, shuffle=False, num_workers=4)

    models = []
    for cp in ckpts:
        ck = torch.load(cp, map_location=device, weights_only=False)
        m = build_classifier(backbone, num_classes=len(label_cols)).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)

    probs, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = sum(torch.sigmoid(m(x)) for m in models) / len(models)
            probs.append(p.cpu().numpy()); trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(probs)


# ---------------------------------------------------------------- L3 figures

def fig_macro_box(df, fig_dir):
    cases = sorted(df["case"].unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4), squeeze=False)
    for ax, case in zip(axes[0], cases):
        sub = df[df["case"] == case]
        data = [s["folds"] for _, s in sub.iterrows()]
        labels = list(sub["condition"])
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=30)
        ax.set_title(case); ax.set_ylabel("test macro-F1")
    fig.suptitle("3-fold variance (model stability)")
    savefig(fig, fig_dir, "fold_variance_boxplots")


def fig_macro_bars(df, fig_dir):
    cases = sorted(df["case"].unique()); conds = list(df["condition"].unique())
    x = np.arange(len(cases)); w = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(1.8 * len(cases) + 2, 4.5))
    for i, cond in enumerate(conds):
        means = [df[(df.case == c) & (df.condition == cond)]["macro_mean"].mean() for c in cases]
        stds = [df[(df.case == c) & (df.condition == cond)]["macro_std"].mean() for c in cases]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=cond,
               color=PALETTE.get(cond, None))
    ax.set_xticks(x + w * (len(conds) - 1) / 2); ax.set_xticklabels(cases)
    ax.set_ylabel("test macro-F1 (mean±std)"); ax.set_ylim(0.6, 1.0); ax.legend()
    ax.set_title("Macro-F1 by condition")
    savefig(fig, fig_dir, "macro_f1_bars")


def fig_perclass_bars(outputs_root, conds, cases_cfg, fig_dir):
    for case, body in cases_cfg["cases"].items():
        label_cols = case_label_columns(body)
        data = {}  # cond -> class -> (f1 mean, std)
        for tag, lbl in conds:
            pc = load_fold_perclass(outputs_root, tag, case)
            if pc:
                data[lbl] = pc
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(2.2 * len(label_cols) + 2, 4.5))
        x = np.arange(len(label_cols)); w = 0.8 / max(1, len(data))
        for i, (lbl, pc) in enumerate(data.items()):
            means = [pc.get(c, {}).get("f1", (np.nan, 0))[0] for c in label_cols]
            stds = [pc.get(c, {}).get("f1", (np.nan, 0))[1] for c in label_cols]
            ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=lbl, color=PALETTE.get(lbl, None))
        ax.set_xticks(x + w * (len(data) - 1) / 2)
        ax.set_xticklabels([c[:16] for c in label_cols], rotation=20)
        ax.set_ylabel("F1 (mean±std)"); ax.set_ylim(0, 1.0); ax.legend(); ax.set_title(f"{case}: per-class F1")
        savefig(fig, fig_dir, f"perclass_f1_{case}")


def fig_learning_curves(outputs_root, conds, fig_dir):
    for tag, lbl in conds:
        for case_dir in sorted((outputs_root / tag).glob("case_*")):
            logs = []
            for fd in sorted((case_dir / "densenet121").glob("fold_*")):
                lp = fd / "log.csv"
                if lp.exists():
                    logs.append(pd.read_csv(lp))
            if not logs:
                continue
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax2 = ax1.twinx()
            for k, lg in enumerate(logs):
                ax1.plot(lg["epoch"], lg["train_loss"], color="tab:red", alpha=0.4)
                ax2.plot(lg["epoch"], lg["test_macro_f1"], color="tab:blue", alpha=0.5)
            ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
            ax2.set_ylabel("test macro-F1", color="tab:blue"); ax2.set_ylim(0, 1)
            ax1.set_title(f"{lbl} / {case_dir.name}: learning curves (3 folds)")
            savefig(fig, fig_dir, f"learning_curve_{tag}_{case_dir.name}")


def fig_confusion_roc_pr(outputs_root, conds, cases_cfg, fig_dir, device, log):
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
    for case, body in cases_cfg["cases"].items():
        label_cols = case_label_columns(body)
        inf = {}
        for tag, lbl in conds:
            r = reinfer(outputs_root, tag, case, label_cols, "densenet121", 224, device, log)
            if r is not None:
                inf[lbl] = r
        if not inf:
            continue
        # Confusion matrices (one per condition)
        for lbl, (yt, yp) in inf.items():
            true_idx = yt.argmax(1); pred_idx = yp.argmax(1)
            cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(label_cols))))
            fig, ax = plt.subplots(figsize=(4.5, 4))
            if HAVE_SNS:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=[c[:10] for c in label_cols],
                            yticklabels=[c[:10] for c in label_cols], ax=ax)
            else:
                ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{case} / {lbl}")
            savefig(fig, fig_dir, f"confusion_{case}_{lbl}")
        # ROC + PR (macro-average over classes), overlay conditions
        for curve in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for lbl, (yt, yp) in inf.items():
                xs = np.linspace(0, 1, 200); ys_acc = []
                for ci in range(len(label_cols)):
                    if yt[:, ci].sum() == 0:
                        continue
                    if curve == "roc":
                        fpr, tpr, _ = roc_curve(yt[:, ci], yp[:, ci]); ys_acc.append(np.interp(xs, fpr, tpr))
                    else:
                        pr, rc, _ = precision_recall_curve(yt[:, ci], yp[:, ci])
                        ys_acc.append(np.interp(xs, rc[::-1], pr[::-1]))
                if ys_acc:
                    ym = np.mean(ys_acc, 0)
                    a = auc(xs, ym)
                    ax.plot(xs, ym, label=f"{lbl} (AUC={a:.3f})", color=PALETTE.get(lbl, None))
            if curve == "roc":
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                ax.set_title(f"{case}: macro-avg ROC")
            else:
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"{case}: macro-avg PR")
            ax.legend(fontsize=8)
            savefig(fig, fig_dir, f"{curve}_{case}")


def fig_win_heatmap(df, fig_dir):
    piv = df.pivot_table(index="case", columns="condition", values="macro_mean")
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 2, 1.0 * piv.shape[0] + 2))
    if HAVE_SNS:
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    else:
        ax.imshow(piv.values, cmap="YlGnBu")
    ax.set_title("Macro-F1 heatmap (case × condition)")
    savefig(fig, fig_dir, "win_heatmap")


def fig_radar(outputs_root, conds, cases_cfg, fig_dir):
    for case, body in cases_cfg["cases"].items():
        label_cols = case_label_columns(body)
        if len(label_cols) < 3:
            continue  # radar needs >=3 axes
        series = {}
        for tag, lbl in conds:
            pc = load_fold_perclass(outputs_root, tag, case)
            if pc:
                series[lbl] = [pc.get(c, {}).get("f1", (0, 0))[0] for c in label_cols]
        if not series:
            continue
        ang = np.linspace(0, 2 * np.pi, len(label_cols), endpoint=False).tolist(); ang += ang[:1]
        fig = plt.figure(figsize=(5, 5)); ax = plt.subplot(polar=True)
        for lbl, vals in series.items():
            v = vals + vals[:1]
            ax.plot(ang, v, label=lbl, color=PALETTE.get(lbl, None)); ax.fill(ang, v, alpha=0.08)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([c[:12] for c in label_cols], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"{case}: per-class F1 radar"); ax.legend(loc="upper right", fontsize=7)
        savefig(fig, fig_dir, f"radar_{case}")


def fig_class_distribution(outputs_root, cases_cfg, fig_dir, log):
    """Real per-class counts (cv_folds) + N_synth for DDPM/ProGAN (acceptance reports)."""
    rows = []
    for case, body in cases_cfg["cases"].items():
        cv = outputs_root / "02b_roi" / case / "cv_folds.csv"
        if not cv.exists():
            continue
        df = pd.read_csv(cv); label_cols = case_label_columns(body)
        for c in label_cols:
            rows.append({"case": case, "class": c, "stage": "Real", "count": int(df[c].sum())})
        for stage, tag in (("+DDPM", "07f_diffusion_filtered"), ("+ProGAN", "07f_progan_filtered")):
            for slug, (cname, ccase) in SYNTH_CLASSES.items():
                if ccase != case:
                    continue
                rp = outputs_root / tag / slug / "acceptance_report.json"
                if rp.exists():
                    rep = json.loads(rp.read_text())
                    rows.append({"case": case, "class": cname, "stage": stage, "count": int(rep.get("n_kept", 0))})
    if not rows:
        return
    dd = pd.DataFrame(rows)
    for case in sorted(dd["case"].unique()):
        sub = dd[dd["case"] == case]
        piv = sub.pivot_table(index="class", columns="stage", values="count", aggfunc="sum", fill_value=0)
        order = [s for s in ["Real", "+DDPM", "+ProGAN"] if s in piv.columns]
        fig, ax = plt.subplots(figsize=(2 * len(piv) + 2, 4))
        piv[order].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_ylabel("train patch count"); ax.set_title(f"{case}: class distribution (real + accepted synthetic)")
        ax.tick_params(axis="x", rotation=20)
        savefig(fig, fig_dir, f"class_distribution_{case}")


def fig_montages(outputs_root, fig_dir, log):
    import cv2
    for slug, (cname, case) in SYNTH_CLASSES.items():
        cols = []  # (title, paths)
        real_csv = outputs_root / "02b_roi" / case / "train.csv"
        if real_csv.exists():
            rdf = pd.read_csv(real_csv); rdf = rdf[rdf[cname] == 1] if cname in rdf.columns else rdf.iloc[0:0]
            cols.append(("Real", rdf["path"].head(4).tolist()))
        for title, tag in (("DDPM", "07f_diffusion_filtered"), ("ProGAN", "07f_progan_filtered")):
            mp = outputs_root / tag / slug / "manifest.csv"
            if mp.exists():
                cols.append((title, pd.read_csv(mp)["path"].head(4).tolist()))
        cols = [(t, p) for t, p in cols if p]
        if not cols:
            continue
        nrow = 4; ncol = len(cols)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), squeeze=False)
        for j, (title, paths) in enumerate(cols):
            axes[0, j].set_title(title)
            for i in range(nrow):
                ax = axes[i, j]; ax.axis("off")
                if i < len(paths):
                    im = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
                    if im is not None:
                        ax.imshow(im, cmap="gray")
        fig.suptitle(f"{cname}: real vs synthetic")
        savefig(fig, fig_dir, f"montage_{slug}")


# ---------------------------------------------------------------- L4 tables + text

def write_tables(df, outputs_root, conds, cases_cfg, tab_dir, log):
    tab_dir.mkdir(parents=True, exist_ok=True)
    # Macro table
    macro = df.copy()
    macro["macroF1"] = macro.apply(lambda r: f"{r['macro_mean']:.4f} ± {r['macro_std']:.4f}", axis=1)
    mt = macro.pivot(index="case", columns="condition", values="macroF1")
    (tab_dir / "macro_f1.md").write_text("# Macro-F1 (mean±std)\n\n" + mt.to_markdown())
    (tab_dir / "macro_f1.tex").write_text(mt.to_latex(caption="Macro-F1 (mean±std) by condition.", label="tab:macro"))
    # Per-class table (f1 mean±std)
    rows = []
    for case, body in cases_cfg["cases"].items():
        for c in case_label_columns(body):
            row = {"case": case, "class": c}
            for tag, lbl in conds:
                pc = load_fold_perclass(outputs_root, tag, case)
                if c in pc:
                    m, s = pc[c]["f1"]; row[lbl] = f"{m:.3f}±{s:.3f}"
                    row.setdefault("n_pos", pc[c]["n_pos"])
            rows.append(row)
    pcdf = pd.DataFrame(rows)
    (tab_dir / "per_class_f1.md").write_text("# Per-class F1 (mean±std)\n\n" + pcdf.to_markdown(index=False))
    (tab_dir / "per_class_f1.tex").write_text(pcdf.to_latex(index=False, caption="Per-class F1 (mean±std).", label="tab:perclass"))
    return mt, pcdf


def write_synth_quality(outputs_root, tab_dir):
    rows = []
    for slug, (cname, case) in SYNTH_CLASSES.items():
        rec = {"class": cname, "case": case}
        sel = outputs_root / "06d_ckpt_select" / f"{slug}.json"
        if sel.exists():
            s = json.loads(sel.read_text()); rec["DDPM_best_iter"] = s.get("best_iter"); rec["DDPM_FID"] = round(s.get("best_fid", float("nan")), 2)
        for title, tag in (("DDPM", "07f_diffusion_filtered"), ("ProGAN", "07f_progan_filtered")):
            rp = outputs_root / tag / slug / "acceptance_report.json"
            if rp.exists():
                r = json.loads(rp.read_text())
                rec[f"{title}_accept%"] = round(100 * r.get("acceptance_rate", 0), 1)
                rec[f"{title}_n_kept"] = r.get("n_kept")
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        (tab_dir / "synthetic_quality.md").write_text("# Synthetic quality (FID + jury acceptance)\n\n" + df.to_markdown(index=False))
    return df


def write_insights(df, outputs_root, conds, cases_cfg, out_root):
    lines = ["# Auto-generated insights", ""]
    # macro deltas vs baseline
    for case in sorted(df["case"].unique()):
        sub = df[df["case"] == case].set_index("condition")
        if "Baseline" not in sub.index:
            continue
        base = sub.loc["Baseline", "macro_mean"]
        for cond in [l for _, l in conds if l != "Baseline" and l in sub.index]:
            d = sub.loc[cond, "macro_mean"] - base
            std = max(sub.loc[cond, "macro_std"], sub.loc["Baseline", "macro_std"])
            sig = "" if abs(d) > std else "  (within fold-std → not significant)"
            lines.append(f"- {case}: {cond} macro-F1 Δ vs Baseline = {d:+.4f}{sig}")
    lines.append("")
    # per-class minority gains
    lines.append("## Per-class (minority focus)")
    for case, body in cases_cfg["cases"].items():
        base_pc = load_fold_perclass(outputs_root, "03cv_baseline", case)
        for c in body["positives"]:
            if c not in base_pc:
                continue
            b = base_pc[c]["f1"][0]
            seg = [f"{c} ({case}, n_pos={base_pc[c]['n_pos']}): baseline F1={b:.3f}"]
            for tag, lbl in conds:
                if lbl == "Baseline":
                    continue
                pc = load_fold_perclass(outputs_root, tag, case)
                if c in pc:
                    seg.append(f"{lbl}={pc[c]['f1'][0]:.3f} (Δ{pc[c]['f1'][0]-b:+.3f})")
            lines.append("- " + "; ".join(seg))
    (out_root / "ablation_insights.txt").write_text("\n".join(lines))
    return lines


def write_report_md(out_root, conds, insights_lines):
    fig = "figures"; tab = "tables"
    md = ["# Project 3 — Results Report", "",
          "Conditions present: " + ", ".join(l for _, l in conds), "",
          "## Headline tables", f"- [Macro-F1]({tab}/macro_f1.md) · [Per-class F1]({tab}/per_class_f1.md) · [Synthetic quality]({tab}/synthetic_quality.md)", "",
          "## Key figures",
          f"- Macro-F1 bars: `{fig}/macro_f1_bars.png`", f"- Per-class F1: `{fig}/perclass_f1_<case>.png`",
          f"- Confusion matrices: `{fig}/confusion_<case>_<cond>.png`", f"- ROC/PR: `{fig}/roc_<case>.png`, `{fig}/pr_<case>.png`",
          f"- Fold-variance boxplots: `{fig}/fold_variance_boxplots.png`", f"- Radar: `{fig}/radar_<case>.png`",
          f"- Class distribution: `{fig}/class_distribution_<case>.png`", f"- Real-vs-synthetic montages: `{fig}/montage_<slug>.png`",
          f"- Win heatmap: `{fig}/win_heatmap.png`", "", "## Auto insights", ""]
    md += insights_lines[:60]
    (out_root / "report.md").write_text("\n".join(md))


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--cases", default="configs/cases.yaml")
    ap.add_argument("--no-reinfer", action="store_true", help="skip CM/ROC/PR (no GPU / fast mode)")
    args = ap.parse_args()

    base = load_config(args.config)
    cases_cfg = load_config(args.cases)
    log = get_logger()
    outputs_root = Path(base["paths"]["outputs_root"])
    out_root = outputs_root / "06_reports"
    fig_dir = out_root / "figures"; tab_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)

    conds = present_conditions(outputs_root)
    if not conds:
        log.warning("no cv_summary.json found under any condition — nothing to report"); return
    log.info(f"conditions present: {[l for _,l in conds]}")

    df = load_cv_summaries(outputs_root, conds)

    # Figures that need only aggregated data
    fig_macro_bars(df, fig_dir); fig_macro_box(df, fig_dir); fig_win_heatmap(df, fig_dir)
    fig_perclass_bars(outputs_root, conds, cases_cfg, fig_dir)
    fig_learning_curves(outputs_root, conds, fig_dir)
    fig_radar(outputs_root, conds, cases_cfg, fig_dir)
    try:
        fig_class_distribution(outputs_root, cases_cfg, fig_dir, log)
    except Exception as e:
        log.warning(f"class_distribution skipped: {e}")
    try:
        fig_montages(outputs_root, fig_dir, log)
    except Exception as e:
        log.warning(f"montages skipped: {e}")

    # Re-inference figures (CM/ROC/PR)
    if not args.no_reinfer:
        try:
            import torch
            device = torch.device("cuda") if torch.cuda.is_available() else (
                torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
            fig_confusion_roc_pr(outputs_root, conds, cases_cfg, fig_dir, device, log)
        except Exception as e:
            log.warning(f"re-inference figures skipped: {e}")

    # Tables + text
    write_tables(df, outputs_root, conds, cases_cfg, tab_dir, log)
    write_synth_quality(outputs_root, tab_dir)
    insights = write_insights(df, outputs_root, conds, cases_cfg, out_root)
    write_report_md(out_root, conds, insights)
    log.info(f"report written to {out_root}")


if __name__ == "__main__":
    main()
