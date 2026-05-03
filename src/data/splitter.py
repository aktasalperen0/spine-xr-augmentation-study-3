"""Per-Case split logic for Project 3.

Builds, for each Case (defined in configs/cases.yaml):

- train.csv          : official VinDr train images selected for the case, NF capped at
                       `no_finding_cap_train`, with a study-disjoint internal-val slice removed.
- internal_val.csv   : the carved-out study-disjoint slice from the train pool.
- test.csv           : official VinDr test images for the case (NF uncapped, untouched).
- manifest.json      : counts, seed, cap, code & data version metadata.

Decisions baked in (see plan):
- D1: keep all positive-bearing images; only NF is capped.
- D2: multi-label allowed inside a case (the case-positives columns are kept as a
      multi-hot vector; an image with both case classes contributes to both).
- D3: official VinDr train/test split is used as-is.
- D8: 10% study-disjoint internal-val cut from train (guardrail, not the test set).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

NO_FINDING = "No finding"


@dataclass
class CaseSpec:
    name: str
    positives: list[str]
    second_rotation_deg: int


def load_cases(cases_cfg: dict) -> list[CaseSpec]:
    cases = []
    for name, body in cases_cfg["cases"].items():
        cases.append(
            CaseSpec(
                name=name,
                positives=list(body["positives"]),
                second_rotation_deg=int(body["second_rotation_deg"]),
            )
        )
    return cases


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _select_for_case(df: pd.DataFrame, case: CaseSpec) -> pd.DataFrame:
    """Rows where any case-positive is 1 OR No-finding is 1."""
    pos_mask = df[case.positives].sum(axis=1) > 0
    nf_mask = df[NO_FINDING] == 1
    return df[pos_mask | nf_mask].copy()


def _cap_no_finding(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Subsample No-finding rows to at most `cap` (study-disjoint random sample)."""
    pos_rows = df[df[NO_FINDING] == 0]
    nf_rows = df[df[NO_FINDING] == 1]
    if len(nf_rows) <= cap:
        return df
    # Sample studies (not images), then take all images from those studies up to the cap.
    studies = nf_rows["study_id"].drop_duplicates().to_numpy()
    rng.shuffle(studies)
    picked: list[str] = []
    n_picked_imgs = 0
    study_to_n = nf_rows.groupby("study_id").size()
    for s in studies:
        if n_picked_imgs >= cap:
            break
        picked.append(s)
        n_picked_imgs += int(study_to_n.loc[s])
    nf_keep = nf_rows[nf_rows["study_id"].isin(picked)]
    if len(nf_keep) > cap:
        # Trim images at random within picked studies to land exactly at `cap`.
        nf_keep = nf_keep.sample(n=cap, random_state=int(rng.integers(0, 2**31 - 1)))
    return pd.concat([pos_rows, nf_keep], ignore_index=True)


def _internal_val_split(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fraction <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["study_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _case_columns(case: CaseSpec) -> list[str]:
    """Output label columns for the case: case-positives + No finding."""
    return list(case.positives) + [NO_FINDING]


def build_case_split(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    case: CaseSpec,
    no_finding_cap_train: int,
    internal_val_fraction: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    keep_cols = ["image_id", "study_id", "path", "source"] + _case_columns(case)

    # ---- TRAIN side ----
    train_pool = _select_for_case(train_labels, case)
    train_pool = _cap_no_finding(train_pool, no_finding_cap_train, rng)
    train_pool = train_pool[keep_cols].reset_index(drop=True)

    train_df, internal_val_df = _internal_val_split(train_pool, internal_val_fraction, seed)

    # ---- TEST side (untouched) ----
    test_df = _select_for_case(test_labels, case)[keep_cols].reset_index(drop=True)

    return {
        "train": train_df,
        "internal_val": internal_val_df,
        "test": test_df,
    }


def assert_split_invariants(case: CaseSpec, splits: dict) -> None:
    train, ival, test = splits["train"], splits["internal_val"], splits["test"]

    # study-id disjointness within {train, internal_val, test}
    s_tr, s_iv, s_te = set(train["study_id"]), set(ival["study_id"]), set(test["study_id"])
    assert not (s_tr & s_iv), f"{case.name}: study leak train <-> internal_val"
    assert not (s_tr & s_te), f"{case.name}: study leak train <-> test"
    assert not (s_iv & s_te), f"{case.name}: study leak internal_val <-> test"

    # every case-positive class has at least 1 row in test
    for c in case.positives:
        n = int(test[c].sum()) if c in test.columns else 0
        assert n >= 1, f"{case.name}: test has 0 positives for {c}"

    # all paths exist on disk
    missing = [p for p in train["path"].tolist() + ival["path"].tolist() + test["path"].tolist() if not Path(p).exists()]
    assert not missing, f"{case.name}: {len(missing)} missing paths (first: {missing[:1]})"


def write_case_split(out_dir: Path, case: CaseSpec, splits: dict, seed: int, no_finding_cap_train: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        counts[name] = {
            "n_rows": int(len(df)),
            "n_studies": int(df["study_id"].nunique()),
            **{c: int(df[c].sum()) for c in _case_columns(case)},
        }
    manifest = {
        "case": case.name,
        "positives": case.positives,
        "second_rotation_deg": case.second_rotation_deg,
        "no_finding_cap_train": no_finding_cap_train,
        "seed": seed,
        "git_hash": _git_hash(),
        "counts": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def render_summary_md(manifests: list[dict]) -> str:
    lines = ["# Splits summary (Project 3)", ""]
    for m in manifests:
        lines.append(f"## {m['case']} — positives: {', '.join(m['positives'])}")
        lines.append("")
        lines.append("| split | n_rows | n_studies | " + " | ".join(m["positives"]) + " | No finding |")
        lines.append("|---|---|---|" + "|".join(["---"] * (len(m["positives"]) + 1)) + "|")
        for split_name in ("train", "internal_val", "test"):
            c = m["counts"][split_name]
            row = [split_name, str(c["n_rows"]), str(c["n_studies"])]
            row += [str(c[p]) for p in m["positives"]]
            row.append(str(c[NO_FINDING]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)
