"""Build multi-label tables from VinDr-SpineXR CSVs (one row per image)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NO_FINDING_CLASS = "No finding"


def lesion_classes(class_names: list[str]) -> list[str]:
    return [c for c in class_names if c != NO_FINDING_CLASS]


def build_label_table(
    annotations_csv: Path,
    normal_metadata_csv: Path,
    abnormal_pngs_dir: Path,
    normal_pngs_dir: Path,
    class_names: list[str],
) -> pd.DataFrame:
    lesions = lesion_classes(class_names)
    ann_raw = pd.read_csv(annotations_csv)
    ann = ann_raw[ann_raw["lesion_type"].isin(lesions)]

    one_hot = (
        ann.assign(v=1)
        .pivot_table(index="image_id", columns="lesion_type", values="v", aggfunc="max", fill_value=0)
        .reindex(columns=lesions, fill_value=0)
        .reset_index()
    )
    one_hot["source"] = "abnormal"
    one_hot["path"] = one_hot["image_id"].apply(lambda i: str(abnormal_pngs_dir / f"{i}.png"))

    if "study_id" in ann_raw.columns:
        ab_studies = ann_raw[["image_id", "study_id"]].drop_duplicates("image_id")
        one_hot = one_hot.merge(ab_studies, on="image_id", how="left")
    else:
        one_hot["study_id"] = pd.NA

    normal_raw = pd.read_csv(normal_metadata_csv)
    normal_cols = ["image_id"] + (["study_id"] if "study_id" in normal_raw.columns else [])
    normal = normal_raw[normal_cols].drop_duplicates("image_id")
    if "study_id" not in normal.columns:
        normal["study_id"] = pd.NA
    for c in lesions:
        normal[c] = 0
    normal["source"] = "normal"
    normal["path"] = normal["image_id"].apply(lambda i: str(normal_pngs_dir / f"{i}.png"))

    df = pd.concat([one_hot, normal], ignore_index=True)
    if NO_FINDING_CLASS in class_names:
        df[NO_FINDING_CLASS] = (df[lesions].sum(axis=1) == 0).astype(int)
    missing_study = df["study_id"].isna()
    if missing_study.any():
        df.loc[missing_study, "study_id"] = df.loc[missing_study, "image_id"]
    df["study_id"] = df["study_id"].astype(str)
    return df[["image_id", "study_id", "path", "source"] + class_names]


def class_counts(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    rows = [{"class": c, "n_positive": int(df[c].sum())} for c in class_names]
    rows.append({"class": "__any_lesion__", "n_positive": int(df[lesion_classes(class_names)].sum(axis=1).gt(0).sum())})
    rows.append({"class": "__no_lesion__", "n_positive": int(df[lesion_classes(class_names)].sum(axis=1).eq(0).sum())})
    return pd.DataFrame(rows)


def co_occurrence(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    return df[class_names].T.dot(df[class_names])
