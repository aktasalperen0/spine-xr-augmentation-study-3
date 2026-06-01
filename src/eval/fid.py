"""FID between a set of real patches and a set of synthetic patches.

Thin wrapper over `pytorch-fid`. Both inputs are lists of PNG paths (e.g. real patches of a class
from the ROI manifest, and a generator's manifest paths). We stage them into temp dirs because
pytorch-fid operates on directories.

Note: FID is noisy with few images (some minority classes have <300 real patches). Report it as
an indicative quality measure, not a precise score.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def compute_fid(real_paths: list[str], fake_paths: list[str], device: str = "cuda", batch_size: int = 50) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as fd:
        for i, p in enumerate(real_paths):
            shutil.copy(p, Path(rd) / f"r{i:06d}.png")
        for i, p in enumerate(fake_paths):
            shutil.copy(p, Path(fd) / f"f{i:06d}.png")
        dims = 2048
        bs = min(batch_size, len(real_paths), len(fake_paths))
        return float(calculate_fid_given_paths([rd, fd], bs, device, dims))


def real_paths_for_class(roi_train_csv: str, class_name: str) -> list[str]:
    import pandas as pd
    df = pd.read_csv(roi_train_csv)
    return df[df[class_name] == 1]["path"].tolist()
