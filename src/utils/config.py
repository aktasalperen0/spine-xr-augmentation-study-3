from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = v
    return out


def load_with_base(path: str | Path, base_path: str | Path = "configs/base.yaml") -> dict:
    return merge_configs(load_config(base_path), load_config(path))
