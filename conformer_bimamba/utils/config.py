"""Config loading + CLI override helpers shared by train/evaluate/inference."""
from __future__ import annotations

import yaml


def parse_override(s: str):
    """Parse a CLI override value: int / float / bool / list / str."""
    s = s.strip()
    if "," in s:
        return [parse_override(x) for x in s.split(",")]
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for o in overrides:
        key, _, val = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_override(val)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides or [])
    cfg["_config_path"] = path
    return cfg
