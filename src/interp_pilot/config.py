"""Experiment configuration (YAML). A config may name a parent with `inherit: <file>`; keys are deep-merged."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load(path.parent / parent), cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg
