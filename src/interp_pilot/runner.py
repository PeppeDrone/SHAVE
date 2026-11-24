"""Shared plumbing for the per-model scripts: paths, data, blind pool, base predictions, scoring, recording."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from . import civil as C
from .artifacts import RunArtifacts, now_rome, sha256_file, write_json
from .metrics import paper_metrics

THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def set_thread_env():
    for k, v in THREAD_ENV.items():
        os.environ.setdefault(k, v)


def data_dir(out: Path) -> Path:
    return Path(out) / "data"


def results_root(out: Path) -> Path:
    return Path(out) / "results"


def base_model_path(cfg: dict, out: Path, arm: str, seed: int) -> str:
    if cfg.get("smoke"):
        return str(Path(out) / "_tiny" / arm / f"s{seed}" / "base")
    return str(cfg["model"].get("local_dir") or os.environ.get("IP_BASE_MODEL") or "/tmp/hf_local/Qwen3.5-0.8B")


def adapter_path(out: Path, arm: str, seed: int) -> Path:
    return results_root(out) / arm / f"s{seed}" / "train" / "adapter"


def device_for(cfg: dict) -> str:
    import torch
    return "cpu" if cfg.get("smoke") or not torch.cuda.is_available() else "cuda"


def artifacts(out: Path, arm: str, seed: int, config: dict | None = None) -> RunArtifacts:
    return RunArtifacts(results_root(out), arm, seed, config=config)


def load_eval(out: Path) -> dict:
    return C.load_eval(data_dir(out))


def blind_pool(out: Path, seed: int, pool_size: int):
    """Validation rows shuffled with random.Random(seed); the pool is the first pool_size rows.
    Returns (rng, val_shuffled, pool_rows). The same rng object continues to be used by SHAVE (step 1 swaps)."""
    rng = random.Random(seed)
    val = C.jl(data_dir(out) / "val.jsonl"); rng.shuffle(val)
    return rng, val, val[:pool_size]


def outside_pool(out: Path, seed: int, pool_size: int) -> list[dict]:
    _, val, pool = blind_pool(out, seed, pool_size)
    ids = {r["id"] for r in pool}
    return [r for r in val if r["id"] not in ids]


def load_model(cfg: dict, out: Path, arm: str, seed: int):
    from .model import load_classifier
    return load_classifier(base_model_path(cfg, out, arm, seed), str(adapter_path(out, arm, seed)), device=device_for(cfg),
                           max_length=int(cfg["model"]["max_length"]), fp32=True)


def score_held(m, held: dict, ivs=None, batch_size: int = 32) -> dict:
    return {k: (m.run(v, interventions=ivs or [], batch_size=batch_size) if v else {"margins": np.zeros(0), "logits2": np.zeros((0, 2))})
            for k, v in held.items()}


def save_base(art: RunArtifacts, full: dict):
    p = art.dir("check") / "base_margins.npz"
    np.savez(p, **{k: v["margins"] for k, v in full.items()})
    return p


def load_base(art: RunArtifacts) -> dict:
    p = art.dir("check") / "base_margins.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}: run scripts/check.py first")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def record(art: RunArtifacts, method: str, full: dict, base: dict, ev: dict, arm: str, extra: dict | None = None) -> dict:
    r = {k: v["margins"] for k, v in full.items()}
    met = paper_metrics(r, base, ev, arm)
    art.predictions(method, full, C.meta_rows(ev))
    write_json(art.dir(method) / "metrics.json", {"method": method, "arm": art.arm, "seed": art.seed, "metrics": met,
                                                  "written_rome": now_rome(), **(extra or {})})
    return met


def readout_features(m, texts, batch_size: int = 32) -> np.ndarray:
    key = ("readout", "last")
    return m.run(texts, capture=[key], batch_size=batch_size)["captured"][key]


def cached_val_features(m, art: RunArtifacts, out: Path, seed: int, pool_size: int, batch_size: int = 32):
    """Readout features of every validation row outside SHAVE's blind pool (shared by LEACE and DFR)."""
    p = art.dir("features") / "readout_val_outside_pool.npz"
    rows = outside_pool(out, seed, pool_size)
    ids = np.array([r["id"] for r in rows])
    if p.exists():
        z = np.load(p)
        if np.array_equal(z["ids"], ids):
            return rows, z["X"]
    X = readout_features(m, [r["text"] for r in rows], batch_size)
    np.savez(p, ids=ids, X=X.astype(np.float32))
    return rows, X


def done_marker(art: RunArtifacts, method: str) -> Path:
    return art.dir(method) / "DONE.json"


def mark_done(art: RunArtifacts, method: str, info: dict | None = None):
    write_json(done_marker(art, method), {"finished_rome": now_rome(), **(info or {})})


def is_done(out: Path, arm: str, seed: int, method: str) -> bool:
    return (results_root(out) / arm / f"s{seed}" / method / "DONE.json").exists()
