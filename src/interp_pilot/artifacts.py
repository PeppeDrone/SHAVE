"""Artifact writer: one directory per (arm, seed, method), hashed manifest, per-example
predictions, arrays as .npy (+ .safetensors when available), timings, code snapshot and
environment capture. Pure Python/numpy; torch is optional."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

ROME = ZoneInfo("Europe/Rome")
SCHEMA = "interp_pilot_release.artifacts/1"


def now_rome() -> str:
    return datetime.now(ROME).strftime("%Y-%m-%d %H:%M:%S %Z")


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "detach"):
        return jsonable(o.detach().float().cpu().numpy())
    return o


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2) + "\n")
    os.replace(tmp, path)
    return path


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = [jsonable(r) for r in rows]
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return path


def table_format() -> str:
    try:
        import pyarrow  # noqa: F401
        import pandas  # noqa: F401
        return "parquet"
    except Exception:
        return "jsonl.gz"


def write_table(path_stem: Path, columns: dict[str, Sequence]) -> Path:
    """Write a column table as parquet if pyarrow is available, else jsonl.gz."""
    path_stem = Path(path_stem); path_stem.parent.mkdir(parents=True, exist_ok=True)
    n = {len(v) for v in columns.values()}
    if len(n) > 1:
        raise ValueError(f"ragged columns {[(k, len(v)) for k, v in columns.items()]}")
    if table_format() == "parquet":
        import pandas as pd
        p = path_stem.with_suffix(".parquet")
        pd.DataFrame({k: list(v) if not isinstance(v, np.ndarray) else v for k, v in columns.items()}).to_parquet(p, index=False)
        return p
    p = path_stem.with_suffix(".jsonl.gz")
    keys = list(columns)
    with gzip.open(p, "wt") as f:
        for i in range(next(iter(n)) if n else 0):
            f.write(json.dumps({k: jsonable(columns[k][i]) for k in keys}) + "\n")
    return p


def read_table(path: Path) -> dict[str, list]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return {c: df[c].tolist() for c in df.columns}
    cols: dict[str, list] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            for k, v in r.items():
                cols.setdefault(k, []).append(v)
    return cols


def save_array(path_stem: Path, arr, name: str = "data", meta: dict | None = None) -> list[Path]:
    """Save as .npy, and also .safetensors if the safetensors package is present."""
    path_stem = Path(path_stem); path_stem.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(np.asarray(jsonable(arr) if hasattr(arr, "detach") else arr, dtype=np.float32))
    out = [path_stem.with_suffix(".npy")]
    np.save(out[0], a)
    try:
        from safetensors.numpy import save_file
        p = path_stem.with_suffix(".safetensors")
        save_file({name: a}, str(p), metadata={k: str(v) for k, v in (meta or {}).items()})
        out.append(p)
    except Exception:
        pass
    return out


def _cuda():
    """torch.cuda if torch is already imported and a GPU is visible, else None (never imports torch itself)."""
    torch = sys.modules.get("torch")
    try:
        return torch.cuda if torch is not None and torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


class Timer:
    """Wall-clock timings per stage plus peak GPU memory per stage (torch.cuda.max_memory_allocated / _reserved).
    Stages may nest: at every start/stop the peak since the last reset is folded into every open stage, then reset,
    so each stage reports the peak over its own interval. On CPU the memory fields are null."""

    def __init__(self):
        self.rows: list[dict] = []
        self._open: dict[str, tuple[float, str]] = {}
        self._peak: dict[str, list[int]] = {}

    def _fold(self):
        cu = _cuda()
        if cu is None:
            return
        a, r = int(cu.max_memory_allocated()), int(cu.max_memory_reserved())
        for v in self._peak.values():
            v[0] = max(v[0], a); v[1] = max(v[1], r)
        cu.reset_peak_memory_stats()

    def start(self, stage: str):
        self._fold()
        self._open[stage] = (time.time(), now_rome())
        self._peak[stage] = [0, 0]

    def stop(self, stage: str, **extra):
        self._fold()
        t0, s = self._open.pop(stage, (time.time(), now_rome()))
        pk = self._peak.pop(stage, None)
        cu = _cuda()
        mem = {"peak_alloc_bytes": None, "peak_reserved_bytes": None, "peak_alloc_gib": None, "peak_reserved_gib": None,
               "gpu_name": None, "gpu_total_bytes": None}
        if cu is not None and pk is not None:
            props = cu.get_device_properties(cu.current_device())
            mem = {"peak_alloc_bytes": pk[0], "peak_reserved_bytes": pk[1], "peak_alloc_gib": round(pk[0] / 2**30, 3),
                   "peak_reserved_gib": round(pk[1] / 2**30, 3), "gpu_name": props.name, "gpu_total_bytes": int(props.total_memory)}
        self.rows.append({"stage": stage, "start_rome": s, "end_rome": now_rome(), "seconds": round(time.time() - t0, 3), **mem, **extra})

    class _Ctx:
        def __init__(self, t, s):
            self.t, self.s = t, s

        def __enter__(self):
            self.t.start(self.s)

        def __exit__(self, *a):
            self.t.stop(self.s)

    def stage(self, s: str):
        return Timer._Ctx(self, s)


class RunArtifacts:
    """<root>/<arm>/s<seed>/<method>/... plus <arm>/s<seed>/manifest_<stage>.json."""

    def __init__(self, root: Path, arm: str, seed: int, config: dict | None = None):
        self.root = Path(root)
        self.arm, self.seed = arm, int(seed)
        self.run_dir = self.root / arm / f"s{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timer = Timer()
        self.config = config or {}

    def dir(self, method: str) -> Path:
        d = self.run_dir / method
        d.mkdir(parents=True, exist_ok=True)
        return d

    def json(self, method: str, name: str, obj) -> Path:
        return write_json(self.dir(method) / name, obj)

    def csv(self, method: str, name: str, rows, fields=None) -> Path:
        return write_csv(self.dir(method) / name, rows, fields)

    def array(self, method: str, name: str, arr, meta=None) -> list[Path]:
        return save_array(self.dir(method) / name, arr, meta=meta)

    def predictions(self, method: str, sets: dict[str, dict], meta_rows: dict[str, Sequence[dict]] | None = None,
                    name: str = "predictions") -> Path:
        """sets: set_name -> {"margins": (n,), "logits2": (n,2) optional}. meta_rows: set_name -> rows (id, label, groups)."""
        cols: dict[str, list] = {k: [] for k in ("set", "row", "example_id", "label", "identity_any", "groups", "margin", "logit0", "logit1", "pred")}
        for sname, res in sets.items():
            m = np.asarray(res["margins"], dtype=float)
            lg = res.get("logits2")
            lg = np.asarray(lg, dtype=float) if lg is not None and len(lg) == len(m) else np.full((len(m), 2), np.nan)
            rows = (meta_rows or {}).get(sname) or [{}] * len(m)
            for i in range(len(m)):
                r = rows[i] if i < len(rows) else {}
                cols["set"].append(sname); cols["row"].append(i)
                cols["example_id"].append(str(r.get("id", r.get("example_id", ""))))
                cols["label"].append(int(r["label"]) if "label" in r and r["label"] is not None else -1)
                cols["identity_any"].append(int(r.get("identity_any", -1)))
                g = r.get("groups") or {}
                cols["groups"].append(",".join(k for k, v in g.items() if v))
                cols["margin"].append(float(m[i])); cols["logit0"].append(float(lg[i, 0])); cols["logit1"].append(float(lg[i, 1]))
                cols["pred"].append(int(m[i] > 0))
        return write_table(self.dir(method) / name, cols)

    def copy_tree(self, method: str, src: Path, name: str):
        import shutil
        dst = self.dir(method) / name
        if Path(src).exists():
            if dst.exists():
                shutil.rmtree(dst)
            # copyfile only (no copystat): some network filesystems reject utime/chmod
            for f in sorted(Path(src).rglob("*")):
                t = dst / f.relative_to(src)
                if f.is_dir():
                    t.mkdir(parents=True, exist_ok=True)
                elif f.is_file():
                    t.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, t)
        return dst

    def write_manifest(self, extra: dict | None = None, stage: str = "run") -> Path:
        if self.timer.rows:
            self.json("_meta", f"timings_{stage}.json", self.timer.rows)
        files = []
        for p in sorted(self.run_dir.rglob("*")):
            if p.is_file() and not p.name.startswith("manifest") and not p.name.endswith(".tmp"):
                files.append({"path": str(p.relative_to(self.run_dir)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        man = {"schema": SCHEMA, "arm": self.arm, "seed": self.seed, "written_rome": now_rome(),
               "config": self.config, "files": files, **(extra or {})}
        return write_json(self.run_dir / f"manifest_{stage}.json", man)


def verify_manifest(path: Path) -> list[str]:
    """Return a list of problems (missing files or hash mismatches)."""
    path = Path(path); man = json.loads(path.read_text()); bad = []
    for f in man.get("files", []):
        p = path.parent / f["path"]
        if not p.exists():
            bad.append(f"missing {f['path']}")
        elif sha256_file(p) != f["sha256"]:
            bad.append(f"hash {f['path']}")
    return bad


def code_snapshot(root: Path, rel_paths: Iterable[str], out_dir: Path) -> dict:
    """Tar of the listed source files plus SHA256SUMS and git-style blob hashes."""
    root = Path(root); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    tar_p = out_dir / "code_snapshot.tar.gz"
    with tarfile.open(tar_p, "w:gz") as tar:
        for rp in sorted(set(rel_paths)):
            p = root / rp
            if p.is_file():
                tar.add(p, arcname=rp)
                entries.append({"path": rp, "bytes": p.stat().st_size, "sha256": sha256_file(p), "git_blob_sha1": git_blob_sha1(p)})
    (out_dir / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['path']}\n" for e in entries))
    tree = hashlib.sha256("".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    info = {"files": entries, "tree_sha256": tree, "tar": tar_p.name, "tar_sha256": sha256_file(tar_p), "written_rome": now_rome()}
    write_json(out_dir / "code_manifest.json", info)
    return info


def environment(out_dir: Path, python: str | None = None) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    py = python or sys.executable
    info = {"python": sys.version, "executable": py, "platform": platform.platform(), "hostname": platform.node(), "written_rome": now_rome()}
    try:
        fr = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        (out_dir / "pip_freeze.txt").write_text(fr)
    except Exception as exc:
        info["pip_freeze_error"] = repr(exc)
    for mod in ("torch", "transformers", "peft", "numpy", "scipy", "sklearn", "safetensors", "pandas", "pyarrow", "concept_erasure", "datasets"):
        try:
            m = __import__(mod); info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            info[f"{mod}_version"] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        pass
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        (out_dir / "nvidia_smi.txt").write_text(smi)
    except Exception:
        pass
    write_json(out_dir / "environment.json", info)
    return info
