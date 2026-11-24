"""Reuse inputs that are already in the cache or in earlier runs before downloading anything (standard library only).

Three kinds of input are looked up, always by sha256, never by name alone:
  * the CivilComments-WILDS archive and CSV,
  * the Qwen model folder at the pinned revision,
  * a tarball of the Python environment (venv.tar.gz) built from the same requirements.
A file or folder that is found and matches is used where it is (nothing is copied, nothing is modified). Only when no
candidate matches is the input downloaded, once, into the shared cache folder.

Also here: the run's SHA256SUMS writer, which skips symlinks, non-regular files and the venv, and never raises."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

# Pins of scripts/setup_env.sh (a test checks that both agree). Part of the venv key.
TORCH_VERSION = "2.8.0"
PIP_VERSION = "25.3"
PYTHON_MM = "3.12"
VENV_KEY_RECIPE = "v1"


# ---------------------------------------------------------------- hashing

def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _matches(path: Path, sha: str) -> bool:
    try:
        return Path(path).is_file() and sha256_file(path) == sha
    except OSError:
        return False


def read_sha_list(path) -> dict[str, str]:
    """'<sha256>  <relative path>' lines (comments and blank lines ignored) -> {path: sha}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, rel = line.split(None, 1)
        out[rel.strip().lstrip("*")] = sha
    return out


# ---------------------------------------------------------------- data

def find_data(candidate_dirs, csv_name: str, csv_sha: str, archive_name: str, archive_sha: str) -> dict:
    """First folder (in the given order) whose CSV matches csv_sha. The archive is taken from the same folder if it
    matches, else from any other candidate folder where it matches, else None (the CSV is what the data build reads;
    the archive hash is only recorded). Returns {"csv", "archive", "source_dir"} with None values if no CSV matches."""
    dirs = [Path(d) for d in candidate_dirs if d]
    res = {"csv": None, "archive": None, "source_dir": None}
    for d in dirs:
        if _matches(d / csv_name, csv_sha):
            res.update(csv=str(d / csv_name), source_dir=str(d))
            break
    if res["csv"] is None:
        return res
    first = Path(res["source_dir"])
    for d in [first] + [x for x in dirs if x != first]:
        if _matches(d / archive_name, archive_sha):
            res["archive"] = str(d / archive_name)
            break
    return res


def extract_members(archive, dest, basenames) -> list[str]:
    """Write the archive members whose file name is in `basenames` straight into dest (flat), by streaming their bytes.
    No owner, mode, mtime or hardlink is ever set, so this works on filesystems that refuse chmod/utime.
    The archive's own "./" entry is never touched. Returns the written paths."""
    import tarfile
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    want = set(basenames); written = []
    with tarfile.open(archive, "r:*") as tf:
        for m in tf:
            name = os.path.basename(m.name.rstrip("/"))
            if not m.isfile() or name not in want or ".." in Path(m.name).parts:
                continue
            src = tf.extractfile(m)
            with open(dest / name, "wb") as f:
                for b in iter(lambda: src.read(1 << 22), b""):
                    f.write(b)
            written.append(str(dest / name))
    return written


def find_archive(candidate_dirs, archive_name: str, archive_sha: str) -> str | None:
    for d in candidate_dirs:
        if d and _matches(Path(d) / archive_name, archive_sha):
            return str(Path(d) / archive_name)
    return None


# ---------------------------------------------------------------- model

def check_model_dir(d, expected: dict[str, str], required) -> list[str]:
    """Problems with a model folder: missing required files, hash mismatches of listed files that are present.
    Empty list = usable. Files are only read (an HF-cache snapshot of symlinks into blobs/ is fine)."""
    d = Path(d)
    if not d.is_dir():
        return [f"not a folder: {d}"]
    bad = [f"missing {r}" for r in required if not (d / r).is_file()]
    if bad:
        return bad
    for rel, sha in expected.items():
        p = d / rel
        if p.exists() and not _matches(p, sha):
            bad.append(f"sha256 mismatch {rel}")
    return bad


def hf_snapshot(repo_cache_dir, revision: str) -> Path:
    """Snapshot folder of an existing Hugging Face cache (models--Org--Name) at an exact revision."""
    return Path(repo_cache_dir) / "snapshots" / revision


def find_model(candidate_dirs, expected: dict[str, str], required) -> tuple[str | None, dict]:
    """First candidate folder that passes check_model_dir. Returns (path or None, {candidate: problems})."""
    tried = {}
    for d in candidate_dirs:
        if not d:
            continue
        probs = check_model_dir(d, expected, required)
        tried[str(d)] = probs
        if not probs:
            return str(d), tried
    return None, tried


# ---------------------------------------------------------------- venv

def venv_key(requirements_text: str, flavour: str, torch_version: str = TORCH_VERSION, pip_version: str = PIP_VERSION,
             python_mm: str = PYTHON_MM) -> str:
    """Hash that identifies an environment: requirements.txt bytes + torch build + pip + python minor version."""
    s = f"recipe={VENV_KEY_RECIPE}\npython={python_mm}\npip=={pip_version}\ntorch=={torch_version}+{flavour}\n---\n{requirements_text}"
    return hashlib.sha256(s.encode()).hexdigest()


def run_venv_key(run_dir) -> str | None:
    """venv key of an earlier run folder: env/venv_key.txt if written, else derived from that run's
    code/requirements.txt, the torch build in env/pip_freeze.txt and the python version in env/system.txt
    (runs made before venv_key.txt existed, e.g. the 24 GB memory test). None if it cannot be derived."""
    run = Path(run_dir)
    kf = run / "env" / "venv_key.txt"
    try:
        if kf.is_file():
            return kf.read_text().split()[0]
        req = (run / "code" / "requirements.txt").read_text()
        m = re.search(r"^torch==([0-9.]+)\+(\w+)\s*$", (run / "env" / "pip_freeze.txt").read_text(), re.M)
        py = re.search(r"Python (\d+\.\d+)", (run / "env" / "system.txt").read_text())
        if not (m and py):
            return None
        return venv_key(req, m.group(2), torch_version=m.group(1), python_mm=py.group(1))
    except (OSError, IndexError):
        return None


def find_venv_tar(key: str, run_dirs) -> str | None:
    """First run folder whose env/venv.tar.gz was built for the same key."""
    for r in run_dirs:
        t = Path(r) / "env" / "venv.tar.gz"
        try:
            if t.is_file() and t.stat().st_size > 0 and run_venv_key(r) == key:
                return str(t)
        except OSError:
            continue
    return None


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def freeze_mismatches(requirements_text: str, freeze_text: str, torch_version: str, flavour: str) -> list[str]:
    """Pinned requirements (name==version) missing from a pip freeze, plus the torch build. Empty list = matches."""
    have = {}
    for line in freeze_text.splitlines():
        if "==" in line:
            n, v = line.split("==", 1)
            have[_norm(n.strip())] = v.strip()
    bad = []
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        n, v = (x.strip() for x in line.split("==", 1))
        if have.get(_norm(n)) != v:
            bad.append(f"{n}=={v} (have {have.get(_norm(n))})")
    want_t = f"{torch_version}+{flavour}"
    if have.get("torch") != want_t:
        bad.append(f"torch=={want_t} (have {have.get('torch')})")
    return bad


# ---------------------------------------------------------------- run checksums

def write_sha256sums(out, exclude_rel=("SHA256SUMS", "logs/run_full_rerun.log"), exclude_dirs=("env/venv",),
                     errors_path=None) -> dict:
    """Write <out>/SHA256SUMS for every regular file under out. Skipped: symlinks (files or folders, never followed),
    anything that is not a regular file, the listed folders (the venv) and files. Never raises: unreadable files are
    listed in errors_path (default <out>/logs/sha256sums_errors.txt) and the count is returned."""
    out = Path(out)
    ex_files = set(exclude_rel)
    ex_dirs = tuple(d.rstrip("/") for d in exclude_dirs)
    lines, errors, skipped = [], [], []
    try:
        for root, dirs, files in os.walk(out, followlinks=False):
            rel_root = os.path.relpath(root, out)
            rel_root = "" if rel_root == "." else rel_root
            keep = []
            for d in sorted(dirs):
                rd = os.path.join(rel_root, d) if rel_root else d
                if os.path.islink(os.path.join(root, d)) or rd in ex_dirs or any(rd.startswith(x + "/") for x in ex_dirs):
                    skipped.append(rd + "/")
                else:
                    keep.append(d)
            dirs[:] = keep
            for f in files:
                rf = os.path.join(rel_root, f) if rel_root else f
                p = os.path.join(root, f)
                if rf in ex_files:
                    continue
                try:
                    st = os.lstat(p)
                    if not stat.S_ISREG(st.st_mode):
                        skipped.append(rf)
                        continue
                    lines.append((rf, sha256_file(p)))
                except OSError as e:
                    errors.append(f"{rf}: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 - the checksum step must never stop the run
        errors.append(f"walk: {type(e).__name__}: {e}")
    lines.sort()
    res = {"n_files": len(lines), "n_errors": len(errors), "skipped": skipped}
    try:
        (out / "SHA256SUMS").write_text("".join(f"{s}  ./{r}\n" for r, s in lines))
    except Exception as e:  # noqa: BLE001
        errors.append(f"write SHA256SUMS: {type(e).__name__}: {e}")
        res["n_errors"] = len(errors)
    ep = Path(errors_path) if errors_path else out / "logs" / "sha256sums_errors.txt"
    try:
        if errors:
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text("\n".join(errors) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return res


def verify_sha256sums(out) -> list[str]:
    """Re-hash every entry of <out>/SHA256SUMS; returns problems (never raises)."""
    out = Path(out)
    bad = []
    try:
        for line in (out / "SHA256SUMS").read_text().splitlines():
            sha, rel = line.split(None, 1)
            p = out / rel.strip()
            try:
                if sha256_file(p) != sha:
                    bad.append(f"mismatch {rel}")
            except OSError as e:
                bad.append(f"{rel}: {type(e).__name__}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"SHA256SUMS: {type(e).__name__}: {e}")
    return bad


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


if __name__ == "__main__":  # pragma: no cover
    print(__doc__, file=sys.stderr)
