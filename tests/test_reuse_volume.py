"""Reuse-before-download, venv key, tar options for filesystems that refuse chmod/utime, and the run checksum step. CPU only, no network."""
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

from interp_pilot import reuse as U

ROOT = Path(__file__).resolve().parents[1]
TAR_OPTS = "--no-same-owner --no-same-permissions -m --no-overwrite-dir"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write(p: Path, b: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b); return p


# ---------------------------------------------------------------- venv key

def test_setup_env_pins_match_venv_key_constants():
    s = (ROOT / "scripts" / "setup_env.sh").read_text()
    assert re.search(r"^TORCH=(\S+); PIP=(\S+)$", s, re.M).groups() == (U.TORCH_VERSION, U.PIP_VERSION)
    assert "-e " not in s.replace("set -euo", "") and "--no-deps -e" not in s   # no editable install of the project


def test_venv_key_changes_with_inputs():
    k = U.venv_key("a==1\n", "cu128")
    assert k == U.venv_key("a==1\n", "cu128") and len(k) == 64
    assert len({k, U.venv_key("a==2\n", "cu128"), U.venv_key("a==1\n", "cpu"), U.venv_key("a==1\n", "cu128", python_mm="3.13")}) == 4


def _legacy_run(root: Path, name: str, req: str, torch_line: str, py: str = "Python 3.12.3") -> Path:
    r = root / name
    write(r / "code" / "requirements.txt", req.encode())
    write(r / "env" / "pip_freeze.txt", f"numpy==2.5.4\n{torch_line}\n".encode())
    write(r / "env" / "system.txt", f"{py}\nLinux x\n".encode())
    write(r / "env" / "venv.tar.gz", b"not empty")
    return r


def test_find_venv_tar_by_key_including_runs_without_key_file(tmp_path):
    req = (ROOT / "requirements.txt").read_text()
    key = U.venv_key(req, "cu128")
    other = _legacy_run(tmp_path, "a_other_flavour", req, "torch==2.8.0+cpu")
    old = _legacy_run(tmp_path, "b_memtest_like", req, "torch==2.8.0+cu128")      # no venv_key.txt: derived
    assert U.run_venv_key(old) == key and U.run_venv_key(other) != key
    new = _legacy_run(tmp_path, "c_new", "x==1\n", "torch==2.8.0+cu128")
    (new / "env" / "venv_key.txt").write_text(key + "\n")                          # key file wins over derivation
    assert U.find_venv_tar(key, [tmp_path / "missing", other, old, new]) == str(old / "env" / "venv.tar.gz")
    assert U.find_venv_tar(key, [other, new]) == str(new / "env" / "venv.tar.gz")
    assert U.find_venv_tar("0" * 64, [other, old, new]) is None
    (old / "env" / "venv.tar.gz").write_bytes(b"")                                  # empty tarball is ignored
    assert U.find_venv_tar(key, [old]) is None


def test_freeze_mismatches():
    req = "# c\nnumpy==2.5.4\nPyYAML==6.0.3  # x\nconcept-erasure==0.2.4\n"
    good = "numpy==2.5.4\npyyaml==6.0.3\nconcept_erasure==0.2.4\ntorch==2.8.0+cu128\nextra==1\n"
    assert U.freeze_mismatches(req, good, "2.8.0", "cu128") == []
    bad = U.freeze_mismatches(req, good.replace("2.5.4", "2.5.3"), "2.8.0", "cpu")
    assert len(bad) == 2 and bad[0].startswith("numpy==2.5.4") and "torch" in bad[1]


# ---------------------------------------------------------------- data

def test_find_data_order_hash_and_archive(tmp_path):
    csv, arc = b"id,text\n1,a\n", b"archive bytes"
    a = tmp_path / "cache"; b = tmp_path / "seed"; c = tmp_path / "old"
    write(a / "all.csv", b"corrupted")                       # wrong hash: skipped
    write(b / "all.csv", csv)                                # first match
    write(c / "all.csv", csv); write(c / "arc.tgz", arc)     # archive only here
    r = U.find_data([a, None, b, c], "all.csv", sha(csv), "arc.tgz", sha(arc))
    assert r == {"csv": str(b / "all.csv"), "archive": str(c / "arc.tgz"), "source_dir": str(b)}
    assert U.find_data([a], "all.csv", sha(csv), "arc.tgz", sha(arc))["csv"] is None
    assert U.find_archive([a, c], "arc.tgz", sha(arc)) == str(c / "arc.tgz")


def _make_archive(path: Path, csv: bytes) -> Path:
    """Like the CodaLab archive: a './' entry first, then the files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        d = tarfile.TarInfo("./"); d.type = tarfile.DIRTYPE; d.mode = 0o2775; tf.addfile(d)
        for name, data in (("./all_data_with_identities.csv", csv), ("./RELEASE_v1.0.txt", b"v1\n"), ("../evil.csv", b"x")):
            ti = tarfile.TarInfo(name); ti.size = len(data); ti.mode = 0o664; ti.mtime = 1; tf.addfile(ti, io.BytesIO(data))
    return path


def _derived_config(tmp_path: Path, url: str, csv: bytes, arc: Path, model_list: Path | None = None, required=None) -> Path:
    cfg = {"inherit": str(ROOT / "configs" / "experiment.yaml"),
           "data": {"archive_url": url, "archive_sha256": sha(arc.read_bytes()), "csv_sha256": sha(csv)}}
    if model_list:
        cfg["model"] = {"expected_sha256_file": str(model_list), "required_files": required,
                        "weights_file": "w.safetensors", "weights_sha256": U.read_sha_list(model_list)["w.safetensors"],
                        "tokenizer_sha256": U.read_sha_list(model_list)["tokenizer.json"]}
    p = tmp_path / "derived.yaml"; p.write_text(json.dumps(cfg))   # JSON is valid YAML
    return p


def _inputs(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "inputs.py"), *map(str, args)], capture_output=True, text=True)


def test_data_cli_reuses_without_download_then_downloads_once(tmp_path):
    csv = b"id,comment_text\n1,hello\n"
    arc = _make_archive(tmp_path / "src" / "civilcomments_v1.0.tar.gz", csv)
    seed = tmp_path / "vol" / "runs" / "memtest" / "data_src"
    write(seed / "all_data_with_identities.csv", csv); shutil.copy(arc, seed / "civilcomments_v1.0.tar.gz")
    cache = tmp_path / "vol" / "cache"
    # 1) present in the search path: used in place, nothing downloaded (the URL is unreachable on purpose)
    cfg = _derived_config(tmp_path, "file:///nonexistent/archive.tgz", csv, arc)
    r = _inputs("data", "--config", cfg, "--cache", cache, "--search", seed, "--record", tmp_path / "rec1.json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"csv": str(seed / "all_data_with_identities.csv"), "archive": str(seed / "civilcomments_v1.0.tar.gz")}
    assert json.loads((tmp_path / "rec1.json").read_text())["how"] == "reused" and not cache.exists()
    # 2) nothing in the search path: downloaded once into the cache (file:// stands in for the CodaLab URL)
    cfg = _derived_config(tmp_path, arc.as_uri(), csv, arc)
    r = _inputs("data", "--config", cfg, "--cache", cache, "--record", tmp_path / "rec2.json")
    assert r.returncode == 0, r.stderr
    d = cache / "civilcomments"
    assert json.loads(r.stdout)["csv"] == str(d / "all_data_with_identities.csv") and (d / "RELEASE_v1.0.txt").exists()
    assert json.loads((tmp_path / "rec2.json").read_text())["how"] == "downloaded"
    assert not (tmp_path / "vol" / "evil.csv").exists() and not (cache / "evil.csv").exists()
    # 3) second call: the cache copy is reused
    cfg = _derived_config(tmp_path, "file:///nonexistent/archive.tgz", csv, arc)
    r = _inputs("data", "--config", cfg, "--cache", cache, "--record", tmp_path / "rec3.json")
    assert r.returncode == 0 and json.loads((tmp_path / "rec3.json").read_text())["how"] == "reused"
    # 4) a corrupt download is rejected and removed
    bad = tmp_path / "bad.tgz"; bad.write_bytes(b"garbage")
    cfg = _derived_config(tmp_path, bad.as_uri(), csv, arc)
    r = _inputs("data", "--config", cfg, "--cache", tmp_path / "cache2")
    assert r.returncode == 2 and not (tmp_path / "cache2" / "civilcomments" / "civilcomments_v1.0.tar.gz").exists()


# ---------------------------------------------------------------- model

FILES = {"config.json": b"{}", "model.safetensors.index.json": b"{\"w\":1}", "w.safetensors": b"W" * 1000,
         "tokenizer.json": b"{\"t\":1}", "tokenizer_config.json": b"{}", "README.md": b"readme"}
REQ = ["config.json", "model.safetensors.index.json", "w.safetensors", "tokenizer.json", "tokenizer_config.json"]


def _model_list(tmp_path):
    p = tmp_path / "model_sha.txt"
    p.write_text("# test\n" + "".join(f"{sha(b)}  {n}\n" for n, b in FILES.items()))
    return p


def _hf_cache(root: Path, rev: str, files: dict) -> Path:
    """HF cache layout: snapshots/<rev>/<file> are symlinks into blobs/<sha>."""
    repo = root / "models--Qwen--Qwen3.5-0.8B"
    for n, b in files.items():
        write(repo / "blobs" / sha(b), b)
        link = repo / "snapshots" / rev / n; link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(repo / "blobs" / sha(b), link.parent))
    write(repo / "refs" / "main", rev.encode())
    return repo


def _tree_state(root: Path):
    return sorted((str(p.relative_to(root)), p.is_symlink(), p.lstat().st_mtime_ns, p.lstat().st_mode) for p in root.rglob("*"))


def test_check_model_dir_rules(tmp_path):
    exp = U.read_sha_list(_model_list(tmp_path))
    d = tmp_path / "m"
    for n, b in FILES.items():
        write(d / n, b)
    assert U.check_model_dir(d, exp, REQ) == []
    (d / "README.md").unlink()                                  # optional file missing: still fine
    assert U.check_model_dir(d, exp, REQ) == []
    (d / "README.md").write_bytes(b"changed")                   # listed file present but different: rejected
    assert U.check_model_dir(d, exp, REQ) == ["sha256 mismatch README.md"]
    (d / "README.md").unlink(); (d / "tokenizer.json").unlink()
    assert U.check_model_dir(d, exp, REQ) == ["missing tokenizer.json"]
    assert U.check_model_dir(tmp_path / "nope", exp, REQ)


def test_model_cli_prefers_cache_then_seed_then_hf_cache_read_only(tmp_path):
    lst = _model_list(tmp_path)
    csv = b"x"; arc = _make_archive(tmp_path / "a.tgz", csv)
    cfg = _derived_config(tmp_path, "file:///x", csv, arc, model_list=lst, required=REQ)
    rev = "2fc06364715b967f1860aea9cf38778875588b17"
    hf = _hf_cache(tmp_path / "hf_cache", rev, FILES)
    for p in sorted(hf.rglob("*"), reverse=True):                # read only, like the existing cache must stay
        if not p.is_symlink():
            p.chmod(p.stat().st_mode & ~0o222)
    before = _tree_state(hf)
    seed = tmp_path / "runs" / "memtest" / "model" / "Qwen3.5-0.8B"
    write(seed / "config.json", b"{}")                            # incomplete seed: rejected
    cache = tmp_path / "cache"
    r = _inputs("model", "--config", cfg, "--cache", cache, "--search", seed, "--hf-cache", hf, "--record", tmp_path / "m.json")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(hf / "snapshots" / rev)
    rec = json.loads((tmp_path / "m.json").read_text())
    assert rec["how"] == "reused" and rec["searched"][str(seed)] and set(rec["files_present_and_verified"]) == set(FILES)
    assert _tree_state(hf) == before and not cache.exists()      # nothing written into the HF cache or the cache
    for n, b in FILES.items():                                    # now a complete seed copy exists: it comes first
        write(seed / n, b)
    r = _inputs("model", "--config", cfg, "--cache", cache, "--search", seed, "--hf-cache", hf)
    assert r.stdout.strip() == str(seed)
    for p in sorted(hf.rglob("*")):
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | 0o200)


def test_model_download_once_into_cache(tmp_path, monkeypatch):
    import inputs as I
    lst = _model_list(tmp_path)
    csv = b"x"; arc = _make_archive(tmp_path / "a.tgz", csv)
    cfg = _derived_config(tmp_path, "file:///x", csv, arc, model_list=lst, required=REQ)
    calls = []

    def fake_snapshot_download(repo_id, revision, local_dir):
        calls.append((repo_id, revision, local_dir))
        for n, b in FILES.items():
            write(Path(local_dir) / n, b)
        return local_dir
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    cache = tmp_path / "cache"
    assert I.main(["model", "--config", str(cfg), "--cache", str(cache), "--hf-cache", str(tmp_path / "no_hf")]) == 0
    assert len(calls) == 1 and calls[0][2].endswith("Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17")
    assert I.main(["model", "--config", str(cfg), "--cache", str(cache)]) == 0 and len(calls) == 1   # reused


# ---------------------------------------------------------------- tar options and a volume that refuses chmod

def test_tar_options_set_for_every_extraction():
    run = (ROOT / "run_full_rerun.sh").read_text()
    assert f'export TAR_OPTIONS="{TAR_OPTS}"' in run
    assert run.index("export TAR_OPTIONS") < run.index("tar -C")          # exported before the first tar call
    assert "MPLBACKEND=Agg" in run and 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in run
    assert "pip install" not in run and "ln -s" not in run                # no editable install, no symlink in OUT
    fd = (ROOT / "scripts" / "fetch_data.sh").read_text()
    assert f"TAR_OPTIONS:-{TAR_OPTS}" in fd and fd.index("TAR_OPTIONS") < fd.index("tar -x")


def test_extract_members_streams_only_named_files(tmp_path):
    arc = _make_archive(tmp_path / "a.tgz", b"csv")
    out = U.extract_members(arc, tmp_path / "d", ["all_data_with_identities.csv", "RELEASE_v1.0.txt", "evil.csv"])
    assert sorted(Path(p).name for p in out) == ["RELEASE_v1.0.txt", "all_data_with_identities.csv"]
    assert (tmp_path / "d" / "all_data_with_identities.csv").read_bytes() == b"csv"


@pytest.fixture(scope="module")
def shim(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler for the volume simulator")
    so = tmp_path_factory.mktemp("shim") / "nochmod_shim.so"
    r = subprocess.run([cc, "-shared", "-fPIC", "-O1", "-o", str(so), str(ROOT / "tests" / "fixtures" / "nochmod_shim.c"), "-ldl"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("shim does not build: " + r.stderr[-300:])
    return so


def _vol_env(shim, vol, **extra):
    env = {k: v for k, v in os.environ.items() if k != "TAR_OPTIONS"}
    return {**env, "LD_PRELOAD": str(shim), "NOCHMOD_PREFIX": str(vol), **extra}


def test_volume_simulator_refuses_chmod_and_utime(shim, tmp_path):
    vol = tmp_path / "vol"; vol.mkdir(); (vol / "f").write_text("x")
    code = "import os,sys; p=sys.argv[1]\nfor f in (lambda: os.chmod(p, 0o600), lambda: os.utime(p, (1, 1))):\n" \
           "    try: f(); print('allowed')\n    except PermissionError: print('EPERM')"
    r = subprocess.run([sys.executable, "-c", code, str(vol / "f")], env=_vol_env(shim, vol), capture_output=True, text=True)
    assert r.stdout.split() == ["EPERM", "EPERM"], r.stderr


def test_tar_needs_the_options_on_the_volume(shim, tmp_path):
    src = tmp_path / "src"; write(src / "sub" / "a.txt", b"a"); os.utime(src / "sub" / "a.txt", (1, 1))
    arc = tmp_path / "x.tar"
    subprocess.run(["tar", "-C", str(src), "-cf", str(arc), "sub"], check=True)
    vol = tmp_path / "vol"
    vol.mkdir()
    r = subprocess.run(["tar", "-C", str(vol), "-xf", str(arc)], env=_vol_env(shim, vol), capture_output=True, text=True)
    assert r.returncode != 0 and "Operation not permitted" in r.stderr           # the filesystem rejected the operation
    shutil.rmtree(vol); vol.mkdir()
    r = subprocess.run(["tar", "-C", str(vol), "-xf", str(arc)], env=_vol_env(shim, vol, TAR_OPTIONS=TAR_OPTS), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (vol / "sub" / "a.txt").read_bytes() == b"a"


def test_code_snapshot_and_data_extraction_work_on_the_volume(shim, tmp_path):
    vol = tmp_path / "vol"; (vol / "code").mkdir(parents=True)
    snap = re.search(r'\( cd "\$SRC" && (find .*?)\) \| tar -C "\$OUT/code" -xf -', (ROOT / "run_full_rerun.sh").read_text(), re.S)
    cmd = f'( cd "$SRC" && {snap.group(1)}) | tar -C "$OUT/code" -xf -'.replace("\\\n", " ")
    r = subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], capture_output=True, text=True,
                       env=_vol_env(shim, vol, SRC=str(ROOT), OUT=str(vol), TAR_OPTIONS=TAR_OPTS))
    assert r.returncode == 0, r.stderr
    assert (vol / "code" / "run_full_rerun.sh").exists() and (vol / "code" / "src" / "interp_pilot" / "reuse.py").exists()
    assert not list((vol / "code").rglob("__pycache__")) and not (vol / "code" / ".pytest_cache").exists()
    arc = _make_archive(tmp_path / "a.tgz", b"csv")
    code = "import sys; from interp_pilot import reuse as U; print(len(U.extract_members(sys.argv[1], sys.argv[2], ['all_data_with_identities.csv'])))"
    r = subprocess.run([sys.executable, "-c", code, str(arc), str(vol / "cache")], capture_output=True, text=True,
                       env=_vol_env(shim, vol, PYTHONPATH=str(ROOT / "src")))
    assert r.returncode == 0 and r.stdout.strip() == "1", r.stderr


# ---------------------------------------------------------------- checksums

def test_sha256sums_skip_symlinks_and_venv_and_never_crash(tmp_path):
    out = tmp_path / "run"
    write(out / "results" / "a.json", b"a"); write(out / "env" / "venv.tar.gz", b"tgz")
    write(out / "env" / "venv" / "bin" / "python", b"py")                 # a real venv folder: excluded
    write(out / "logs" / "run_full_rerun.log", b"still growing")          # excluded (written while hashing)
    (out / "env" / "venv_link").symlink_to(tmp_path)                      # symlink to a folder (the pod's env/venv case)
    (out / "results" / "link.json").symlink_to(out / "results" / "a.json")
    (out / "results" / "dangling").symlink_to(tmp_path / "gone")
    os.mkfifo(out / "results" / "fifo")
    unread = write(out / "results" / "unreadable.bin", b"u"); unread.chmod(0)
    r = U.write_sha256sums(out)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    names = [l.split("  ", 1)[1] for l in lines]
    expect = ["./env/venv.tar.gz", "./results/a.json"] + (["./results/unreadable.bin"] if os.access(unread, os.R_OK) else [])
    assert names == expect
    assert "env/venv/" in r["skipped"] and "env/venv_link/" in r["skipped"] and "results/fifo" in r["skipped"]
    assert lines[1].split()[0] == sha(b"a")
    if not os.access(unread, os.R_OK):
        assert r["n_errors"] == 1 and "unreadable.bin" in (out / "logs" / "sha256sums_errors.txt").read_text()
    unread.chmod(0o644)
    assert U.verify_sha256sums(out) == []
    (out / "results" / "a.json").write_bytes(b"changed")
    assert U.verify_sha256sums(out) == ["mismatch ./results/a.json"]
    # the CLI always exits 0, even for a folder that does not exist
    r = _inputs("sha256sums", "--out", tmp_path / "does_not_exist")
    assert r.returncode == 0 and ("SHA256SUMS" in r.stdout or "WARNING" in r.stdout)
