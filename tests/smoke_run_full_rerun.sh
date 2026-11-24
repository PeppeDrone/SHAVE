#!/usr/bin/env bash
# CPU smoke of run_full_rerun.sh, end to end, on a simulated network filesystem (tests only; not collected by pytest).
# A folder plays the mount: an LD_PRELOAD shim (tests/fixtures/nochmod_shim.c) makes chmod/chown/utime/link
# fail there with EPERM. Synthetic data and a fake model stand in for CivilComments and Qwen (their sha256
# go into a derived smoke config). Three runs:
#   1. fresh environment disk: venv installed (needs network for pip), data and model reused from an earlier run folder;
#   2. new environment disk: venv restored from run 1's venv.tar.gz; data downloaded once (file:// URL) into the cache;
#      model reused read-only from an HF-cache-style snapshot;
#   3. MEMTEST=1: existing venv reused as is; data reused from the cache.
# Usage: bash tests/smoke_run_full_rerun.sh [work_dir]   (BASE_PYTHON must be a Python 3.12)
set -euo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${1:-/tmp/ip_smoke}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3.12 || command -v python3)}"
rm -rf "$W"; mkdir -p "$W/container" "$W/vol" "$W/src"
VOL="$W/vol"; RUNS="$VOL/runs"; CACHE="$VOL/cache"
SEED="$RUNS/prior_run"
cc -shared -fPIC -O1 -o "$W/nochmod_shim.so" "$REL/tests/fixtures/nochmod_shim.c" -ldl
# the repository copy also sits on the simulated mount
mkdir -p "$VOL/interp_pilot_release"
( cd "$REL" && find . -mindepth 1 -maxdepth 1 ! -name .pytest_cache -printf '%P\0' | tar --exclude='__pycache__' -cf - --null -T - ) | tar -C "$VOL/interp_pilot_release" -xf -
"$BASE_PYTHON" - "$W" "$REL" <<'PY'
import hashlib, io, json, os, sys, tarfile
from pathlib import Path
W, REL = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(REL / "tests" / "fixtures")]
from synth_csv import make_csv
csv = make_csv(W / "src" / "all_data_with_identities.csv")
data = csv.read_bytes()
arc = W / "src" / "civilcomments_v1.0.tar.gz"
with tarfile.open(arc, "w:gz") as tf:          # a './' entry first, like the CodaLab archive
    d = tarfile.TarInfo("./"); d.type = tarfile.DIRTYPE; d.mode = 0o2775; tf.addfile(d)
    for n, b in (("./all_data_with_identities.csv", data), ("./RELEASE_v1.0.txt", b"synthetic\n")):
        ti = tarfile.TarInfo(n); ti.size = len(b); ti.mode = 0o664; ti.uid = 7357; tf.addfile(ti, io.BytesIO(b))
files = {"config.json": b"{}", "model.safetensors.index.json": b"{}", "model.safetensors-00001-of-00001.safetensors": b"W" * 4096,
         "tokenizer.json": b"{\"tok\":1}", "tokenizer_config.json": b"{}", "README.md": b"fake"}
fm = W / "src" / "fake_model"; fm.mkdir(parents=True, exist_ok=True)
for n, b in files.items():
    (fm / n).write_bytes(b)
sha = lambda b: hashlib.sha256(b).hexdigest()
(W / "src" / "model_sha.txt").write_text("".join(f"{sha(b)}  {n}\n" for n, b in files.items()))
json.dump({"csv_sha": sha(data), "arc_sha": sha(arc.read_bytes()), "w": sha(files["model.safetensors-00001-of-00001.safetensors"]),
           "t": sha(files["tokenizer.json"])}, open(W / "src" / "hashes.json", "w"))
PY
H() { "$BASE_PYTHON" -c "import json; print(json.load(open('$W/src/hashes.json'))['$1'])"; }
derived() {  # <file> <out> <archive_url>
  cat > "$1" <<YAML
inherit: $2/code/configs/smoke_cpu.yaml
data: {archive_url: "$3", archive_sha256: $(H arc_sha), csv_sha256: $(H csv_sha)}
model: {expected_sha256_file: $W/src/model_sha.txt, weights_sha256: $(H w), tokenizer_sha256: $(H t)}
YAML
}
# memtest-like seed run folder with data_src and model (no venv_key.txt)
mkdir -p "$SEED/data_src" "$SEED/model/Qwen3.5-0.8B"
cp "$W/src/all_data_with_identities.csv" "$W/src/civilcomments_v1.0.tar.gz" "$SEED/data_src/"
cp "$W/src/fake_model/"* "$SEED/model/Qwen3.5-0.8B/"
COMMON=(env LD_PRELOAD="$W/nochmod_shim.so" NOCHMOD_PREFIX="$VOL" BASE_PYTHON="$BASE_PYTHON" TORCH_FLAVOUR=cpu
        IP_CACHE="$CACHE" RUNS_ROOT="$RUNS" SEED_RUNS="$SEED" EXTRA_DATA_DIRS= OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 MPLBACKEND=module://matplotlib_inline.backend_inline)
RUN="$VOL/interp_pilot_release/run_full_rerun.sh"
echo "### run 1 $(date '+%H:%M:%S')"
O1="$RUNS/smoke1"; derived "$W/cfg1.yaml" "$O1" "file:///nonexistent/never_downloaded.tgz"
"${COMMON[@]}" VENV_DIR="$W/container/env1/venv" HF_CACHE_SRC= OUT="$O1" CONFIG="$W/cfg1.yaml" bash "$RUN"
echo "### run 2 $(date '+%H:%M:%S')"
rm -rf "$SEED/data_src" "$SEED/model"              # force: data download into the cache, model from the HF cache
HF="$VOL/hf_cache/models--Qwen--Qwen3.5-0.8B"; REV=2fc06364715b967f1860aea9cf38778875588b17
mkdir -p "$HF/blobs" "$HF/snapshots/$REV" "$HF/refs"
for f in "$W/src/fake_model/"*; do s=$(sha256sum "$f" | cut -d' ' -f1); cp "$f" "$HF/blobs/$s"; ln -s "../../blobs/$s" "$HF/snapshots/$REV/$(basename "$f")"; done
echo "$REV" > "$HF/refs/main"; chmod -R a-w "$HF"
( cd "$HF" && find . -printf '%p %y %m %T@\n' | sort ) > "$W/hf_before.txt"
O2="$RUNS/smoke2"; derived "$W/cfg2.yaml" "$O2" "file://$W/src/civilcomments_v1.0.tar.gz"
"${COMMON[@]}" VENV_DIR="$W/container/env2/venv" HF_CACHE_SRC="$HF" OUT="$O2" CONFIG="$W/cfg2.yaml" bash "$RUN"
( cd "$HF" && find . -printf '%p %y %m %T@\n' | sort ) > "$W/hf_after.txt"
echo "### run 3 (MEMTEST=1) $(date '+%H:%M:%S')"
O3="$RUNS/smoke3"; derived "$W/cfg3.yaml" "$O3" "file:///nonexistent/never_downloaded.tgz"
"${COMMON[@]}" VENV_DIR="$W/container/env2/venv" HF_CACHE_SRC="$HF" OUT="$O3" CONFIG="$W/cfg3.yaml" MEMTEST=1 bash "$RUN"
chmod -R u+w "$HF"
echo "### checks $(date '+%H:%M:%S')"
"$BASE_PYTHON" - "$W" "$O1" "$O2" "$O3" <<'PY'
import json, os, sys
from pathlib import Path
W, O1, O2, O3 = map(Path, sys.argv[1:])
def j(p): return json.loads(Path(p).read_text())
def log(o): return (o / "logs" / "run_full_rerun.log").read_text()
checks = {
 "run1 venv installed": "Successfully" in log(O1) or "env ok" in log(O1),
 "run1 data reused from memtest-like run": j(O1/"env/inputs_data.json")["how"] == "reused" and "prior_run" in j(O1/"env/inputs_data.json")["csv"],
 "run1 model reused from memtest-like run": j(O1/"env/inputs_model.json")["how"] == "reused" and "prior_run" in j(O1/"env/inputs_model.json")["path"],
 "run2 venv restored from run1 tarball": "venv: restoring " + str(O1 / "env" / "venv.tar.gz") in log(O2),
 "run2 data downloaded once into the cache": j(O2/"env/inputs_data.json")["how"] == "downloaded" and "/cache/civilcomments/" in j(O2/"env/inputs_data.json")["csv"],
 "run2 model reused from HF cache snapshot": j(O2/"env/inputs_model.json")["path"].endswith("/snapshots/2fc06364715b967f1860aea9cf38778875588b17"),
 "HF cache unchanged": (W/"hf_before.txt").read_text() == (W/"hf_after.txt").read_text(),
 "run3 existing venv reused": "matches key" in log(O3),
 "run3 data reused from cache": j(O3/"env/inputs_data.json")["how"] == "reused" and "/cache/" in j(O3/"env/inputs_data.json")["csv"],
 "run3 memory report": (O3/"logs/memory_report.md").exists(),
}
for o in (O1, O2, O3):
    links = [str(p) for p in o.rglob("*") if p.is_symlink()]
    checks[f"{o.name}: no symlink in run folder"] = not links
    checks[f"{o.name}: no venv folder in run folder"] = not (o / "env" / "venv").exists()
    checks[f"{o.name}: venv.tar.gz + key"] = (o/"env/venv.tar.gz").stat().st_size > 0 and (o/"env/venv_key.txt").exists()
    checks[f"{o.name}: SHA256SUMS verified"] = "verified" in log(o) and (o/"SHA256SUMS").stat().st_size > 0
    checks[f"{o.name}: no pycache in run folder"] = not list(o.rglob("__pycache__"))
    checks[f"{o.name}: no 'Operation not permitted'"] = "Operation not permitted" not in log(o)
    checks[f"{o.name}: pytest passed"] = " passed" in (o/"logs/pytest.txt").read_text() and " failed" not in (o/"logs/pytest.txt").read_text()
for o in (O1, O2):
    checks[f"{o.name}: figures"] = (o / "figures").is_dir() and any((o / "figures").iterdir())
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
PY
echo "### smoke done $(date '+%H:%M:%S')"
