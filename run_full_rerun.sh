#!/usr/bin/env bash
# End-to-end run of the paper: two arms, the seeds in the config (or $SEEDS), SHAVE, LEACE, DFR and the unedited model.
# Results, a code snapshot, logs and manifests go to $OUT (default: <repo>/runs/<UTC timestamp>).
# Data and the base model are looked up first (sha256-checked, read only) in $IP_CACHE (default: <repo>/cache),
# in earlier run folders and in an existing Hugging Face cache; only if nothing matches are they downloaded,
# once, into $IP_CACHE. A matching $OUT/env/venv.tar.gz from this or an earlier run is restored instead of
# reinstalling. Tar extraction does not restore owner, mode or mtime, so the script also works on network
# filesystems that refuse chmod/chown/utime/hardlinks. HF_TOKEN is read from the environment if set and is never printed.
# Usage: bash run_full_rerun.sh                      (SLOTS=auto: every visible GPU or MIG slice)
#        SLOTS=0,1 bash run_full_rerun.sh            (explicit CUDA_VISIBLE_DEVICES values: GPU indices or MIG UUIDs)
#        MEMTEST=1 bash run_full_rerun.sh            (one model, every stage, one GPU; writes logs/memory_report.{json,md}
#                                                     and stops; the model's results are kept and reused by a full run)
#        SEEDS=42,84,126,168,210,252,294,336,378,420 OUT=./runs/paper bash run_full_rerun.sh
#        CONFIG=configs/experiment.yaml OUT=./runs/myrun bash run_full_rerun.sh   (resume: same OUT)
# Other settings (defaults in brackets): VENV_DIR [<repo>/.venv], IP_CACHE [<repo>/cache],
#   RUNS_ROOT [<repo>/runs], SEED_RUNS [empty: an earlier run folder whose data/model to reuse],
#   HF_CACHE_SRC [empty: an existing Hugging Face cache folder of Qwen3.5-0.8B],
#   EXTRA_DATA_DIRS [empty: folders holding the CivilComments archive], BASE_PYTHON [python3.12], TORCH_FLAVOUR [cu128],
#   SEEDS / ARMS / ONLY [empty: all models in the config] to run a subset.
set -euo pipefail
umask 022
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$SRC/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_REL="${CONFIG:-configs/experiment.yaml}"
SLOTS="${SLOTS:-auto}"
MEMTEST="${MEMTEST:-0}"
FLAVOUR="${TORCH_FLAVOUR:-cu128}"
VENV_DIR="${VENV_DIR:-$SRC/.venv}"
IP_CACHE="${IP_CACHE:-$SRC/cache}"
RUNS_ROOT="${RUNS_ROOT:-$SRC/runs}"
SEED_RUNS="${SEED_RUNS-}"
HF_CACHE_SRC="${HF_CACHE_SRC-}"
EXTRA_DATA_DIRS="${EXTRA_DATA_DIRS-}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3.12 || command -v python3)}"
export BASE_PYTHON
# tar must not restore owner, mode or mtime (safe on ordinary disks and on network mounts that refuse those operations)
export TAR_OPTIONS="--no-same-owner --no-same-permissions -m --no-overwrite-dir"
# headless plotting even when launched from a Jupyter kernel (which sets an inline backend)
export MPLBACKEND=Agg
# fewer fragmentation-driven reservations on 24 GB cards (allocator setting only; no effect on the numbers computed)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# bytecode caches and pip cache next to the venv, not in the run folder
VROOT="$(dirname "$VENV_DIR")"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$VROOT/pycache}" PIP_CACHE_DIR="${PIP_CACHE_DIR:-$VROOT/pip-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$OUT"/{logs,env} "$VROOT"
exec > >(tee -a "$OUT/logs/run_full_rerun.log") 2>&1
echo "== start $(date '+%Y-%m-%d %H:%M:%S %Z')  OUT=$OUT  SLOTS=$SLOTS  VENV_DIR=$VENV_DIR  IP_CACHE=$IP_CACHE"

# 1. code snapshot (the run uses this copy, so the code that produced the results is kept with them)
if [ ! -f "$OUT/env/code_sha256.txt" ]; then
  mkdir -p "$OUT/code"
  # members are listed by name (no "./" entry), so tar never has to chmod/utime the existing $OUT/code folder
  ( cd "$SRC" && find . -mindepth 1 -maxdepth 1 ! -name .pytest_cache ! -name '.venv*' ! -name __pycache__ -printf '%P\0' \
      | tar --exclude='results/*' --exclude='__pycache__' --exclude='*.pyc' -cf - --null -T - ) | tar -C "$OUT/code" -xf -
  ( cd "$OUT/code" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$OUT/env/code_sha256.txt"
fi
CODE="$OUT/code"
case "$CONFIG_REL" in /*) CONFIG="$CONFIG_REL";; *) CONFIG="$CODE/$CONFIG_REL";; esac
INPUTS_SYS() { "$BASE_PYTHON" "$CODE/scripts/inputs.py" "$@"; }   # standard library only

# 2. environment: restore a matching venv.tar.gz from an earlier run, else install
echo "== env $(date '+%H:%M:%S')"
KEY="$(INPUTS_SYS venv-key --requirements "$CODE/requirements.txt" --flavour "$FLAVOUR")"
PY="$VENV_DIR/bin/python"
venv_ok() {
  [ -x "$PY" ] && "$PY" -c "import torch, transformers, peft, sklearn, concept_erasure" >/dev/null 2>&1 \
    && "$PY" -m pip freeze > "$VROOT/.freeze_check.txt" 2>/dev/null \
    && INPUTS_SYS freeze-check --requirements "$CODE/requirements.txt" --freeze "$VROOT/.freeze_check.txt" --flavour "$FLAVOUR"
}
OWN_TAR="$OUT/env/venv.tar.gz"
if [ -f "$VENV_DIR/.ip_venv_key" ] && [ "$(cat "$VENV_DIR/.ip_venv_key")" = "$KEY" ] && venv_ok; then
  echo "venv: existing $VENV_DIR matches key $KEY"
else
  RUN_LIST=("$OUT")
  for r in $SEED_RUNS; do RUN_LIST+=("$r"); done
  for r in "$RUNS_ROOT"/*/; do r="${r%/}"; [ "$r" = "$OUT" ] || RUN_LIST+=("$r"); done
  FOUND="$(INPUTS_SYS find-venv --key "$KEY" --runs "${RUN_LIST[@]}")"
  RESTORED=0
  if [ -n "$FOUND" ]; then
    echo "venv: restoring $FOUND (key $KEY) into $VENV_DIR"
    TMPX="$(mktemp -d "$VROOT/.restore.XXXXXX")"
    if tar -C "$TMPX" -xzf "$FOUND" && [ -d "$TMPX/venv" ]; then
      rm -rf "$VENV_DIR"; mv "$TMPX/venv" "$VENV_DIR"; rm -rf "$TMPX"
      if venv_ok; then RESTORED=1; else echo "venv: restored copy does not match the pins; reinstalling"; fi
    else
      rm -rf "$TMPX"; echo "venv: could not extract $FOUND; reinstalling"
    fi
  fi
  if [ "$RESTORED" = 0 ]; then
    rm -rf "$VENV_DIR"
    bash "$CODE/scripts/setup_env.sh" "$VENV_DIR" "$FLAVOUR"
    venv_ok || { echo "venv check failed after install"; exit 2; }
  fi
  echo "$KEY" > "$VENV_DIR/.ip_venv_key"
fi
"$PY" -m pip freeze > "$OUT/env/pip_freeze.txt"
{ "$PY" -V; uname -a; nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"; nvidia-smi 2>/dev/null || true; } > "$OUT/env/system.txt"
# full copy of the environment in the run folder (no symlink to the venv is made inside $OUT)
if [ ! -s "$OWN_TAR" ] || [ "$(cat "$OUT/env/venv_key.txt" 2>/dev/null)" != "$KEY" ]; then
  if [ -n "${FOUND:-}" ] && [ "$FOUND" != "$OWN_TAR" ] && [ "${RESTORED:-0}" = 1 ]; then
    cp "$FOUND" "$OWN_TAR"; echo "restored_from $FOUND" > "$OUT/env/venv_origin.txt"
  else
    tar -C "$(dirname "$VENV_DIR")" -czf "$OWN_TAR" "$(basename "$VENV_DIR")" --transform "s,^$(basename "$VENV_DIR"),venv,"
    echo "built_at $VENV_DIR" > "$OUT/env/venv_origin.txt"
  fi
  echo "$KEY" > "$OUT/env/venv_key.txt"
fi
echo "venv: $VENV_DIR  (copy in the run folder: $OWN_TAR)" | tee "$OUT/env/venv_location.txt"
SMOKE="$("$PY" -c "import sys; sys.path.insert(0, '$CODE/src'); from interp_pilot.config import load; print(1 if load('$CONFIG').get('smoke') else 0)")"

# 3. data: reuse a verified CSV in the cache or an earlier run (or download once into the cache), rebuild, check hashes
echo "== data $(date '+%H:%M:%S')"
DATA_SEARCH=()
for r in $SEED_RUNS; do DATA_SEARCH+=("$r/data_src"); done
for r in "$RUNS_ROOT"/*/; do r="${r%/}"; [ "$r" = "$OUT" ] || DATA_SEARCH+=("$r/data_src"); done
for d in $EXTRA_DATA_DIRS; do DATA_SEARCH+=("$d"); done
DJSON="$("$PY" "$CODE/scripts/inputs.py" data --config "$CONFIG" --cache "$IP_CACHE" --search "${DATA_SEARCH[@]}" --record "$OUT/env/inputs_data.json")"
CSV="$("$PY" -c "import json,sys; print(json.loads(sys.argv[1])['csv'])" "$DJSON")"
ARCH="$("$PY" -c "import json,sys; print(json.loads(sys.argv[1])['archive'] or '')" "$DJSON")"
BUILD_FLAGS=(--verify); [ "$SMOKE" = 1 ] && BUILD_FLAGS=(--small-sizes)
if [ -f "$OUT/data/VERIFY_EXPECTED.txt" ] && [ "$(cat "$OUT/data/VERIFY_EXPECTED.txt")" = "OK" ] && [ "$SMOKE" = 0 ]; then
  echo "data already built and verified in $OUT/data"
else
  "$PY" "$CODE/scripts/build_data.py" --config "$CONFIG" --out "$OUT" --csv "$CSV" ${ARCH:+--archive "$ARCH"} "${BUILD_FLAGS[@]}"
fi

# 4. model at the pinned revision: reuse a verified folder (read only, not copied) or download once into the cache
echo "== model $(date '+%H:%M:%S')"
MODEL_SEARCH=()
for r in $SEED_RUNS; do MODEL_SEARCH+=("$r/model/Qwen3.5-0.8B"); done
for r in "$RUNS_ROOT"/*/; do r="${r%/}"; [ "$r" = "$OUT" ] || MODEL_SEARCH+=("$r/model/Qwen3.5-0.8B"); done
IP_BASE_MODEL="$("$PY" "$CODE/scripts/inputs.py" model --config "$CONFIG" --cache "$IP_CACHE" --search "${MODEL_SEARCH[@]}" \
                 --hf-cache "$HF_CACHE_SRC" --record "$OUT/env/inputs_model.json")"
export IP_BASE_MODEL

# 5. unit tests before any long job
( cd "$CODE" && OMP_NUM_THREADS=2 "$PY" -m pytest -q -p no:cacheprovider tests ) | tee "$OUT/logs/pytest.txt"

# 6. all jobs (resumable: finished stages have DONE.json and are skipped); MEMTEST=1: one model, then stop
if [ "$MEMTEST" = "1" ]; then
  "$PY" "$CODE/scripts/schedule.py" --config "$CONFIG" --out "$OUT" --slots "$SLOTS" --python "$PY" --memtest \
    || echo "WARNING: memory-test stages failed; see $OUT/logs/schedule_status.json"
  "$PY" "$CODE/scripts/inputs.py" sha256sums --out "$OUT" || echo "WARNING: checksum step failed"
  echo "== memory test done $(date '+%Y-%m-%d %H:%M:%S %Z'); report: $OUT/logs/memory_report.md"
  exit 0
fi
# optional subset: SEEDS="42,84" ARMS="muslim90" ONLY="muslim90:42" (comma lists; default: every arm and seed in the config)
SCHED_SEL=()
[ -n "${SEEDS:-}" ] && SCHED_SEL+=(--seeds "$SEEDS")
[ -n "${ARMS:-}" ] && SCHED_SEL+=(--arms "$ARMS")
[ -n "${ONLY:-}" ] && SCHED_SEL+=(--only "$ONLY")
"$PY" "$CODE/scripts/schedule.py" --config "$CONFIG" --out "$OUT" --slots "$SLOTS" --python "$PY" "${SCHED_SEL[@]}" || echo "WARNING: some stages failed; see $OUT/logs/schedule_status.json"

# 7. tables and figures
"$PY" "$CODE/scripts/aggregate.py" --config "$CONFIG" --out "$OUT"
"$PY" "$CODE/scripts/figures.py" --config "$CONFIG" --out "$OUT"

# 8. SHA-256 manifest of every regular file under OUT (symlinks and the venv folder skipped), then verify it; never fatal
"$PY" "$CODE/scripts/inputs.py" sha256sums --out "$OUT" || echo "WARNING: checksum step failed"
echo "== end $(date '+%Y-%m-%d %H:%M:%S %Z')  results in $OUT"
