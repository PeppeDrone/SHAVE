#!/usr/bin/env bash
# Create the pinned Python environment. Usage: setup_env.sh <venv_dir> [cpu|cu128]
# The project itself is not pip-installed: every script and the tests put src/ on sys.path.
# BASE_PYTHON (default python3.12, else python3) must be Python 3.12. torch 2.8.0 is installed from the official
# index (cu128 or cpu). Keep TORCH and PIP in step with src/interp_pilot/reuse.py (a test checks it).
# If the target filesystem refuses chmod (some network mounts), put <venv_dir> on a local disk.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:?venv dir}"; FLAVOUR="${2:-cu128}"
TORCH=2.8.0; PIP=25.3
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3.12 || command -v python3)}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "need python 3.12 (BASE_PYTHON=$BASE_PYTHON)"; exit 2; }
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$VENV")/pip-cache}" PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$(dirname "$VENV")"
[ -x "$VENV/bin/python" ] || "$BASE_PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade "pip==$PIP"
"$VENV/bin/python" -m pip install -q --no-cache-dir "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$FLAVOUR"
"$VENV/bin/python" -m pip install -q --no-cache-dir -r "$HERE/requirements.txt"
"$VENV/bin/python" -c "import torch, transformers, peft, sklearn, concept_erasure; print('env ok', torch.__version__, transformers.__version__, peft.__version__)"
