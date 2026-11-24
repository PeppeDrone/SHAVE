"""Put <release>/src on sys.path and set conservative thread defaults. Imported first by every script."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
for k, v in (("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(k, v)
