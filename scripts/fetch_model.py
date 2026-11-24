#!/usr/bin/env python3
"""Manual helper (run_full_rerun.sh uses scripts/inputs.py model, which first reuses a verified copy in the cache).
Download Qwen/Qwen3.5-0.8B at the pinned revision into a plain local directory and verify every file against
configs/model_expected_sha256.txt. Uses HF_TOKEN from the environment if set (never printed).
Usage: fetch_model.py --config C --dest D"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot import reuse as U
from interp_pilot.config import load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    M = cfg["model"]
    expected = U.read_sha_list(_bootstrap.ROOT / M["expected_sha256_file"])
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(repo_id=M["hf_id"], revision=M["revision"], local_dir=str(a.dest)))
    bad = U.check_model_dir(p, expected, M["required_files"])
    if bad:
        print("MODEL HASH MISMATCH:", bad, file=sys.stderr)
        return 3
    print("model ready and verified at", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
