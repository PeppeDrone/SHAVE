#!/usr/bin/env python3
"""Build the data into <out>/data from the WILDS CSV and (optionally) verify it against the expected hashes."""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot import build as B
from interp_pilot.config import load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_bootstrap.ROOT / "configs" / "experiment.yaml")
    ap.add_argument("--out", type=Path, required=True, help="run root; data goes to <out>/data")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--verify", action="store_true", help="compare with configs/data_expected_sha256.txt; exit 3 on mismatch")
    ap.add_argument("--small-sizes", action="store_true", help="tests/smoke only: tiny eval sizes, largest feasible arm size")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    kw = {}
    if a.small_sizes:
        kw = {"n_total": None, "sizes": {"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, "top_k": 12}
    st = B.build(a.csv, a.out / "data", arms=tuple(cfg["arms"]), archive=a.archive, **kw)
    print(json.dumps({"arms": {k: v["muslim_design"] for k, v in st["arms"].items()}, "eval": st["eval"]}, indent=1))
    if a.verify:
        bad = B.verify_expected(a.out / "data", _bootstrap.ROOT / cfg["data"]["expected_sha256_file"])
        (a.out / "data" / "VERIFY_EXPECTED.txt").write_text("OK\n" if not bad else "\n".join(bad) + "\n")
        if bad:
            print("DATA MISMATCH:", bad, file=sys.stderr)
            return 3
        print("data verified against expected sha256 list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
