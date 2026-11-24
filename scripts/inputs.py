#!/usr/bin/env python3
"""Find inputs already in the cache or in earlier runs (by sha256) before downloading; write the run's SHA256SUMS.

Subcommands (all used by run_full_rerun.sh):
  venv-key    --requirements R --flavour F                      print the environment key
  find-venv   --key K --runs DIR...                             print an earlier run's env/venv.tar.gz with that key
  freeze-check --requirements R --freeze FILE --flavour F       exit 1 if the installed packages differ from the pins
  data        --config C --cache DIR [--search DIR...] --record FILE     print the CSV path (downloads once if needed)
  model       --config C --cache DIR [--search DIR...] --record FILE     print the model folder (downloads once if needed)
  sha256sums  --out OUT                                         write and verify OUT/SHA256SUMS; always exits 0
venv-key, find-venv, freeze-check and sha256sums need only the standard library; data and model need PyYAML, and
model downloads need huggingface_hub (both in the venv)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot import reuse as U

def _cfg(path):
    from interp_pilot.config import load
    return load(path)


def cmd_data(a):
    D = _cfg(a.config)["data"]
    csv_name, archive_name = D["csv_name"], "civilcomments_v1.0.tar.gz"
    cache = Path(a.cache) / "civilcomments"
    cands = list(dict.fromkeys([str(cache)] + [s for s in a.search if s]))   # order kept, duplicates dropped
    t0 = time.time()
    res = U.find_data(cands, csv_name, D["csv_sha256"], archive_name, D["archive_sha256"])
    how = "reused"
    if res["csv"] is None:
        how = "downloaded"
        cache.mkdir(parents=True, exist_ok=True)
        arch = U.find_archive(cands, archive_name, D["archive_sha256"])
        if arch is None:
            arch = str(cache / archive_name)
            print(f"data: no verified copy found; downloading once into {cache}", file=sys.stderr, flush=True)
            subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", arch, D["archive_url"]], check=True)
            if U.sha256_file(arch) != D["archive_sha256"]:
                os.remove(arch)
                print("archive sha256 mismatch; deleted the download", file=sys.stderr)
                return 2
        # extract into the cache only (never into another run's folder or the HF cache)
        U.extract_members(arch, cache, [csv_name, "RELEASE_v1.0.txt"])
        res = U.find_data([str(cache)], csv_name, D["csv_sha256"], archive_name, D["archive_sha256"])
        if res["csv"] is None:
            print("CSV sha256 mismatch after extraction", file=sys.stderr)
            return 3
        if res["archive"] is None:
            res["archive"] = arch
    rec = {**res, "how": how, "csv_sha256": D["csv_sha256"], "archive_sha256": D["archive_sha256"],
           "searched": cands, "seconds": round(time.time() - t0, 1)}
    if a.record:
        U.write_json(a.record, rec)
    print(f"data {how}: {res['csv']}", file=sys.stderr)
    print(json.dumps({"csv": res["csv"], "archive": res["archive"]}))
    return 0


def cmd_model(a):
    cfg = _cfg(a.config)
    M = cfg["model"]
    expected = U.read_sha_list(_bootstrap.ROOT / M["expected_sha256_file"])
    for k, v in ((M["weights_file"], M["weights_sha256"]), ("tokenizer.json", M["tokenizer_sha256"])):
        if expected.get(k) != v:
            print(f"config error: {k} sha256 differs between experiment.yaml and {M['expected_sha256_file']}", file=sys.stderr)
            return 4
    required = list(M["required_files"])
    name = M["hf_id"].split("/")[-1]
    dest = Path(a.cache) / "models" / f"{name}@{M['revision']}"
    cands = [str(dest)] + [s for s in a.search if s]
    if a.hf_cache:
        cands.append(str(U.hf_snapshot(a.hf_cache, M["revision"])))
    cands = list(dict.fromkeys(cands))   # order kept, duplicates dropped
    t0 = time.time()
    path, tried = U.find_model(cands, expected, required)
    how = "reused"
    if path is None:
        how = "downloaded"
        print(f"model: no verified copy found; downloading {M['hf_id']}@{M['revision']} once into {dest}", file=sys.stderr, flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=M["hf_id"], revision=M["revision"], local_dir=str(dest))
        probs = U.check_model_dir(dest, expected, required)
        if probs:
            print("MODEL HASH MISMATCH:", probs, file=sys.stderr)
            return 3
        path = str(dest)
    files = {r: s for r, s in expected.items() if (Path(path) / r).exists()}
    rec = {"path": path, "how": how, "hf_id": M["hf_id"], "revision": M["revision"], "files_present_and_verified": files,
           "searched": {c: tried.get(c, "not checked") for c in cands}, "seconds": round(time.time() - t0, 1)}
    if a.record:
        U.write_json(a.record, rec)
    print(f"model {how}: {path}", file=sys.stderr)
    print(path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("venv-key"); p.add_argument("--requirements", required=True); p.add_argument("--flavour", required=True)
    p = sub.add_parser("find-venv"); p.add_argument("--key", required=True); p.add_argument("--runs", nargs="*", default=[])
    p = sub.add_parser("freeze-check"); p.add_argument("--requirements", required=True); p.add_argument("--freeze", required=True)
    p.add_argument("--flavour", required=True)
    for n in ("data", "model"):
        p = sub.add_parser(n)
        p.add_argument("--config", required=True); p.add_argument("--cache", required=True)
        p.add_argument("--search", nargs="*", default=[]); p.add_argument("--record", default=None)
        if n == "model":
            p.add_argument("--hf-cache", default=None, help="existing HF cache folder models--Org--Name (read only)")
    p = sub.add_parser("sha256sums"); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "venv-key":
        print(U.venv_key(Path(a.requirements).read_text(), a.flavour)); return 0
    if a.cmd == "find-venv":
        print(U.find_venv_tar(a.key, list(dict.fromkeys(a.runs))) or ""); return 0
    if a.cmd == "freeze-check":
        bad = U.freeze_mismatches(Path(a.requirements).read_text(), Path(a.freeze).read_text(), U.TORCH_VERSION, a.flavour)
        for b in bad:
            print("venv differs:", b, file=sys.stderr)
        return 1 if bad else 0
    if a.cmd == "data":
        return cmd_data(a)
    if a.cmd == "model":
        return cmd_model(a)
    if a.cmd == "sha256sums":
        try:
            r = U.write_sha256sums(a.out)
            bad = U.verify_sha256sums(a.out)
            print(f"SHA256SUMS: {r['n_files']} files, {r['n_errors']} unreadable, {len(r['skipped'])} symlinks/venv/other skipped; "
                  + ("verified" if not bad else f"VERIFY PROBLEMS {len(bad)}: {bad[:5]}"))
        except Exception as e:  # noqa: BLE001 - never stop the run here
            print(f"WARNING: checksum step failed: {type(e).__name__}: {e}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
