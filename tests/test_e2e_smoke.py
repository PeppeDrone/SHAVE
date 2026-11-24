"""End-to-end CPU smoke: synthetic data -> scheduler (all stages, one arm) -> aggregate -> figures. Plumbing only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from synth_csv import make_csv

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, env=None):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_cpu_smoke_pipeline(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    stages = "train,check,shave,leace,dfr"
    csv = make_csv(tmp_path / "src" / "all.csv")
    cfg = ROOT / "configs" / "smoke_cpu.yaml"
    _run(ROOT / "scripts/build_data.py", "--config", cfg, "--out", tmp_path, "--csv", csv, "--small-sizes", env=env)
    _run(ROOT / "scripts/schedule.py", "--config", cfg, "--out", tmp_path, "--slots", "cpu", "--only", "muslim90:42",
         "--stages", stages, env=env)
    d = tmp_path / "results" / "muslim90" / "s42"
    for meth in ["base", "shave", "shave_no_threshold", "leace_oracle", "dfr"]:
        m = json.loads((d / meth / "metrics.json").read_text())
        assert "muslim_ba" in m["metrics"] and (d / meth / "predictions.parquet").exists(), meth
    assert (d / "shave" / "candidate_sweep.csv").exists() and (d / "train" / "adapter").exists()
    tr = json.loads((d / "train" / "training.json").read_text())       # one pass over the arm, each row once
    n = tr["n_train_rows"]
    assert tr["epochs"] == 1 and tr["rows_seen"] == n and tr["n_steps"] == -(-n // tr["batch_size"]) and "steps" not in tr
    tt = json.loads((d / "_meta" / "timings_train.json").read_text())
    assert all("peak_alloc_bytes" in r and "peak_reserved_bytes" in r for r in tt)
    assert {"train_total", "train_loop", "train_save_eval"} <= {r["stage"] for r in tt}   # memory freed between loop and save/eval
    assert (tmp_path / "logs" / "memory_report.json").exists()
    assert json.loads((d / "leace_oracle" / "metrics.json").read_text())["fit_rows"] == "arm training rows"
    s = json.loads((d / "shave" / "summary.json").read_text())
    assert s["target"]["Vstar"] == ["present"] if "target" in s else True
    g = s["strength_gate"]                                              # strength threshold is recorded and applied to SHAVE
    assert g["min_wilson_lb"] == 0.20
    mo = json.loads((d / "shave" / "metrics.json").read_text())
    nt = json.loads((d / "shave_no_threshold" / "metrics.json").read_text())
    if g["decision"] == "abstain":
        assert mo["edited"] is False and mo["reason"] == "weak_target" and s["shave"]["decision"] == "abstain"
        # the paper applied the threshold after the candidate search, so the search still ran
        if s.get("value_scan", {}).get("fstar", "").startswith("cc_w:"):
            assert s["candidate_search_run"] is True
    elif g["decision"] == "search":
        assert s["candidate_search_run"] is True
        assert mo["edited"] == nt["edited"]
    # resumable: a second run skips every finished stage
    out = _run(ROOT / "scripts/schedule.py", "--config", cfg, "--out", tmp_path, "--slots", "cpu", "--only", "muslim90:42",
               "--stages", stages, env=env)
    assert "stage runs 0" in out
    _run(ROOT / "scripts/aggregate.py", "--config", cfg, "--out", tmp_path, env=env)
    assert (tmp_path / "aggregate" / "REPORT_aggregate.md").exists()
    inc = json.loads((d / "check" / "inclusion.json").read_text())                 # written by the check stage
    assert inc["status"] == "included" and inc["rule"]["shortcut_arms"] == ["muslim90"]
    ms = (tmp_path / "aggregate" / "model_status.csv").read_text().splitlines()
    assert len(ms) == 3 and "muslim90,42,included" in ms[2] and "muslim_nat,42,missing" in ms[1]   # every seed listed
    assert (tmp_path / "aggregate" / "table_model_status.tex").exists()
    _run(ROOT / "scripts/figures.py", "--config", cfg, "--out", tmp_path, env=env)
    for name in ("fig1_wilson_lb.pdf", "fig2_delta_ba.pdf", "fig3_sens_spec.pdf"):
        assert (tmp_path / "figures" / name).exists(), name
    for man in d.glob("manifest_*.json"):
        from interp_pilot.artifacts import verify_manifest
        assert verify_manifest(man) == [], man


def test_cpu_smoke_memtest(tmp_path):
    """MEMTEST path on CPU with the smoke config: one model, every stage, memory report with one row per stage."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    csv = make_csv(tmp_path / "src" / "all.csv")
    cfg = ROOT / "configs" / "smoke_cpu.yaml"
    _run(ROOT / "scripts/build_data.py", "--config", cfg, "--out", tmp_path, "--csv", csv, "--small-sizes", env=env)
    out = _run(ROOT / "scripts/schedule.py", "--config", cfg, "--out", tmp_path, "--slots", "cpu", "--memtest", env=env)
    assert "JOBS 1" in out and "STAGES ['train', 'check', 'shave', 'leace', 'dfr']" in out
    done = {p.parent.name for p in (tmp_path / "results").rglob("DONE.json")}
    assert {"train", "check", "shave", "shave_no_threshold", "leace_oracle", "dfr"} <= done <= \
        {"train", "check", "shave", "shave_no_threshold", "leace_oracle", "dfr", "base"}, done
    rep = (tmp_path / "logs" / "memory_report.md").read_text()
    for st in ("train", "check", "shave", "leace", "dfr"):
        assert st in rep, st
