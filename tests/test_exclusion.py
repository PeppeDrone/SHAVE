"""Fixed model-exclusion rule: decided on the unedited model only, the same for every method, every seed reported."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from interp_pilot.config import load
from interp_pilot.exclusion import EXCLUDED, INCLUDED, MISSING, decide, rule_from_config, status_for

ROOT = Path(__file__).resolve().parents[1]
RULE = {"auroc_min": 0.80, "acc_min": 0.80, "shortcut_gap_min": 0.05, "shortcut_arms": ["muslim90"]}


def _met(au=0.92, acc=0.90, ba=0.60, gap=0.70):
    return {"auroc_test": au, "acc_test": acc, "muslim_ba": ba, "gap_muslim": gap}


def test_frozen_rule_in_config():
    c = load(ROOT / "configs" / "experiment.yaml")
    assert rule_from_config(c) == RULE
    assert c["exclusion"]["acc_min"] == c["stats"]["collapse_acc"]          # same value as the descriptive collapse count
    assert c["exclusion"]["auroc_min"] == c["check"]["auroc_min"]           # same value as the check's "learned" bar
    assert c["exclusion"]["shortcut_gap_min"] == c["check"]["gap_go"]       # same value as the check's strong-gap bar
    with pytest.raises(ValueError):
        rule_from_config({"exclusion": {"auroc_min": 0.8}})


def test_included_when_all_conditions_hold():
    for arm in ("muslim90", "muslim_nat"):
        st = decide(_met(), arm, RULE)
        assert st["status"] == INCLUDED and st["failed"] == [] and st["reason"] == "all conditions met"


def test_each_condition_excludes():
    assert decide(_met(au=0.79), "muslim90", RULE)["failed"] == ["learned"]
    assert decide(_met(acc=0.79), "muslim90", RULE)["failed"] == ["not_collapsed"]
    assert decide(_met(ba=None), "muslim90", RULE)["failed"] == ["primary_defined"]
    assert decide(_met(gap=0.049), "muslim90", RULE)["failed"] == ["shortcut_measurable"]
    assert decide(_met(gap=None), "muslim90", RULE)["failed"] == ["shortcut_measurable"]
    st = decide(_met(au=None, acc=0.5, gap=0.0), "muslim90", RULE)
    assert st["status"] == EXCLUDED and st["failed"] == ["learned", "not_collapsed", "shortcut_measurable"]
    assert "AUROC n/a" in st["reason"] and "accuracy 0.5000 < 0.8" in st["reason"]


def test_boundaries_are_inclusive():
    assert decide(_met(au=0.80, acc=0.80, gap=0.05), "muslim90", RULE)["status"] == INCLUDED


def test_shortcut_condition_only_on_planted_arm():
    st = decide(_met(gap=-0.01), "muslim_nat", RULE)                         # natural arm: a weak or absent gap is a result
    assert st["status"] == INCLUDED and st["shortcut_condition_applies"] is False
    assert decide(_met(gap=-0.01), "muslim90", RULE)["status"] == EXCLUDED


def test_rule_ignores_method_outcomes():
    m = _met()
    a = decide(dict(m, shave_delta=0.3, leace_delta=-0.2), "muslim90", RULE)
    b = decide(m, "muslim90", RULE)
    assert a == b


def test_status_for_reads_stored_recomputes_or_missing(tmp_path):
    d = tmp_path / "muslim90" / "s42"
    (d / "base").mkdir(parents=True)
    (d / "base" / "metrics.json").write_text(json.dumps({"metrics": _met(gap=0.01)}))
    st = status_for(tmp_path, "muslim90", 42, RULE)                           # no inclusion.json: recomputed, same rule
    assert st["status"] == EXCLUDED and st["recomputed_by_aggregate"]
    (d / "check").mkdir()
    (d / "check" / "inclusion.json").write_text(json.dumps({**decide(_met(gap=0.01), "muslim90", RULE), "arm": "muslim90", "seed": 42}))
    assert status_for(tmp_path, "muslim90", 42, RULE)["status"] == EXCLUDED and "recomputed_by_aggregate" not in status_for(tmp_path, "muslim90", 42, RULE)
    assert status_for(tmp_path, "muslim90", 84, RULE)["status"] == MISSING


METHODS = ["base", "shave", "leace_oracle", "dfr"]


def _write_model(res, arm, seed, base_ba, gap, delta, edited=True):
    for m in METHODS:
        p = res / arm / f"s{seed}" / m; p.mkdir(parents=True, exist_ok=True)
        met = {"muslim_ba": base_ba + (0 if m == "base" else delta), "gap_muslim": gap, "acc_test": 0.9, "auroc_test": 0.92,
               "worst_group_acc": 0.3, "spillover_mean_abs": 0.0, "muslim_tpr": 0.5, "muslim_fpr": 0.1}
        (p / "metrics.json").write_text(json.dumps({"method": m, "edited": edited if m != "base" else False, "metrics": met}))


def test_aggregate_uses_included_models_only_and_lists_every_seed(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    cfg.update(arms=["muslim90"], seeds=[42, 84, 126]); cfg["stats"]["n_boot"] = 200
    cfg["data"]["expected_sha256_file"] = str(ROOT / "configs" / "data_expected_sha256.txt")
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    res = tmp_path / "results"
    _write_model(res, "muslim90", 42, 0.55, 0.70, 0.10)
    _write_model(res, "muslim90", 84, 0.55, 0.01, -0.40)                       # shortcut did not take hold: excluded
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    agg = tmp_path / "aggregate"
    ms = {row["seed"]: row for row in csv.DictReader(open(agg / "model_status.csv"))}
    assert (ms["42"]["status"], ms["84"]["status"], ms["126"]["status"]) == ("included", "excluded", "missing")
    assert ms["84"]["failed"] == "shortcut_measurable"
    st = json.loads((agg / "paired_stats.json").read_text())
    prim = [x for x in st if x["metric"] == "muslim_ba" and x["population"].startswith("all")]
    assert prim and all(x["n"] == 1 and abs(x["mean_delta"] - 0.10) < 1e-9 for x in prim)   # seed 84 left out for every method
    per_seed = list(csv.DictReader(open(agg / "per_seed_metrics.csv")))
    assert {row["model_status"] for row in per_seed if row["seed"] == "84"} == {"excluded"}  # still listed, marked
    counts = list(csv.DictReader(open(agg / "edit_counts.csv")))
    assert all(c["n_seeds"] == "1" for c in counts)
    rep = (agg / "REPORT_aggregate.md").read_text()
    assert "## Model status" in rep and "| muslim90 | 84 | excluded |" in rep and "| muslim90 | 126 | missing |" in rep
    assert "muslim90: 1 included, 1 excluded, 1 missing." in rep
    tex = (agg / "table_model_status.tex").read_text()
    assert tex.count("\\\\") >= 4 and "reason: M gap 0.0100" in tex and "excluded" in tex
