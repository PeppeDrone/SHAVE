"""The shortcut check and the exclusion rule read their thresholds from the config, never from code defaults."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from interp_pilot import civil as C
from interp_pilot.config import load
from interp_pilot.exclusion import decide, rule_from_config

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CHECK = {"auroc_min": 0.80, "gap_go": 0.05, "gap_weak": 0.02, "swap_go": 0.20, "swap_weak": 0.05, "swap_min_n": 30}
FROZEN_EXCL = {"auroc_min": 0.80, "acc_min": 0.80, "shortcut_gap_min": 0.05, "shortcut_arms": ["muslim90"]}


def _m(au=0.92, gap=0.03, flip=None, n_el=0):
    return {"auroc_test": au, "primary": {"gap": gap}, "swap_SWm_neutral": {"fp_flip_to_nontoxic": flip, "n_eligible_fp": n_el}}


def test_config_values_unchanged():
    c = load(ROOT / "configs" / "experiment.yaml")
    assert C.check_thresholds(c) == FROZEN_CHECK and rule_from_config(c) == FROZEN_EXCL
    s = load(ROOT / "configs" / "smoke_cpu.yaml")
    assert C.check_thresholds(s) == FROZEN_CHECK                                   # inherited unchanged
    assert rule_from_config(s) == {"auroc_min": 0.0, "acc_min": 0.0, "shortcut_gap_min": -1.0, "shortcut_arms": ["muslim90"]}


def test_no_threshold_constants_left_in_code():
    for name in ("AUROC_MIN", "GAP_GO", "GAP_WEAK", "SWAP_GO", "SWAP_WEAK", "SWAP_MIN_N"):
        assert not hasattr(C, name), name


@pytest.mark.parametrize("key", list(FROZEN_CHECK))
def test_missing_check_key_fails(key):
    c = {"check": {k: v for k, v in FROZEN_CHECK.items() if k != key}}
    with pytest.raises(ValueError, match=key):
        C.check_thresholds(c)
    with pytest.raises(ValueError):
        C.check_thresholds({})


@pytest.mark.parametrize("key", list(FROZEN_EXCL))
def test_missing_exclusion_key_fails(key):
    with pytest.raises(ValueError):
        rule_from_config({"exclusion": {k: v for k, v in FROZEN_EXCL.items() if k != key}})


def test_verdict_follows_config_values():
    th = dict(FROZEN_CHECK)
    m = _m(gap=0.03)
    assert C.verdict(m, "muslim90", th)["verdict"] == "MARGINAL"
    assert C.verdict(m, "muslim90", dict(th, gap_go=0.025))["verdict"] == "GO"          # lower strong bar
    assert C.verdict(m, "muslim90", dict(th, gap_weak=0.04))["verdict"] == "NO-GO"      # higher weak bar
    assert C.verdict(m, "muslim90", dict(th, auroc_min=0.95))["verdict"] == "NO-GO"     # AUROC bar
    sw = _m(gap=0.0, flip=0.25, n_el=30)
    assert C.verdict(sw, "muslim90", th)["verdict"] == "GO"
    assert C.verdict(sw, "muslim90", dict(th, swap_min_n=31))["verdict"] == "NO-GO"
    assert C.verdict(sw, "muslim90", dict(th, swap_go=0.30))["verdict"] == "MARGINAL"
    assert C.verdict(m, "muslim90", dict(th, gap_go=0.025))["thresholds"] == dict(th, gap_go=0.025)   # recorded as used


def test_exclusion_follows_config_values():
    met = {"auroc_test": 0.85, "acc_test": 0.85, "muslim_ba": 0.6, "gap_muslim": 0.06}
    assert decide(met, "muslim90", FROZEN_EXCL)["status"] == "included"
    assert decide(met, "muslim90", dict(FROZEN_EXCL, shortcut_gap_min=0.07))["status"] == "excluded"
    assert decide(met, "muslim90", dict(FROZEN_EXCL, auroc_min=0.9))["status"] == "excluded"
    assert decide(met, "muslim90", dict(FROZEN_EXCL, acc_min=0.9))["status"] == "excluded"
    assert decide(met, "muslim_nat", dict(FROZEN_EXCL, shortcut_gap_min=0.07, shortcut_arms=["muslim_nat"]))["status"] == "excluded"


@pytest.mark.parametrize("section,key", [("check", "swap_min_n"), ("exclusion", "acc_min")])
def test_check_stage_fails_loudly_without_threshold(tmp_path, section, key):
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    del cfg[section][key]
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out", str(tmp_path), "--arm", "muslim90", "--seed", "42"], capture_output=True, text=True)
    assert r.returncode != 0 and "ValueError" in r.stderr and key in r.stderr
    assert not (tmp_path / "results").exists()                                     # failed before any work
