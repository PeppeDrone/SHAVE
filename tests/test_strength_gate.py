"""Target-strength gate of SHAVE (shave.strength_gate, shave.min_override_wilson_lb) and its config key."""
from pathlib import Path

import pytest

from interp_pilot.config import load
from interp_pilot.shave import DEFAULT_MIN_WILSON_LB, GATE_KEY, WEAK_TARGET, min_override_wilson_lb, select_target, strength_gate, wilson_lb

ROOT = Path(__file__).resolve().parents[1]


def sel_with_lb(lb):
    return {"choice": {"fstar": "cc_w:x", "cstar": 1, "evidence": {"value": "present", "class": 1, "rate": lb, "n": 150,
                                                                       "wilson_lb": lb}, "converted_from": None}}


def test_gate_abstains_below_threshold():
    g = strength_gate(sel_with_lb(0.15), 0.20)
    assert g["passed"] is False and g["decision"] == "abstain" and g["reason"] == WEAK_TARGET
    assert g["wilson_lb"] == pytest.approx(0.15) and g["min_wilson_lb"] == pytest.approx(0.20)


def test_gate_passes_above_threshold():
    g = strength_gate(sel_with_lb(0.58), 0.20)
    assert g["passed"] is True and g["decision"] == "search" and g["reason"] is None


def test_gate_boundary_passes():
    assert strength_gate(sel_with_lb(0.20), 0.20)["passed"] is True
    assert strength_gate(sel_with_lb(0.20 - 1e-9), 0.20)["passed"] is False


def test_gate_without_word_target():
    g = strength_gate({"choice": None}, 0.20)
    assert g["passed"] is False and g["decision"] == "no_word_target" and g["reason"] is None


def test_gate_reads_evidence_of_select_target():
    """End to end on the target rule: a word that flips 10 of 150 texts is weak, one that flips 90 of 150 is not."""
    def scan(k_present):
        return {"cc_w:a": [{"value": "present", "count": 50, "override_to1": k_present / 150, "override_to0": 0.0, "n_to1": 150, "n_to0": 150},
                           {"value": "absent", "count": 2950, "override_to1": 0.0, "override_to0": 0.0, "n_to1": 150, "n_to0": 150}]}
    weak, strong = select_target(scan(10), ["cc_w:a"]), select_target(scan(90), ["cc_w:a"])
    assert weak["choice"]["evidence"]["wilson_lb"] == pytest.approx(wilson_lb(10 / 150, 150))
    assert strength_gate(weak, 0.20)["passed"] is False
    assert strength_gate(strong, 0.20)["passed"] is True


def test_config_key_read():
    cfg = load(ROOT / "configs" / "experiment.yaml")
    assert GATE_KEY == "min_override_wilson_lb"
    assert min_override_wilson_lb(cfg["shave"]) == pytest.approx(0.20)
    assert "always_edit" not in cfg["shave"]
    smoke = load(ROOT / "configs" / "smoke_cpu.yaml")            # inherits the gate unchanged
    assert min_override_wilson_lb(smoke["shave"]) == pytest.approx(0.20)


def test_config_key_default_and_range(tmp_path):
    assert min_override_wilson_lb({"flip_min": 0.9}) == pytest.approx(DEFAULT_MIN_WILSON_LB) == pytest.approx(0.20)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            min_override_wilson_lb({GATE_KEY: bad})
    p = tmp_path / "c.yaml"
    p.write_text(f"inherit: {ROOT / 'configs' / 'experiment.yaml'}\nshave: {{{GATE_KEY}: 0.35}}\n")
    assert min_override_wilson_lb(load(p)["shave"]) == pytest.approx(0.35)   # a config value overrides the default
