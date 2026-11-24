"""Fixed model-exclusion rule, applied once per trained model at the `check` stage, before any method runs.

The rule looks only at the unedited (base) model's own check-stage measurements; it never looks at what SHAVE, LEACE or
DFR do. A model is excluded from every method's tables if any condition below fails; it is never replaced by another
seed. Thresholds are the `exclusion` entry of the config.

  1. learned     test AUROC on T              >= auroc_min           (the classifier learned the task)
  2. not_collapsed test accuracy on T         >= acc_min             (it did not collapse to predicting "toxic")
  3. primary_defined  Muslim balanced accuracy on G is defined        (the primary metric can be computed)
  4. shortcut_measurable  only for arms in shortcut_arms (the planted arm):
                 M gap = acc(Mm) - acc(Dm)    >= shortcut_gap_min    (the planted shortcut took hold and is measurable)

A model without check-stage metrics (a failed stage) is reported as "missing", never as excluded or included.
"""
from __future__ import annotations

import json
from pathlib import Path

INCLUDED, EXCLUDED, MISSING = "included", "excluded", "missing"
KEYS = ("auroc_min", "acc_min", "shortcut_gap_min", "shortcut_arms")


def rule_from_config(cfg: dict) -> dict:
    r = cfg.get("exclusion")
    if not isinstance(r, dict) or any(k not in r for k in KEYS):
        raise ValueError(f"config needs an 'exclusion' entry with keys {KEYS}")
    return {"auroc_min": float(r["auroc_min"]), "acc_min": float(r["acc_min"]),
            "shortcut_gap_min": float(r["shortcut_gap_min"]), "shortcut_arms": list(r["shortcut_arms"])}


def decide(met: dict, arm: str, rule: dict) -> dict:
    """met: the base model's metrics (as written by the check stage). Returns status, failed conditions and reasons."""
    au, acc, ba, gap = met.get("auroc_test"), met.get("acc_test"), met.get("muslim_ba"), met.get("gap_muslim")
    failed, reasons = [], []
    if au is None or au < rule["auroc_min"]:
        failed.append("learned"); reasons.append(f"test AUROC {_f(au)} < {rule['auroc_min']}")
    if acc is None or acc < rule["acc_min"]:
        failed.append("not_collapsed"); reasons.append(f"test accuracy {_f(acc)} < {rule['acc_min']}")
    if ba is None:
        failed.append("primary_defined"); reasons.append("Muslim balanced accuracy undefined")
    needs_gap = arm in rule["shortcut_arms"]
    if needs_gap and (gap is None or gap < rule["shortcut_gap_min"]):
        failed.append("shortcut_measurable"); reasons.append(f"M gap {_f(gap)} < {rule['shortcut_gap_min']} (planted shortcut not measurable)")
    return {"status": EXCLUDED if failed else INCLUDED, "failed": failed,
            "reason": "; ".join(reasons) if reasons else "all conditions met",
            "values": {"auroc_test": au, "acc_test": acc, "muslim_ba": ba, "gap_muslim": gap},
            "shortcut_condition_applies": needs_gap, "rule": rule}


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


def status_for(results_root: Path, arm: str, seed: int, rule: dict) -> dict:
    """Status of one model: from check/inclusion.json if present, else recomputed from base/metrics.json with the same
    rule, else missing. A stored status whose rule differs from the config is recomputed and flagged."""
    d = Path(results_root) / arm / f"s{seed}"
    p, b = d / "check" / "inclusion.json", d / "base" / "metrics.json"
    if p.exists():
        st = json.loads(p.read_text())
        if st.get("rule") == rule:
            return st
    if b.exists():
        st = decide(json.loads(b.read_text())["metrics"], arm, rule)
        st["recomputed_by_aggregate"] = True
        return st
    return {"status": MISSING, "failed": [], "reason": "no check-stage metrics (stage failed or not run)", "values": {},
            "shortcut_condition_applies": arm in rule["shortcut_arms"], "rule": rule}
