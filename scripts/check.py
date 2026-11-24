#!/usr/bin/env python3
"""Score the trained (unedited) model on every held-out set; save base margins, the descriptive shortcut check and the
model's inclusion status under the fixed exclusion rule (config `exclusion`, base-model measurements only)."""
import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot.config import load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if cfg.get("smoke"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from interp_pilot import civil as C
    from interp_pilot import runner as R
    from interp_pilot.artifacts import now_rome
    from interp_pilot.exclusion import decide, rule_from_config
    C.check_thresholds(cfg); rule_from_config(cfg)          # fail loudly before any work if a threshold is missing
    art = R.artifacts(a.out, a.arm, a.seed)
    art.timer.start("check_total")
    m = R.load_model(cfg, a.out, a.arm, a.seed)
    ev = R.load_eval(a.out); held = C.held_texts(ev)
    full = R.score_held(m, held, batch_size=int(cfg["shave"]["batch_size"]))
    R.save_base(art, full)
    base = {k: v["margins"] for k, v in full.items()}
    met = R.record(art, "base", full, base, ev, a.arm, {"note": "trained model before any edit"})
    vd = C.verdict(met, a.arm, C.check_thresholds(cfg), smoke=bool(cfg.get("smoke")))
    art.json("check", "shortcut.json", {"arm": a.arm, "seed": a.seed, "finished_rome": now_rome(), "metrics_summary": met["summary"],
                                        **vd, "note": "descriptive only; every seed continues to every method"})
    inc = decide(met, a.arm, rule_from_config(cfg))
    art.json("check", "inclusion.json", {"arm": a.arm, "seed": a.seed, "finished_rome": now_rome(), **inc,
                                         "note": "fixed rule on the unedited model only; every method still runs; tables use included models"})
    art.timer.stop("check_total"); art.write_manifest(stage="check")
    R.mark_done(art, "check", {"verdict": vd["verdict"], "model_status": inc["status"]})
    print("CHECK", a.arm, a.seed, vd["verdict"], inc["status"], json.dumps(met["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
