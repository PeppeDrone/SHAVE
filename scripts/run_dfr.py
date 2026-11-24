#!/usr/bin/env python3
"""DFR (Kirichenko et al. 2023): group-balanced last-layer retraining on held-out labelled data.
Groups = M (exact word "muslim") x gold label; rows = validation rows outside SHAVE's blind pool."""
import argparse
import os
import random
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
    import numpy as np
    from interp_pilot import civil as C
    from interp_pilot import dfr as D
    from interp_pilot import runner as R
    from interp_pilot.artifacts import now_rome
    bs = int(cfg["shave"]["batch_size"]); Dc = cfg["dfr"]
    art = R.artifacts(a.out, a.arm, a.seed)
    art.timer.start("dfr_total")
    m = R.load_model(cfg, a.out, a.arm, a.seed)
    rows, X = R.cached_val_features(m, art, a.out, a.seed, int(cfg["shave"]["pool_size"]), bs)
    y = np.array([r["label"] for r in rows]); g = np.array([2 * int(r["M"]) + int(r["label"]) for r in rows])
    train = C.jl(R.data_dir(a.out) / "arms" / a.arm / "train.jsonl")
    rs = random.Random(a.seed); idx = list(range(len(train))); rs.shuffle(idx)
    Xs = R.readout_features(m, [train[i]["text"] for i in idx[: int(Dc["n_scaler_train_rows"])]], bs)
    fit = D.fit_dfr(X, y, g, Xs, seed=a.seed, n_runs=int(Dc["n_runs"]), c_options=list(Dc["c_options"]))
    ev = R.load_eval(a.out); held = C.held_texts(ev); base = R.load_base(art)
    key = ("readout", "last")
    full = {}
    for k, v in held.items():
        F = m.run(v, capture=[key], batch_size=bs)["captured"][key] if v else np.zeros((0, m.hidden))
        full[k] = {"margins": D.margins(F, fit) if len(F) else np.zeros(0), "logits2": None}
    for nm in ("w", "coef_scaled", "scaler_mean", "scaler_scale"):
        art.array("dfr", nm, fit[nm])
    met = R.record(art, "dfr", full, base, ev, a.arm,
                   {"groups": "M (exact word muslim) x gold label", "n_rows": int(len(y)), "group_counts": fit["group_counts"],
                    "best_c": fit["best_c"], "tune": fit["tune"], "n_runs": fit["n_runs"], "intercept": fit["b"],
                    "n_nonzero_coef": fit["n_nonzero"], "note": "margins are the DFR head's decision values"})
    art.timer.stop("dfr_total"); art.write_manifest(stage="dfr")
    R.mark_done(art, "dfr")
    print(now_rome(), "DFR_DONE", a.arm, a.seed, met["summary"], "C", fit["best_c"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
