#!/usr/bin/env python3
"""LEACE (Belrose et al. 2023) with the oracle Muslim concept: erase the concept from the frozen readout of the
classifier, fit the logistic-regression head on the erased training features (paper Sec. 5.1-5.2; concept-erasure
README, Batch usage). See src/interp_pilot/leace_oracle.py."""
import argparse
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
    import numpy as np
    import torch
    from interp_pilot import civil as C
    from interp_pilot import leace_oracle as L
    from interp_pilot import runner as R
    from interp_pilot.artifacts import now_rome
    from interp_pilot.metrics import paper_metrics
    if [t.lower() for t in cfg["leace"]["terms"]] != L.MUSLIM_TERMS:
        raise SystemExit("config leace.terms differs from leace_oracle.MUSLIM_TERMS")
    bs = int(cfg["shave"]["batch_size"])
    art = R.artifacts(a.out, a.arm, a.seed)
    art.timer.start("leace_total")
    m = R.load_model(cfg, a.out, a.arm, a.seed)
    train = C.jl(R.data_dir(a.out) / "arms" / a.arm / "train.jsonl")
    art.timer.start("leace_features")
    fp = art.dir("features") / "readout_train.npz"
    ids = np.array([str(r["id"]) for r in train])
    X = None
    if fp.exists():
        zf = np.load(fp)
        if np.array_equal(zf["ids"], ids):
            X = zf["X"].astype(np.float64)
    if X is None:
        X = R.readout_features(m, [r["text"] for r in train], bs).astype(np.float64)
        np.savez(fp, ids=ids, X=X.astype(np.float32))
    art.timer.stop("leace_features")
    z = L.concept_z([r["comment"] for r in train]); y = np.array([int(r["label"]) for r in train])
    fit = L.fit_eraser(X, z)
    Xe = L.erase(X, fit)
    head = L.fit_head(Xe, y, a.seed)
    check = {"max_abs_cov_before": L.residual_cross_cov(X, z), "max_abs_cov_after": L.residual_cross_cov(Xe, z),
             "probe_after": L.probe_check(Xe, z, a.seed), "head_train_acc": float(head.score(Xe, y))}
    ev = R.load_eval(a.out); held = C.held_texts(ev); base = R.load_base(art)
    key = ("readout", "last")
    full = {}
    for k, v in held.items():
        F = m.run(v, capture=[key], batch_size=bs)["captured"][key].astype(np.float64) if v else np.zeros((0, m.hidden))
        full[k] = {"margins": L.head_margins(F, fit, head), "logits2": None}
    # record only: the classifier's own read-out applied to the erased readout (no head refit)
    iv = [{"site": "readout", "type": "affine", "pos": "all", "A": torch.tensor(fit["A"]), "mu": torch.tensor(fit["mu"])}]
    fixed = R.score_held(m, held, iv, batch_size=bs)
    art.predictions("leace_oracle", fixed, C.meta_rows(ev), name="fixed_head_predictions")
    fixed_met = paper_metrics({k: v["margins"] for k, v in fixed.items()}, base, ev, a.arm)
    art.array("leace_oracle", "A", fit["A"]); art.array("leace_oracle", "mu", fit["mu"])
    art.array("leace_oracle", "head_coef", head.coef_.astype(np.float64)); art.array("leace_oracle", "head_intercept", head.intercept_)
    art.json("leace_oracle", "fit_rows.json", {"ids": ids.tolist(), "z": z.tolist()})
    met = R.record(art, "leace_oracle", full, base, ev, a.arm,
                   {"concept": "group M: exact word muslim (oracle)", "terms": L.MUSLIM_TERMS, "site": "readout (frozen)",
                    "fit_rows": "arm training rows", "n_fit": int(len(z)), "n_z1": int(z.sum()), "impl": fit["impl"],
                    "rank_A": fit["rank"], "head": f"LogisticRegression(max_iter={L.HEAD_MAX_ITER}) on erased training features",
                    "fit_check": check, "record_only_fixed_head_summary": fixed_met["summary"]})
    art.timer.stop("leace_total"); art.write_manifest(stage="leace")
    R.mark_done(art, "leace_oracle")
    print(now_rome(), "LEACE_DONE", a.arm, a.seed, met["summary"], check, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
