#!/usr/bin/env python3
"""SHAVE (Shortcut Hunting with Abstention and Verified Editing): find a word, abstain or verify an edit.

Inputs the method may see: unlabelled validation texts (the blind pool) and the candidate field list.
No labels, no identity lexicon, no fine-tuning. Steps (all thresholds from configs/experiment.yaml):
  1. a descriptive field search (blank and contrast-swap flips on the KC most / least toxic-scored pool texts);
     target selection does not use this step;
  2. value scan (insert or remove each candidate word in NV texts per predicted class; override rates);
  3. rank words by the Wilson 95% lower bound; convert a removal-based top option to the matching
     insertion effect; record the strength of the selected word;
  4. contrast pairs (insert the word into texts not predicted c*; keep those that flip to c*), split in halves;
  5. candidate cuts (span cuts at embed + every layer; last-token cuts at every layer x ranks 1, 2, 4), every
     candidate folded into the weights and scored on T_ev (flip) and O (side, KL); no early stop;
  6. acceptance: flip >= 0.90, side <= 0.05, KL <= 0.10; choose min(KL+side), then max flip, then min rank;
  7. the paper's SHAVE decision applies the 0.20 strength threshold to the saved insertion measurement and
     keeps the verified edit only if the word passes; an ablation without the threshold uses the same
     candidate edits and acceptance criteria. The chosen cut, when kept, is folded and scored on the
     held-out test sets; unblinding follows after the decision is fixed.

The strength threshold in the paper was applied after the runs to the saved measurements, so the
candidate search is always run when a word is selected (even if the word later fails the threshold).
"""
import argparse
import json
import os
from collections import Counter
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
    from interp_pilot import shave as O
    from interp_pilot import runner as R
    from interp_pilot.artifacts import now_rome
    from interp_pilot.civil import blank_field, field_value, is_word_field, preds, replace_value
    from interp_pilot.model import WeightSnapshot, fold_cut, orthonormal_rows, projector, untie_embeddings

    P = cfg["shave"]; s, arm = a.seed, a.arm
    bs = int(P["batch_size"])
    torch.manual_seed(s)
    art = R.artifacts(a.out, arm, s, config={"shave": P, "config_file": cfg["_config_path"]})
    art.timer.start("shave_total")
    out = {"arm": arm, "seed": s, "started_rome": now_rome(), "config": P}

    def save_summary():
        art.json("shave", "summary.json", out)

    fields = C.load_candidate_fields(R.data_dir(a.out) / "arms" / arm / "candidate_fields.json")
    rng, val, pool_rows = R.blind_pool(a.out, s, int(P["pool_size"]))
    pool = [r["text"] for r in pool_rows]
    art.json("shave", "blind_pool_ids.json", [r["id"] for r in pool_rows])
    ev = R.load_eval(a.out); held = C.held_texts(ev)
    base = R.load_base(art)

    m = R.load_model(cfg, a.out, arm, s)
    out["untie"] = untie_embeddings(m)
    snap = WeightSnapshot(m)
    nL = m.n_layers

    def run_sets(sets):
        return {k: (m.run(v, batch_size=bs)["margins"] if v else np.zeros(0)) for k, v in sets.items()}

    # ---------------- step 1: field search
    art.timer.start("shave_step1")
    pm = m.run(pool, batch_size=bs)["margins"]
    order = np.argsort(pm)
    KC = min(int(P["kc"]), max(1, len(pool) // 2))
    Cset = {0: [pool[i] for i in order[:KC]], 1: [pool[i] for i in order[::-1][:KC]]}
    st1, st1_rows = {}, []
    for f in fields:
        r_ = {}
        for c in (0, 1):
            other = [field_value(pool[i], f) for i in (order[::-1][:KC] if c == 0 else order[:KC])]
            other = [v for v in other if v is not None]
            bl = m.run([blank_field(t, f) for t in Cset[c]], batch_size=bs)["margins"]
            sw = []
            for t in Cset[c]:
                v = field_value(t, f); cand = [x for x in other if x != v]
                sw.append(replace_value(t, f, rng.choice(cand)) if (v is not None and cand) else t)
            sm = m.run(sw, batch_size=bs)["margins"]
            r_[c] = {"blank_flip": float(np.mean(preds(bl) != c)), "cswap_flip": float(np.mean(preds(sm) != c)),
                     "n_present": int(sum(field_value(t, f) == "present" for t in Cset[c])) if is_word_field(f) else None}
        st1[f] = {"by_class": r_, "and_score": max(max(v for k, v in x.items() if k != "n_present") for x in r_.values())}
        st1_rows.append({"field": f, "and_score": st1[f]["and_score"], **{f"c{c}_{k}": v for c in (0, 1) for k, v in r_[c].items()}})
    rank1 = sorted(fields, key=lambda f: -st1[f]["and_score"])
    for row in st1_rows:
        row["rank"] = rank1.index(row["field"]) + 1
    art.csv("shave", "step1_ranking.csv", sorted(st1_rows, key=lambda r: r["rank"]))
    out["step1"] = {"ranking": rank1}
    art.timer.stop("shave_step1"); save_summary()

    # ---------------- value scan + target rule
    art.timer.start("shave_value_scan")
    NV = int(P["nv"]); pred_pool = preds(pm)
    N0 = [pool[i] for i in np.where(pred_pool == 0)[0][:NV]]
    N1 = [pool[i] for i in np.where(pred_pool == 1)[0][:NV]]
    scan, scan_rows = {}, []
    for f in fields:
        cnt = Counter(field_value(t, f) for t in pool)
        if is_word_field(f):
            vals = [v for v in ("present", "absent") if cnt[v] >= int(P["min_value_count"])]
        else:
            vals = [v for v, n in cnt.most_common(25) if v is not None and n >= int(P["min_value_count"])]
        rows = []
        for v in vals:
            a1 = [replace_value(t, f, v) for t in N0 if field_value(t, f) != v]
            a0 = [replace_value(t, f, v) for t in N1 if field_value(t, f) != v]
            o1 = float(np.mean(preds(m.run(a1, batch_size=bs)["margins"]) == 1)) if a1 else 0.0
            o0 = float(np.mean(preds(m.run(a0, batch_size=bs)["margins"]) == 0)) if a0 else 0.0
            rows.append({"value": v, "count": cnt[v], "override_to1": o1, "override_to0": o0,
                         "n_to1": len(a1), "n_to0": len(a0)})
            scan_rows.append({"field": f, **rows[-1]})
        scan[f] = rows
    sel = O.select_target(scan, fields, P["target_rule"])
    fstar, cstar, Vstar = sel["fstar"], int(sel["cstar"]), list(sel["Vstar"])
    out["value_scan"] = {k: v for k, v in sel.items() if k != "rank4_words"}
    out["value_scan"]["target"] = O.target_str(sel)
    art.csv("shave", "value_scan.csv", scan_rows)
    art.json("shave", "value_scan.json", {**out["value_scan"], "scan": scan})
    art.timer.stop("shave_value_scan"); save_summary()
    print(now_rome(), arm, s, "target", out["value_scan"]["target"], flush=True)

    # ---------------- strength threshold (applied to the saved insertion measurement at decision time)
    gate = O.strength_gate(sel, O.min_override_wilson_lb(P))
    out["strength_gate"] = gate
    save_summary()
    print(now_rome(), arm, s, "strength_gate", gate, flush=True)

    # ---------------- contrast pairs (always built when a word is selected, as in the paper runs)
    word_target = is_word_field(fstar)
    search = word_target
    out["candidate_search_run"] = bool(search)
    art.timer.start("shave_pairs")
    if search:
        base_not = [pool[i] for i in range(len(pool)) if pred_pool[i] != cstar and field_value(pool[i], fstar) not in Vstar][: int(P["n_base_not"])]
        cand = [replace_value(t, fstar, Vstar[i % len(Vstar)]) for i, t in enumerate(base_not)]
        cm_ = m.run(cand, batch_size=bs)["margins"] if cand else np.zeros(0)
        pairs = [(t2, t1) for t1, t2, mg in zip(base_not, cand, cm_) if preds(np.array([mg]))[0] == cstar][: int(P["n_pairs"])]
        Ot = [t for t in pool if field_value(t, fstar) not in Vstar][-int(P["n_side"]):]
    else:
        base_not, cand, pairs, Ot = [], [], [], []
    T = [p[0] for p in pairs]; H = [p[1] for p in pairs]; k2 = max(1, len(T) // 2)
    T_fit, T_ev, H_fit, H_ev = T[:k2], T[k2:] or T[:k2], H[:k2], H[k2:] or H[:k2]
    out["discovery_n"] = {"base_not": len(base_not), "transplants": len(cand), "pairs": len(T), "T_fit": len(T_fit), "T_ev": len(T_ev), "O": len(Ot)}
    out["checks"] = {"T_without_target": int(sum(field_value(t, fstar) not in Vstar for t in T)) if word_target else None,
                     "H_with_target": int(sum(field_value(t, fstar) in Vstar for t in H)) if word_target else None,
                     "zero_capture_rows": {}}
    art.json("shave", "contrast_pairs.json", {"T_fit": T_fit, "H_fit": H_fit, "T_ev": T_ev, "H_ev": H_ev, "O": Ot,
                                             "fstar": fstar, "Vstar": Vstar, "cstar": cstar})
    disc = {"T_ev": T_ev, "O": Ot}
    bdisc = run_sets(disc) if (T_ev and Ot) else {"T_ev": np.zeros(0), "O": np.zeros(0)}
    art.timer.stop("shave_pairs")

    # ---------------- candidate cuts (every candidate built and scored; no early stop)
    art.timer.start("shave_candidates")
    specs = O.grid_specs(nL, len(Vstar), P["grid"], P["ranks"]) if (T_fit and H_fit) else []
    offs = int(P["offsets"])
    span_cache, last_cache, tried = {}, {}, []

    def span_rows(site, per_value):
        if site not in span_cache:
            keys = [(site, f"{fstar}@{k}") for k in range(offs)]
            cT = m.run(T_fit, capture=keys, batch_size=bs)["captured"]; cH = m.run(H_fit, capture=keys, batch_size=bs)["captured"]
            out["checks"]["zero_capture_rows"][str(site)] = {"T": int(sum(int(np.all(cT[k] == 0, axis=1).sum()) for k in keys)),
                                                             "H": int(sum(int(np.all(cH[k] == 0, axis=1).sum()) for k in keys)),
                                                             "rows_per_side": len(keys) * len(T_fit)}
            span_cache[site] = (keys, cT, cH)
        keys, cT, cH = span_cache[site]
        masks = [np.ones(len(T_fit), bool)] if not per_value else [np.array([field_value(t, fstar) == v for t in T_fit]) for v in Vstar]
        return O.span_cut_rows(cT, cH, keys[: (4 if per_value else offs)], masks, float(P["span_keep_frac"]),
                               min_group=2 if cfg.get("smoke") else 5)

    def last_rows(site, k):
        if site not in last_cache:
            key = (site, "last")
            last_cache[site] = (m.run(T_fit, capture=[key], batch_size=bs)["captured"][key], m.run(H_fit, capture=[key], batch_size=bs)["captured"][key])
        XT, XH = last_cache[site]
        return O.last_token_cut_rows(XT, XH, k)

    for i, (site, fam, rank) in enumerate(specs):
        U = span_rows(site, fam == "span_per_value") if fam != "last" else last_rows(site, rank)
        if U is None:
            tried.append({"candidate": i, "site": str(site), "family": fam, "rank": None, "built": False}); continue
        fold_cut(m, U, site); r = run_sets(disc); snap.restore()
        e = {"candidate": i, "site": str(site), "family": fam, "rank": int(orthonormal_rows(U).shape[0]), "built": True,
             **O.cut_scores(bdisc["T_ev"], bdisc["O"], r["T_ev"], r["O"], cstar)}
        e["eligible"] = O.eligible(e, P["flip_min"], P["side_max"], P["kl_max"])
        art.array("shave", f"candidates/cand{i:03d}_{e['site']}_{fam}_r{e['rank']}", U, meta={"site": e["site"], "family": fam})
        e["_U"], e["_site"] = U, site
        tried.append(e)
        print(now_rome(), arm, s, "cand", {k: v for k, v in e.items() if not k.startswith("_")}, flush=True)
    built = [e for e in tried if e.get("built")]
    art.csv("shave", "candidate_sweep.csv", [{k: v for k, v in e.items() if not k.startswith("_")} for e in tried])
    art.timer.stop("shave_candidates")
    out["n_candidates"] = {"specified": len(specs), "built": len(built), "eligible": sum(1 for e in built if e["eligible"])}

    # ---------------- choose, fold, score
    # Same candidate list and acceptance bars for both decisions. SHAVE keeps an eligible edit only
    # when the selected word also passes the strength threshold; the no-threshold ablation does not
    # apply that gate.
    art.timer.start("shave_edit_eval")
    ch_not = O.select_thresholded(built, P["flip_min"], P["side_max"], P["kl_max"])
    ch_t = ch_not if gate["passed"] else None
    scored = {}

    def apply_and_score(e):
        key = e["candidate"]
        if key not in scored:
            info = fold_cut(m, e["_U"], e["_site"])
            scored[key] = (R.score_held(m, held, batch_size=bs), {k: v for k, v in info.items() if k != "edits"})
            snap.restore()
        return scored[key]

    base_full = {k: {"margins": v, "logits2": None} for k, v in base.items()}
    bars_none = "abstain: no candidate passed flip/side/KL" if built else "abstain: no candidate could be built"
    methods = [("shave_no_threshold", ch_not, bars_none),
               ("shave", ch_t, bars_none if gate["passed"] else O.WEAK_TARGET)]
    for method, ch, why_none in methods:
        if ch is None:
            reason = why_none if word_target else "abstain: no word-field target"
            rec = {"edited": False, "decision": "abstain", "reason": reason, "chosen": None}
            if reason == O.WEAK_TARGET:
                rec["detail"] = (f"target Wilson 95% lower bound {gate['wilson_lb']:.4f} < "
                                 f"min_override_wilson_lb {gate['min_wilson_lb']:.4f}")
            met = R.record(art, method, base_full, base, ev, arm, rec)
            out[method] = {k: v for k, v in rec.items() if k != "chosen"}
        else:
            full, info = apply_and_score(ch)
            pub = {k: v for k, v in ch.items() if not k.startswith("_")}
            met = R.record(art, method, full, base, ev, arm, {"edited": True, "decision": "edit", "chosen": pub, "fold": info})
            art.array(method, "chosen_U", ch["_U"], meta={"site": str(ch["_site"])})
            art.array(method, "chosen_projector", projector(ch["_U"], "cpu").numpy(), meta={"site": str(ch["_site"])})
            out[method] = {"edited": True, "decision": "edit", "chosen": pub}
        out[method]["summary"] = met["summary"]
        print(now_rome(), arm, s, method, out[method], flush=True)
    save_summary()
    art.timer.stop("shave_edit_eval")

    # ---------------- structure test (descriptive) and unblinding (after the edit is fixed)
    if word_target and base_not:
        bn2 = base_not[:200]

        def suff(edit):
            return float(np.mean(preds(m.run([edit(t) for t in bn2], batch_size=bs)["margins"]) == cstar))
        f2 = next((f for f in sel["rank4_words"] if f != fstar), None)
        v2 = sel["best"][f2][2] if f2 else None
        out["structure"] = {"per_value_suff": {str(v): suff(lambda t, v=v: replace_value(t, fstar, v)) for v in Vstar},
                            "second_field": f2, "second_value": v2,
                            "suff_second_alone": suff(lambda t: replace_value(t, f2, v2)) if v2 else None,
                            "suff_both": suff(lambda t: replace_value(replace_value(t, fstar, Vstar[0]), f2, v2)) if v2 else None}
    cue_fields = {f"cc_w:{t}" for t in C.SWAP_TABLE}
    rank4 = sel["field_ranking"]
    out["unblind"] = {"fstar": fstar, "fstar_is_identity_term": fstar in cue_fields,
                      "fstar_group": C.SWAP_TABLE[C.word_of(fstar)][0] if fstar in cue_fields else None,
                      "fstar_is_muslim_term": C.word_of(fstar) in ("muslim", "muslims", "islam", "islamic") if is_word_field(fstar) else False,
                      "step1_rank_of_identity_fields": {f: rank1.index(f) + 1 for f in sorted(cue_fields) if f in rank1},
                      "value_scan_rank_of_identity_fields": {f: rank4.index(f) + 1 for f in sorted(cue_fields) if f in rank4},
                      "note": "unblinding only; the identity lexicon was not used to choose the edit"}
    out["finished_rome"] = now_rome(); save_summary()
    art.timer.stop("shave_total"); art.write_manifest(stage="shave")
    for method, _, _ in methods:   # "shave" last: its DONE.json marks the whole stage as finished
        R.mark_done(art, method, {"edited": out[method]["edited"], "decision": out[method]["decision"]})
    print("OURS_DONE", arm, s, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
