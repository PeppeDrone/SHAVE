#!/usr/bin/env python3
"""Aggregate per-seed metrics into tables and paired statistics. Writes <out>/aggregate/.
No numbers are interpreted here; the report lists them.

Model exclusion: every (arm, seed) gets a status from the fixed rule of the check stage (config `exclusion`,
`interp_pilot.exclusion`): included, excluded or missing. All statistics, counts and figures use included models only,
the same set for every method; no seed is added to replace an excluded one. per_seed_metrics.csv keeps every model with
its status, and model_status.{csv,json} plus table_model_status.tex list every seed with its status and reason."""
import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from interp_pilot.config import load
from interp_pilot.paper_tables import write_paper_tables

METHODS = ["shave", "leace_oracle", "dfr"]   # Holm family: these three methods x the two arms (six comparisons)
LABELS = {"shave": "SHAVE", "leace_oracle": "LEACE", "dfr": "DFR", "shave_no_threshold": "SHAVE (no threshold)"}
ARM_LABELS = {"muslim90": "Planted", "muslim_nat": "Natural"}   # internal arm names -> planted arm / natural arm
# labels each method sees (label-free methods first; label-using methods listed separately in every table)
LABELS_USED = {"shave": "none (label-free)", "leace_oracle": "concept labels Z (oracle)",
               "dfr": "validation class labels + M group labels"}
SECONDARY = ["gap_muslim", "worst_group_acc", "acc_test", "spillover_mean_abs", "muslim_tpr", "muslim_fpr", "auroc_test",
             "dba_LGBTQ", "dba_black", "dba_white", "dba_other_religions", "dba_male", "dba_female", "dba_christian", "dba_R", "dba_wilds_muslim_not_M"]


def flat(met: dict) -> dict:
    keys = ["muslim_ba"] + SECONDARY
    return {k: met.get(k) for k in keys}


def latex_status_table(st_rows: list[dict]) -> str:
    """Appendix table: every seed of every arm with its status under the exclusion rule and the reason."""
    g = lambda x: "--" if x is None else f"{x:.3f}"
    esc = lambda t: str(t).replace("_", "\\_").replace(">=", "$\\ge$").replace("<", "$<$")
    out = ["\\begin{tabular}{llllll}", "\\toprule", "Arm & Seed & Status & Test AUROC & Test acc. & $M$ gap \\\\", "\\midrule"]
    for r in st_rows:
        out.append(f"{ARM_LABELS.get(r['arm'], esc(r['arm']))} & {r['seed']} & {r['status']} & {g(r['auroc_test'])} & {g(r['acc_test'])} & {g(r['gap_muslim'])} \\\\")
        if r["status"] != "included":
            out.append(f"\\multicolumn{{6}}{{l}}{{\\quad reason: {esc(r['reason'])}}} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load(a.config)
    import numpy as np
    from interp_pilot.artifacts import now_rome, write_csv, write_json
    from interp_pilot.exclusion import INCLUDED, rule_from_config, status_for
    from interp_pilot.stats import holm, paired_summary
    S = cfg["stats"]; prim = S["primary_metric"]
    res = a.out / "results"; agg = a.out / "aggregate"; agg.mkdir(parents=True, exist_ok=True)
    rule = rule_from_config(cfg)
    status = {(arm, seed): status_for(res, arm, seed, rule) for arm in cfg["arms"] for seed in cfg["seeds"]}
    st_rows = [{"arm": arm, "seed": seed, "status": st["status"], "failed": ",".join(st["failed"]), "reason": st["reason"],
                **{k: st["values"].get(k) for k in ("auroc_test", "acc_test", "muslim_ba", "gap_muslim")},
                "shortcut_condition_applies": st["shortcut_condition_applies"]} for (arm, seed), st in status.items()]
    write_csv(agg / "model_status.csv", st_rows)
    write_json(agg / "model_status.json", {"rule": rule, "models": st_rows})
    included = {k for k, st in status.items() if st["status"] == INCLUDED}
    rows, missing = [], []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            for meth in ["base"] + METHODS + ["shave_no_threshold"]:
                p = res / arm / f"s{seed}" / meth / "metrics.json"
                if not p.exists():
                    missing.append(f"{arm}/s{seed}/{meth}"); continue
                j = json.loads(p.read_text())
                rows.append({"arm": arm, "seed": seed, "method": meth, "model_status": status[(arm, seed)]["status"],
                             "edited": j.get("edited", meth != "base"),
                             "reason": j.get("reason"), **flat(j["metrics"])})
    write_csv(agg / "per_seed_metrics.csv", rows)
    idx = {(r["arm"], r["seed"], r["method"]): r for r in rows if (r["arm"], r["seed"]) in included}   # included models only
    stats, prim_p = [], {}
    for arm in cfg["arms"]:
        for meth in METHODS:
            seeds = [s for s in cfg["seeds"] if (arm, s, meth) in idx and (arm, s, "base") in idx]
            if not seeds:   # method not produced in this run
                continue
            for metric in ["muslim_ba"] + SECONDARY:
                mv = [idx[(arm, s, meth)][metric] for s in seeds]; bv = [idx[(arm, s, "base")][metric] for s in seeds]
                mv = [np.nan if v is None else v for v in mv]; bv = [np.nan if v is None else v for v in bv]
                if metric.startswith("dba_") or metric == "spillover_mean_abs":
                    bv = [0.0] * len(mv)  # these are already changes vs the unedited model
                ps = paired_summary(mv, bv, n_boot=int(S["n_boot"]), seed=int(S["boot_seed"]))
                rec = {"arm": arm, "method": meth, "metric": metric, "population": "all included seeds (abstentions = no edit)", **ps}
                stats.append(rec)
                if metric == prim:
                    prim_p[(arm, meth)] = ps["p_signflip"]
            if meth == "shave":  # conditional on an edit being made (n stated)
                es = [s for s in seeds if idx[(arm, s, meth)]["edited"]]
                mv = [idx[(arm, s, meth)][prim] for s in es]; bv = [idx[(arm, s, "base")][prim] for s in es]
                stats.append({"arm": arm, "method": meth, "metric": prim, "population": "edited included seeds only",
                              **paired_summary(mv, bv, n_boot=int(S["n_boot"]), seed=int(S["boot_seed"]))})
    # secondary, descriptive: SHAVE minus each other method, same seed, primary metric
    for arm in cfg["arms"]:
        for a_ in ("shave",):
            for b_ in METHODS:
                if b_ == a_ or b_.startswith("shave"):
                    continue
                seeds = [s for s in cfg["seeds"] if (arm, s, a_) in idx and (arm, s, b_) in idx]
                av = [np.nan if idx[(arm, s, a_)][prim] is None else idx[(arm, s, a_)][prim] for s in seeds]
                bv = [np.nan if idx[(arm, s, b_)][prim] is None else idx[(arm, s, b_)][prim] for s in seeds]
                stats.append({"arm": arm, "method": a_, "metric": prim, "population": f"versus {b_} (method minus {b_}, included seeds)",
                              **paired_summary(av, bv, n_boot=int(S["n_boot"]), seed=int(S["boot_seed"]))})
    adj = holm({f"{k[0]}|{k[1]}": v for k, v in prim_p.items()})
    for r in stats:
        if r["metric"] == prim and r["population"].startswith("all"):
            r["p_signflip_holm"] = adj.get(f"{r['arm']}|{r['method']}")
    write_csv(agg / "paired_stats.csv", [{k: v for k, v in r.items() if k != "deltas"} for r in stats])
    write_json(agg / "paired_stats.json", stats)
    counts = []
    for arm in cfg["arms"]:
        for meth in METHODS:
            rs = [idx[(arm, s, meth)] for s in cfg["seeds"] if (arm, s, meth) in idx]
            if not rs:
                continue
            counts.append({"arm": arm, "method": meth, "n_seeds": len(rs), "n_edited": sum(1 for r in rs if r["edited"]),
                           "n_no_edit": sum(1 for r in rs if not r["edited"]),
                           f"n_test_acc_below_{S['collapse_acc']}": sum(1 for r in rs if (r["acc_test"] or 0) < float(S["collapse_acc"]))})
    write_csv(agg / "edit_counts.csv", counts)
    # report (tables only)
    L = [f"# Aggregate report", "", f"Written {now_rome()}. Config: `{cfg['_config_path']}`.", "",
         f"Primary metric: `{prim}` (Muslim balanced accuracy: M rows of G, M = exact word 'muslim'). Deltas are method minus the unedited model of the same seed.",
         "Abstentions count as no edit (delta 0 on every metric). p: exact paired sign-flip test; Holm over the primary family "
         "(SHAVE, LEACE and DFR in both arms: six comparisons).",
         f"Only models included by the fixed exclusion rule are used ({len(included)} of {len(status)}); see 'Model status' below.", ""]
    for arm in cfg["arms"]:
        L += [f"## Arm {arm}", "", "| Method | labels used | n | mean base | mean method | mean Δ | median Δ | 95% CI (bootstrap) | p sign-flip | p Holm | p Wilcoxon |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in stats:
            if r["arm"] == arm and r["metric"] == prim and not r["population"].startswith("versus"):
                f = lambda x: "–" if x is None else f"{x:.4f}"
                L.append(f"| {LABELS[r['method']]}{' (edited seeds only)' if r['population'].startswith('edited') else ''} | {LABELS_USED[r['method']]} | {r['n']} | {f(r['mean_base'])} | "
                         f"{f(r['mean_method'])} | {f(r['mean_delta'])} | {f(r['median_delta'])} | [{f(r['ci95_low'])}, {f(r['ci95_high'])}] | "
                         f"{f(r['p_signflip'])} | {f(r.get('p_signflip_holm'))} | {f(r['p_wilcoxon'])} |")
        L += ["", "Secondary (descriptive, no multiplicity correction): SHAVE minus each other method on the primary metric.", "",
              "| Comparison | n | mean Δ | median Δ | 95% CI | p sign-flip |", "|---|---|---|---|---|---|"]
        for r in stats:
            if r["arm"] == arm and r["population"].startswith("versus"):
                f = lambda x: "–" if x is None else f"{x:.4f}"
                L.append(f"| {LABELS[r['method']]} {r['population']} | {r['n']} | {f(r['mean_delta'])} | {f(r['median_delta'])} | "
                         f"[{f(r['ci95_low'])}, {f(r['ci95_high'])}] | {f(r['p_signflip'])} |")
        L += ["", "| Method | metric | mean Δ | 95% CI |", "|---|---|---|---|"]
        for r in stats:
            if r["arm"] == arm and r["metric"] != prim and r["population"].startswith("all"):
                f = lambda x: "–" if x is None else f"{x:.4f}"
                L.append(f"| {LABELS[r['method']]} | {r['metric']} | {f(r['mean_delta'])} | [{f(r['ci95_low'])}, {f(r['ci95_high'])}] |")
        L.append("")
    L += ["## Edits and abstentions", "", "| arm | method | included seeds | edited | no edit | test acc below collapse value |", "|---|---|---|---|---|---|"]
    for c in counts:
        L.append(f"| {c['arm']} | {LABELS[c['method']]} | {c['n_seeds']} | {c['n_edited']} | {c['n_no_edit']} | {list(c.values())[-1]} |")
    L += ["", "## Model status (every seed; fixed exclusion rule on the unedited model)", "",
          f"Rule: test AUROC >= {rule['auroc_min']}; test accuracy >= {rule['acc_min']}; Muslim balanced accuracy defined; "
          f"on {', '.join(rule['shortcut_arms'])} only, M gap >= {rule['shortcut_gap_min']}. Excluded models are not replaced.", "",
          "| arm | seed | status | test AUROC | test acc | M gap | reason |", "|---|---|---|---|---|---|---|"]
    g = lambda x: "–" if x is None else f"{x:.4f}"
    for r in st_rows:
        L.append(f"| {r['arm']} | {r['seed']} | {r['status']} | {g(r['auroc_test'])} | {g(r['acc_test'])} | {g(r['gap_muslim'])} | {r['reason']} |")
    for arm in cfg["arms"]:
        n = {k: sum(1 for r in st_rows if r["arm"] == arm and r["status"] == k) for k in ("included", "excluded", "missing")}
        L.append(f"\n{arm}: {n['included']} included, {n['excluded']} excluded, {n['missing']} missing.")
    (agg / "table_model_status.tex").write_text(latex_status_table(st_rows))
    if missing:
        L += ["", f"Missing metrics files ({len(missing)}): " + ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else "")]
    extra = write_paper_tables(a.out, cfg, idx, included, stats, agg)
    paper = agg / "REPORT_paper_tables.md"
    if paper.exists():
        L += ["", paper.read_text()]
    (agg / "REPORT_aggregate.md").write_text("\n".join(L) + "\n")
    write_json(agg / "aggregate_meta.json", {"written_rome": now_rome(), "n_rows": len(rows), "missing": missing,
                                             "n_models": len(status), "n_included": len(included), "paper_tables": extra})
    print("AGGREGATE_DONE rows", len(rows), "missing", len(missing), "paper_tables", extra, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
