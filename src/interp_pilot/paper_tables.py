"""Tables that appear in the paper, written from a completed run folder.

  * detection: selected word, Wilson lower bound, SHAVE decision, Delta BA_M and Delta WGA
  * no-threshold ablation: same acceptance criteria, without the 0.20 strength threshold
  * word rates: toxic rates of comments with and without black / white / muslim
  * black effects: SHAVE, LEACE and DFR on the T union G comments that contain "black"
  * runtime: mean and sample standard deviation of stage times and peak reserved memory
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import civil as C
from .artifacts import write_csv, write_json
from .stats import paired_summary

ARM_LABELS = {"muslim90": "Planted", "muslim_nat": "Natural"}
METHOD_LABELS = {"shave": "SHAVE", "leace_oracle": "LEACE", "dfr": "DFR",
                 "shave_no_threshold": "SHAVE (no threshold)"}
WORDS = ("black", "white", "muslim")
# Same whole-word rule as Table "blackrates" in the paper: case-insensitive, and a letter
# or apostrophe on either side does not count (so "black's" is excluded).
WORD_RE = {w: re.compile(rf"(?<![A-Za-z']){re.escape(w)}(?![A-Za-z'])", re.I) for w in WORDS}


def has_word(text: str, word: str) -> bool:
    return WORD_RE[word].search(text or "") is not None


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_word(summary: dict) -> str | None:
    vs = summary.get("value_scan") or {}
    fstar = vs.get("fstar")
    if fstar and C.is_word_field(fstar):
        return C.word_of(fstar)
    tgt = vs.get("target") or ""
    if tgt.startswith("cc_w:"):
        return tgt.split("=", 1)[0].split(":", 1)[1]
    return None


def detection_rows(out: Path, cfg: dict, idx: dict) -> list[dict]:
    """One row per included (arm, seed) with SHAVE's selected word and the thresholded decision."""
    rows = []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            summary = _json(out / "results" / arm / f"s{seed}" / "shave" / "summary.json")
            if not summary:
                continue
            gate = summary.get("strength_gate") or {}
            sh = idx.get((arm, seed, "shave"), {})
            base = idx.get((arm, seed, "base"), {})
            dba = None if sh.get("muslim_ba") is None or base.get("muslim_ba") is None else sh["muslim_ba"] - base["muslim_ba"]
            dwga = None if sh.get("worst_group_acc") is None or base.get("worst_group_acc") is None else sh["worst_group_acc"] - base["worst_group_acc"]
            rows.append({
                "arm": arm, "seed": seed, "selected_word": selected_word(summary),
                "wilson_lb": gate.get("wilson_lb"),
                "decision": (summary.get("shave") or {}).get("decision") or ("edit" if sh.get("edited") else "abstain"),
                "delta_muslim_ba": dba, "delta_worst_group_acc": dwga,
            })
    return rows


def word_rate_rows(out: Path) -> list[dict]:
    """Toxic rates of comments with and without each word, on each training arm and the test split."""
    data = out / "data"
    sources = [("Training, natural arm", data / "arms" / "muslim_nat" / "train.jsonl"),
               ("Training, planted arm", data / "arms" / "muslim90" / "train.jsonl"),
               ("Test split (all)", data / "test_all.jsonl" if (data / "test_all.jsonl").exists() else data / "splits" / "test.jsonl")]
    # build.py writes test_all.jsonl at the data root
    if not sources[2][1].exists():
        for cand in (data / "test.jsonl", data / "eval" / "T.jsonl"):
            if cand.exists():
                sources[2] = ("Test split (all)", cand)
                break
    rows = []
    for label, path in sources:
        recs = _jl(path)
        if not recs:
            continue
        comments = [r.get("comment") or C.parse_note(r.get("text", "")) or r.get("text", "") for r in recs]
        labels = [int(r["label"]) for r in recs]
        n = len(recs)
        overall = sum(labels) / n if n else None
        for i, w in enumerate(WORDS):
            mask = [has_word(c, w) for c in comments]
            n_w = sum(mask)
            with_ = (sum(y for y, m in zip(labels, mask) if m) / n_w) if n_w else None
            n_wo = n - n_w
            without = (sum(y for y, m in zip(labels, mask) if not m) / n_wo) if n_wo else None
            rows.append({"data": label, "word": w, "n_with_word": n_w,
                         "toxic_rate_with_word": with_, "toxic_rate_without": without,
                         "overall": overall if i == 0 else None})
    return rows


def _unique_tg(ev: dict) -> list[tuple[str, int, dict]]:
    """Unique comments of T union G, as (set_name, row_index, row). G wins on id collisions."""
    seen, out = {}, []
    for set_name in ("T", "G"):
        for i, r in enumerate(ev.get(set_name) or []):
            seen[str(r["id"])] = (set_name, i, r)
    for item in seen.values():
        out.append(item)
    return out


def _pred_lookup(path: Path) -> dict:
    """(set, row) -> pred, from a method's predictions table."""
    from .artifacts import read_table
    if not path.exists():
        return {}
    cols = read_table(path)
    out = {}
    for i in range(len(cols["set"])):
        out[(cols["set"][i], int(cols["row"][i]))] = int(cols["pred"][i])
    return out


def _pred_path(folder: Path) -> Path:
    for name in ("predictions.parquet", "predictions.jsonl.gz", "predictions.jsonl"):
        p = folder / name
        if p.exists():
            return p
    return folder / "predictions.parquet"


def black_effect_rows(out: Path, cfg: dict, included: set) -> tuple[list[dict], dict]:
    """Natural-arm effects on T union G comments containing 'black' (Table blackeffects)."""
    ev_dir = out / "data" / "eval"
    ev = {k: _jl(ev_dir / f"{k}.jsonl") for k in ("T", "G")}
    items = _unique_tg(ev)
    black, ref = [], []
    for set_name, i, r in items:
        c = r.get("comment") or C.parse_note(r.get("text", "")) or r.get("text", "")
        if has_word(c, "black"):
            black.append((set_name, i, r))
        if int(r["label"]) == 0 and not r.get("identity_any"):
            ref.append((set_name, i, r))
    meta = {"n_black": len(black),
            "n_black_toxic": sum(int(r["label"]) == 1 for _, _, r in black),
            "n_black_nontoxic": sum(int(r["label"]) == 0 for _, _, r in black),
            "n_ref_nontoxic_no_identity": len(ref),
            "arm": "muslim_nat"}

    def rates(lookup):
        if not black or not lookup:
            return None
        y = np.array([int(r["label"]) for _, _, r in black])
        p = np.array([lookup[(s, i)] for s, i, r in black])
        pos, neg = y == 1, y == 0
        acc_t = float(np.mean(p[pos] == 1)) if pos.any() else None
        acc_n = float(np.mean(p[neg] == 0)) if neg.any() else None
        ba = None if acc_t is None or acc_n is None else 0.5 * (acc_t + acc_n)
        fpr_b = float(np.mean(p[neg] == 1)) if neg.any() else None
        if ref:
            pr = np.array([lookup[(s, i)] for s, i, r in ref])
            fpr_r = float(np.mean(pr == 1))
        else:
            fpr_r = None
        gap = None if fpr_b is None or fpr_r is None else fpr_b - fpr_r
        return {"ba": ba, "acc_toxic": acc_t, "acc_nontoxic": acc_n, "fpr_gap": gap}

    seeds = [s for s in cfg["seeds"] if ("muslim_nat", s) in included]
    base_levels, deltas = [], {m: [] for m in ("shave", "leace_oracle", "dfr")}
    for seed in seeds:
        root = out / "results" / "muslim_nat" / f"s{seed}"
        base_lu = _pred_lookup(_pred_path(root / "base")) or _pred_lookup(_pred_path(root / "check"))
        # unedited margins also live in check/base; predictions may be under base after the check stage
        if not base_lu:
            # fall back to a method file that abstained (predictions equal the unedited model)
            for meth in ("shave", "leace_oracle", "dfr"):
                # always prefer explicit base; if missing, use check margins via metrics only
                pass
        br = rates(base_lu) if base_lu else None
        if br:
            base_levels.append(br)
        for meth in deltas:
            lu = _pred_lookup(_pred_path(root / meth))
            mr = rates(lu) if lu else None
            if br and mr:
                # SHAVE abstentions must count as 0; if the method wrote unedited predictions, the delta is already 0
                deltas[meth].append({k: (None if mr[k] is None or br[k] is None else mr[k] - br[k]) for k in br})

    def mean_med(vals, key):
        xs = [v[key] for v in vals if v.get(key) is not None]
        if not xs:
            return None, None
        return float(np.mean(xs)), float(np.median(xs))

    rows = []
    if base_levels:
        mu = {k: mean_med(base_levels, k)[0] for k in ("ba", "acc_toxic", "acc_nontoxic", "fpr_gap")}
        rows.append({"method": "Unedited (level)", "delta_ba_mean": mu["ba"], "delta_ba_median": None,
                     "delta_acc_toxic_mean": mu["acc_toxic"], "delta_acc_toxic_median": None,
                     "delta_acc_nontoxic_mean": mu["acc_nontoxic"], "delta_acc_nontoxic_median": None,
                     "delta_fpr_gap_mean": mu["fpr_gap"], "delta_fpr_gap_median": None, "kind": "level"})
    for meth, vals in deltas.items():
        rec = {"method": METHOD_LABELS[meth], "kind": "delta"}
        for src, dst in (("ba", "delta_ba"), ("acc_toxic", "delta_acc_toxic"),
                         ("acc_nontoxic", "delta_acc_nontoxic"), ("fpr_gap", "delta_fpr_gap")):
            mu, md = mean_med(vals, src)
            rec[f"{dst}_mean"], rec[f"{dst}_median"] = mu, md
        rows.append(rec)
    return rows, meta


def runtime_rows(out: Path) -> tuple[list[dict], list[dict]]:
    """Per-model stage times and a summary (mean, sample sd, median, min, max) in minutes."""
    per, by = [], defaultdict(list)
    for tf in sorted((out / "results").glob("*/s*/_meta/timings_*.json")):
        arm, seed = tf.parts[-4], int(tf.parts[-3][1:])
        for r in json.loads(tf.read_text()):
            sec = r.get("seconds")
            rec = {"arm": arm, "seed": seed, "stage": r.get("stage"), "seconds": sec,
                   "minutes": None if sec is None else sec / 60.0,
                   "peak_reserved_gib": r.get("peak_reserved_gib"),
                   "peak_alloc_gib": r.get("peak_alloc_gib"),
                   "gpu_name": r.get("gpu_name")}
            per.append(rec)
            if sec is not None:
                by[r.get("stage")].append(rec)
    summary = []
    for stage, recs in by.items():
        mins = np.array([r["minutes"] for r in recs], dtype=float)
        mem = [r["peak_reserved_gib"] for r in recs if r["peak_reserved_gib"] is not None]
        summary.append({
            "stage": stage, "n": len(mins),
            "mean_minutes": float(mins.mean()),
            "sd_minutes": float(mins.std(ddof=1)) if len(mins) > 1 else 0.0,
            "median_minutes": float(np.median(mins)),
            "min_minutes": float(mins.min()), "max_minutes": float(mins.max()),
            "mean_peak_reserved_gib": float(np.mean(mem)) if mem else None,
            "max_peak_reserved_gib": float(np.max(mem)) if mem else None,
        })
    return per, summary


def _fmt(x, digits=3, signed=False):
    if x is None:
        return "–"
    v = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return v


def write_paper_tables(out: Path, cfg: dict, idx: dict, included: set, stats: list, agg: Path) -> dict:
    """Write CSV, JSON and a markdown section for the paper tables. Returns paths written."""
    det = detection_rows(out, cfg, idx)
    write_csv(agg / "detection.csv", det)
    rates = word_rate_rows(out)
    if rates:
        write_csv(agg / "word_toxic_rates.csv", rates)
    effects, effects_meta = [], {}
    if (out / "data" / "eval" / "T.jsonl").exists():
        effects, effects_meta = black_effect_rows(out, cfg, included)
        if effects:
            write_csv(agg / "black_effects.csv", effects)
            write_json(agg / "black_effects_meta.json", effects_meta)
    per, rt_sum = runtime_rows(out)
    if per:
        write_csv(agg / "runtime_per_model.csv", per)
        write_csv(agg / "runtime_summary.csv", rt_sum)

    lines = ["", "## Detection (selected word and SHAVE decision)", "",
             "| arm | seed | selected word | Wilson lower bound | SHAVE | ΔBA_M | ΔWGA |",
             "|---|---|---|---|---|---|---|"]
    for r in det:
        lines.append(f"| {ARM_LABELS.get(r['arm'], r['arm'])} | {r['seed']} | {r['selected_word'] or '–'} | "
                     f"{_fmt(r['wilson_lb'])} | {r['decision']} | {_fmt(r['delta_muslim_ba'], 3, True)} | "
                     f"{_fmt(r['delta_worst_group_acc'], 3, True)} |")

    # no-threshold ablation (descriptive; not in the Holm family)
    lines += ["", "## Ablation without the strength threshold", "",
              "Same candidate edits and acceptance criteria as SHAVE; the 0.20 Wilson lower-bound "
              "threshold is not applied. Not included in the primary Holm family.", "",
              "| arm | n | edited | mean ΔBA_M | median ΔBA_M | p sign-flip | mean ΔWGA | median ΔWGA | mean S |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm in cfg["arms"]:
        seeds = [s for s in cfg["seeds"] if (arm, s, "shave_no_threshold") in idx and (arm, s, "base") in idx]
        if not seeds:
            continue
        def col(metric):
            mv = [idx[(arm, s, "shave_no_threshold")][metric] for s in seeds]
            bv = [idx[(arm, s, "base")][metric] for s in seeds]
            if metric == "spillover_mean_abs":
                bv = [0.0] * len(mv)
            mv = [np.nan if v is None else v for v in mv]
            bv = [np.nan if v is None else v for v in bv]
            return paired_summary(mv, bv, n_boot=int(cfg["stats"]["n_boot"]), seed=int(cfg["stats"]["boot_seed"]))
        ba, wga, s = col("muslim_ba"), col("worst_group_acc"), col("spillover_mean_abs")
        n_ed = sum(1 for s in seeds if idx[(arm, s, "shave_no_threshold")].get("edited"))
        lines.append(f"| {ARM_LABELS.get(arm, arm)} | {len(seeds)} | {n_ed}/{len(seeds)} | "
                     f"{_fmt(ba['mean_delta'], 3, True)} | {_fmt(ba['median_delta'], 3, True)} | "
                     f"{_fmt(ba['p_signflip'], 3)} | {_fmt(wga['mean_delta'], 3, True)} | "
                     f"{_fmt(wga['median_delta'], 3, True)} | {_fmt(s['mean_delta'], 3, True)} |")

    if rates:
        lines += ["", "## Toxic rates of comments with and without a word", "",
                  "| data | word | n with word | toxic rate with word | toxic rate without | overall |",
                  "|---|---|---|---|---|---|"]
        for r in rates:
            lines.append(f"| {r['data']} | {r['word']} | {r['n_with_word']} | "
                         f"{_fmt(r['toxic_rate_with_word'])} | {_fmt(r['toxic_rate_without'])} | "
                         f"{_fmt(r['overall'])} |")
    if effects:
        lines += ["", "## Natural arm: effects on T ∪ G comments containing *black*", "",
                  f"n with word = {effects_meta.get('n_black')} "
                  f"({effects_meta.get('n_black_toxic')} toxic, {effects_meta.get('n_black_nontoxic')} non-toxic); "
                  f"FPR gap reference = {effects_meta.get('n_ref_nontoxic_no_identity')} non-toxic comments "
                  f"with no identity mention in T ∪ G. Unedited row is the mean level; other rows are mean / median change "
                  f"(abstentions count as 0).", "",
                  "| method | ΔBA | Δacc. toxic | Δacc. non-toxic | Δ FPR gap |",
                  "|---|---|---|---|---|"]
        for r in effects:
            if r["kind"] == "level":
                lines.append(f"| {r['method']} | {_fmt(r['delta_ba_mean'])} | {_fmt(r['delta_acc_toxic_mean'])} | "
                             f"{_fmt(r['delta_acc_nontoxic_mean'])} | {_fmt(r['delta_fpr_gap_mean'])} |")
            else:
                cell = lambda a, b: f"{_fmt(a, 3, True)} / {_fmt(b, 3, True)}"
                lines.append(f"| {r['method']} | {cell(r['delta_ba_mean'], r['delta_ba_median'])} | "
                             f"{cell(r['delta_acc_toxic_mean'], r['delta_acc_toxic_median'])} | "
                             f"{cell(r['delta_acc_nontoxic_mean'], r['delta_acc_nontoxic_median'])} | "
                             f"{cell(r['delta_fpr_gap_mean'], r['delta_fpr_gap_median'])} |")
    if rt_sum:
        lines += ["", "## Runtime and memory", "",
                  "| stage | n | mean (min) | sd (min) | median (min) | min | max | mean peak reserved (GiB) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rt_sum:
            lines.append(f"| {r['stage']} | {r['n']} | {_fmt(r['mean_minutes'], 2)} | {_fmt(r['sd_minutes'], 2)} | "
                         f"{_fmt(r['median_minutes'], 2)} | {_fmt(r['min_minutes'], 2)} | {_fmt(r['max_minutes'], 2)} | "
                         f"{_fmt(r['mean_peak_reserved_gib'], 2)} |")
    (agg / "REPORT_paper_tables.md").write_text("\n".join(lines) + "\n")
    return {"detection": len(det), "word_rates": len(rates), "black_effects": len(effects), "runtime_stages": len(rt_sum)}
