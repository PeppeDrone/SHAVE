"""Paper metrics on top of civil.full_metrics.

All rates are computed on the human-labelled test sets built by build.py:
  G   - 150 rows per (group x gold label) cell for M, R and the 8 WILDS groups, plus 150/150 background rows
        (no WILDS identity, no M, no R);
  Dm  - 800 non-toxic M rows, Mm - their 1:1 matched no-identity controls (same label, nearest length);
  T   - 6000 random test rows.
Definitions (group g, on G):
  TPR_g = P(pred = 1 | g, y = 1), TNR_g = P(pred = 0 | g, y = 0), FPR_g = 1 - TNR_g,
  BA_g  = (TPR_g + TNR_g) / 2.
M = comments with the exact word "muslim"; R = comments without it that contain "muslims", "islam" or "islamic".
Primary: Muslim balanced accuracy BA_M (on the M rows of G). Secondary: Muslim gap = acc(Mm) - acc(Dm); worst-group
accuracy over the 18 (group x label) cells of G for the 8 WILDS groups and M (cells with n >= 10); test accuracy on T;
BA_R and dBA_R (spread to related words).
Spillover (off-target change) for the 7 WILDS groups other than muslim, each computed on its rows WITHOUT M or R
(so a change on Muslim-word rows is never counted as spillover): dBA_g, dTPR_g, dFPR_g vs the unedited model, and
the summary  S = mean over the 7 groups of (|dTPR_g| + |dFPR_g|) / 2. The WILDS muslim group (rows without M) is
reported descriptively (dba_wilds_muslim_not_M).
"""
from __future__ import annotations

import numpy as np

from . import civil as C

OFF_TARGET = [g for g in C.GROUPS if g != "muslim"]
HIGHLIGHT = ["LGBTQ", "black", "white", "other_religions"]


RATE_GROUPS = ["M", "R"] + C.GROUPS


def group_mask(rows, g: str, exclude_word_groups: bool) -> np.ndarray:
    m = np.array([C.member(r, g) for r in rows], dtype=bool)
    if exclude_word_groups and g not in C.WORD_GROUPS:
        m &= np.array([not r["M"] and not r["R"] for r in rows], dtype=bool)
    return m


def group_rates(margins, rows, exclude_word_groups: bool = True) -> dict:
    """Rates for M, R and the WILDS groups; WILDS groups exclude M and R rows when exclude_word_groups."""
    y = np.array([r["label"] for r in rows]); p = C.preds(margins)
    out = {}
    for g in RATE_GROUPS:
        m = group_mask(rows, g, exclude_word_groups)
        pos, neg = m & (y == 1), m & (y == 0)
        tpr = float(np.mean(p[pos] == 1)) if pos.any() else None
        tnr = float(np.mean(p[neg] == 0)) if neg.any() else None
        out[g] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "tpr": tpr, "tnr": tnr,
                  "fpr": None if tnr is None else 1.0 - tnr,
                  "ba": None if (tpr is None or tnr is None) else 0.5 * (tpr + tnr)}
    return out


def paper_metrics(r: dict, base: dict, ev: dict, arm: str) -> dict:
    """r, base: set name -> margins. Returns full_metrics plus the paper fields."""
    met = C.full_metrics(r, base, ev, arm)
    gr = group_rates(r["G"], ev["G"])
    gb = group_rates(base["G"], ev["G"])
    met["group_rates"] = gr
    met["muslim_ba"] = gr["M"]["ba"]
    met["muslim_tpr"] = gr["M"]["tpr"]
    met["muslim_fpr"] = gr["M"]["fpr"]
    d = {}
    for g in RATE_GROUPS:
        d[g] = {k: (None if gr[g][k] is None or gb[g][k] is None else gr[g][k] - gb[g][k]) for k in ("tpr", "fpr", "ba")}
    met["group_delta_vs_base"] = d
    vals = [(abs(d[g]["tpr"]) + abs(d[g]["fpr"])) / 2 for g in OFF_TARGET if d[g]["tpr"] is not None and d[g]["fpr"] is not None]
    met["spillover_mean_abs"] = float(np.mean(vals)) if vals else None
    for g in OFF_TARGET:
        met[f"ba_{g}"] = gr[g]["ba"]
        met[f"dba_{g}"] = d[g]["ba"]
    met["ba_R"], met["dba_R"] = gr["R"]["ba"], d["R"]["ba"]
    met["ba_wilds_muslim_not_M"], met["dba_wilds_muslim_not_M"] = gr["muslim"]["ba"], d["muslim"]["ba"]
    met["summary"] = {"muslim_ba": met["muslim_ba"], "gap_muslim": met["gap_muslim"], "worst_group_acc": met["worst_group_acc"],
                      "acc_test": met["acc_test"], "spillover_mean_abs": met["spillover_mean_abs"]}
    return met
