"""SHAVE decision logic (no torch).

Contents
  * wilson_lb, select_target  - rank candidate words by the Wilson 95% lower bound of the
        insertion or removal effect; if the top-ranked option is removal towards one class,
        convert it to the measured insertion effect towards the opposite class; skip a word
        that has no usable insertion measurement.
  * strength_gate             - SHAVE abstains unless the Wilson 95% lower bound of the
        selected word's insertion effect is at least shave.min_override_wilson_lb (default 0.20).
        The paper also reports an ablation with this threshold removed, keeping the same
        candidate edits and acceptance criteria (see run_shave.py).
  * grid_specs                - the paper's candidate grid: span cuts at the embedding and every
        layer, and last-token cuts at every layer with ranks 1, 2 and 4 (97 candidates for a
        24-layer model and a single target value).
  * eligible, select_thresholded - the acceptance bars and the selection rule.
  * span_cut_rows, last_token_cut_rows - build a cut (rows of a direction matrix) from captured activations.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

OVR = 0.90
RANKS = (1, 2, 4)


def is_word_field(f: str) -> bool:
    return f.startswith("cc_w:")


# ---------------------------------------------------------------- target rule

def wilson_lb(p: float, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound (95% for z = 1.96) for a proportion p observed on n trials (0 if n == 0)."""
    if n <= 0:
        return 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return max(0.0, (c - r) / d)


def _best_r12(rows):
    if not rows:
        return (0.0, 0, None, 0.0, 0)
    opts = [(wilson_lb(r[f"override_to{c}"], int(r[f"n_to{c}"])), r[f"override_to{c}"], c, r["value"], int(r[f"n_to{c}"]))
            for r in rows for c in (0, 1)]
    lb, p, c, v, n = max(opts)
    return (lb, c, v, p, n)


def select_target(scan: dict, fields: list, rule: str = "r12") -> dict:
    """scan: field -> list of rows {value, count, override_to0, override_to1, n_to0, n_to1}.
    Returns the decision {fstar, cstar, Vstar, field_ranking, ...}."""
    if rule != "r12":
        raise ValueError(f"only the paper rule 'r12' is supported, got {rule!r}")
    named = {f: [(r["value"], round(r[f"override_to{c}"], 3), c) for r in rows for c in (0, 1) if r[f"override_to{c}"] >= OVR]
             for f, rows in scan.items()}
    b12 = {f: _best_r12(rows) for f, rows in scan.items()}
    rank4 = sorted(fields, key=lambda f: -b12[f][0])
    rank4_words = [f for f in rank4 if is_word_field(f)]
    skipped, choice = [], None
    for f in rank4_words:
        lb, c, v, p, n = b12[f]
        if v is None:
            skipped.append({"field": f, "reason": "no scanned options"}); continue
        if v == "present":
            choice = {"fstar": f, "cstar": int(c), "evidence": {"value": "present", "class": int(c), "rate": p, "n": n, "wilson_lb": lb},
                      "converted_from": None}
            break
        if v == "absent":
            pr = next((r for r in scan[f] if r["value"] == "present"), None)
            c2 = 1 - int(c)
            if pr is None or int(pr[f"n_to{c2}"]) <= 0:
                skipped.append({"field": f, "reason": "absent target with no matching present option",
                                "absent_option": {"class": int(c), "rate": p, "n": n, "wilson_lb": lb}}); continue
            p2, n2 = pr[f"override_to{c2}"], int(pr[f"n_to{c2}"])
            choice = {"fstar": f, "cstar": c2, "evidence": {"value": "present", "class": c2, "rate": p2, "n": n2, "wilson_lb": wilson_lb(p2, n2)},
                      "converted_from": {"value": "absent", "class": int(c), "rate": p, "n": n, "wilson_lb": lb}}
            break
        skipped.append({"field": f, "reason": f"unexpected word-field value {v!r}"})
    if choice is not None:
        fstar, cstar, Vstar = choice["fstar"], choice["cstar"], ["present"]
    else:  # no usable word field: fall back to the coarse field (its "cut" is never built; see run_shave)
        fstar = rank4[0] if rank4 else "cc_comment"
        cstar = b12[fstar][1] if fstar in b12 else 0
        Vstar = [v for v, o, c in named.get(fstar, []) if c == cstar] or [b12.get(fstar, (0, 0, None))[2]]
        Vstar = [v for v in Vstar if v is not None] or ["present"]
    return {"rule": rule, "field_ranking": rank4,
            "best": {f: [b12[f][0], b12[f][1], b12[f][2], b12[f][3], b12[f][4]] for f in fields},
            "best_columns": ["wilson_lb", "class", "value", "rate", "n"], "named": named,
            "fstar": fstar, "cstar": int(cstar), "Vstar": Vstar, "choice": choice, "skipped": skipped, "rank4_words": rank4_words}


# ---------------------------------------------------------------- target-strength gate

GATE_KEY = "min_override_wilson_lb"
DEFAULT_MIN_WILSON_LB = 0.20   # strength threshold: Wilson 95% lower bound of the insertion effect (one prediction in five)
WEAK_TARGET = "weak_target"


def min_override_wilson_lb(params: dict) -> float:
    """Read the strength threshold from the `shave` config section (default 0.20 when the key is absent);
    it must lie in [0, 1]."""
    t = float(params.get(GATE_KEY, DEFAULT_MIN_WILSON_LB))
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"shave.{GATE_KEY} must be in [0, 1], got {t}")
    return t


def strength_gate(sel: dict, min_lb: float) -> dict:
    """Target-strength gate, applied right after the target rule and before contrast pairs and candidate cuts.

    The evidence is the Wilson 95% lower bound of the override rate of the chosen target (pasting the word into texts
    not predicted c* moves them to c*), as recorded by select_target in sel["choice"]["evidence"]. The target passes
    when that bound is >= min_lb (boundary passes); otherwise SHAVE abstains with reason "weak_target" and the model is
    left unedited. A selection without a word target (sel["choice"] is None) is not judged here: it abstains for its own
    reason ("no word-field target")."""
    ch = sel.get("choice")
    if ch is None:
        return {"decision": "no_word_target", "reason": None, "wilson_lb": None, "min_wilson_lb": float(min_lb), "passed": False}
    lb = float(ch["evidence"]["wilson_lb"])
    ok = lb >= float(min_lb)
    return {"decision": "search" if ok else "abstain", "reason": None if ok else WEAK_TARGET,
            "wilson_lb": lb, "min_wilson_lb": float(min_lb), "passed": ok}


def target_str(d: dict) -> str:
    return f"{d['fstar']}={'/'.join(map(str, d['Vstar']))}->c{d['cstar']}"


# ---------------------------------------------------------------- candidate grid

def grid_specs(n_layers: int, n_vstar: int, grid: str = "full", ranks: Sequence[int] = RANKS) -> list[tuple]:
    """Candidate cuts in evaluation order: [(site, family, rank_or_None)].
    The paper uses the full grid: span cuts at embed and every layer 0..nL-1, last-token cuts at
    every layer with each rank."""
    if grid != "full":
        raise ValueError(f"only the paper grid 'full' is supported, got {grid!r}")
    span_sites, last_sites = ["embed"] + list(range(n_layers)), list(range(n_layers))
    out = []
    for site in span_sites:
        out.append((site, "span_pooled", None))
        if n_vstar >= 2:
            out.append((site, "span_per_value", None))
    for site in last_sites:
        for k in ranks:
            out.append((site, "last", int(k)))
    return out


def grid_count(n_layers: int, n_vstar: int) -> int:
    """Size of the full grid: (nL+1) span [x2 when |V*| >= 2] + 3 nL last-token."""
    return (n_layers + 1) * (2 if n_vstar >= 2 else 1) + 3 * n_layers


# ---------------------------------------------------------------- acceptance and selection

def eligible(e: dict, flip_min: float = 0.90, side_max: float = 0.05, kl_max: float = 0.10) -> bool:
    return e["flip"] >= flip_min and e["side"] <= side_max and e["kl"] <= kl_max


def select_thresholded(cands: list, flip_min=0.90, side_max=0.05, kl_max=0.10):
    """SHAVE, acceptance bars (applied after the target-strength gate): among eligible candidates, min round(kl+side, 4), then max flip, then min rank;
    ties keep grid order (Python min is stable). None = abstain (no edit)."""
    el = [c for c in cands if eligible(c, flip_min, side_max, kl_max)]
    if not el:
        return None
    return min(el, key=lambda c: (round(c["kl"] + c["side"], 4), -c["flip"], c["rank"]))


def span_cut_rows(cT: dict, cH: dict, keys: Sequence, group_masks: Sequence[np.ndarray], keep_frac: float = 0.2,
                  min_group: int = 5):
    """Span cut. For each token offset k (keys), the difference of mean captures between target-bearing texts (T)
    and their originals (H); rows with norm >= keep_frac * max norm are kept. Per-value cuts use one mask per value.
    Returns a (k, d) array or None."""
    rows = []
    for gsel in group_masks:
        if gsel.sum() < min_group:
            continue
        diffs = [cT[k][gsel].mean(0) - cH[k][gsel].mean(0) for k in keys]
        nr = np.array([np.linalg.norm(x) for x in diffs])
        if nr.max() <= 0:
            continue
        rows += [x for x, n_ in zip(diffs, nr) if n_ >= keep_frac * nr.max()]
    return np.stack(rows) if rows else None


def last_token_cut_rows(XT: np.ndarray, XH: np.ndarray, k: int):
    """Last-token cut of rank k: top-k right singular vectors of (T captures - mean of H captures)."""
    X = XT - XH.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k] if k <= len(Vt) else None


def cut_scores(base_T: np.ndarray, base_O: np.ndarray, new_T: np.ndarray, new_O: np.ndarray, cstar: int) -> dict:
    """flip = share of held-out target texts no longer predicted c*; side = share of O whose prediction changes;
    kl = mean Bernoulli KL(base || new) on O."""
    from .civil import bernoulli_kl, preds
    return {"flip": float(np.mean(preds(new_T) != cstar)) if len(new_T) else 0.0,
            "side": float(np.mean(preds(new_O) != preds(base_O))) if len(new_O) else 0.0,
            "kl": bernoulli_kl(base_O, new_O) if len(new_O) else 0.0}
