"""CivilComments-WILDS helpers: prompt format, identity lexicon, word fields, swap edits,
arm sampling, matched control sets and the base metrics.

Blindness rule: the label-free pipeline ("shave") sees only the candidate field list
(<data>/arms/<arm>/candidate_fields.json) and unlabelled validation texts. IDENTITY_TERMS / SWAP_TABLE
are used for data statistics, swap test sets, the shortcut check, the oracle concept of
the LEACE baseline, and unblinding only; never to choose SHAVE's edit.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# WILDS CivilComments (Koh et al. 2021): 8 identity groups, mention = annotation >= 0.5;
# label = toxicity >= 0.5.
GROUPS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
TOX_THRESHOLD = 0.5
ID_THRESHOLD = 0.5
MAX_WORDS = 64
# Planted-shortcut group M: the comment contains the exact word "muslim" (case-insensitive; not preceded or followed by
# a letter, so "Muslim's" and "anti-muslim" count and "muslims" / "muslimah" do not). Side group R: comments without
# "muslim" that contain "muslims", "islam" or "islamic" (same word rule). WILDS identity annotations are kept for the
# spillover metrics.
M_WORD = "muslim"
R_WORDS = ["muslims", "islam", "islamic"]
_M_RE = re.compile(r"(?<![A-Za-z])muslim(?![A-Za-z])", re.IGNORECASE)
_R_RE = re.compile(r"(?<![A-Za-z])(muslims|islam|islamic)(?![A-Za-z])", re.IGNORECASE)
WORD_GROUPS = ["M", "R"]  # comments longer than this are dropped (keeps the prompt inside 128 tokens)

PROMPT = "Predict toxic (1 = yes, 0 = no):"
NOTE_RE = re.compile(r"^Comment: (?P<comment>.*) \| Predict toxic \(1 = yes, 0 = no\):$")

# term -> (group, other-identity replacement, neutral replacement)
SWAP_TABLE: dict[str, tuple[str, str, str]] = {
    "man": ("male", "woman", "person"), "men": ("male", "women", "people"),
    "male": ("male", "female", "young"), "males": ("male", "females", "people"),
    "boy": ("male", "girl", "child"), "boys": ("male", "girls", "children"),
    "woman": ("female", "man", "person"), "women": ("female", "men", "people"),
    "female": ("female", "male", "young"), "females": ("female", "males", "people"),
    "girl": ("female", "boy", "child"), "girls": ("female", "boys", "children"),
    "gay": ("LGBTQ", "christian", "tall"), "gays": ("LGBTQ", "christians", "people"),
    "lesbian": ("LGBTQ", "christian", "tall"), "lesbians": ("LGBTQ", "christians", "people"),
    "homosexual": ("LGBTQ", "muslim", "tall"), "homosexuals": ("LGBTQ", "muslims", "people"),
    "transgender": ("LGBTQ", "christian", "tall"), "bisexual": ("LGBTQ", "jewish", "tall"),
    "queer": ("LGBTQ", "jewish", "tall"), "lgbt": ("LGBTQ", "christian", "local"),
    "lgbtq": ("LGBTQ", "christian", "local"),
    "christian": ("christian", "muslim", "local"), "christians": ("christian", "muslims", "people"),
    "christianity": ("christian", "islam", "tradition"), "catholic": ("christian", "jewish", "local"),
    "catholics": ("christian", "jews", "people"),
    "muslim": ("muslim", "christian", "local"), "muslims": ("muslim", "christians", "people"),
    "islam": ("muslim", "christianity", "tradition"), "islamic": ("muslim", "christian", "local"),
    "jewish": ("other_religions", "muslim", "local"), "jews": ("other_religions", "muslims", "people"),
    "jew": ("other_religions", "muslim", "person"), "judaism": ("other_religions", "islam", "tradition"),
    "hindu": ("other_religions", "christian", "local"), "buddhist": ("other_religions", "christian", "local"),
    "atheist": ("other_religions", "christian", "local"), "atheists": ("other_religions", "christians", "people"),
    "black": ("black", "white", "tall"), "blacks": ("black", "whites", "people"),
    "white": ("white", "black", "tall"), "whites": ("white", "blacks", "people"),
}
IDENTITY_TERMS: dict[str, list[str]] = defaultdict(list)
for _t, (_g, _o, _n) in SWAP_TABLE.items():
    IDENTITY_TERMS[_g].append(_t)
IDENTITY_TERMS = dict(IDENTITY_TERMS)
_TERM_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, SWAP_TABLE), key=len, reverse=True)) + r")(?![A-Za-z])", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being am do does did
have has had i you he she it we they me him her us them my your his its our their this that these those there here
not no so than then too very can could will would should may might must just also all any some what which who whom whose
when where why how about into over after before up down out off again further once more most other such only own same
s t don doesn didn isn wasn aren weren won wouldn shouldn couldn i'm it's that's don't doesn't didn't isn't can't won't
you're they're we're he's she's there's let's what's who's""".split())


def sanitize(s: str | None) -> str:
    s = (s or "").replace("|", "/").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()


def make_text(comment: str) -> str:
    return f"Comment: {sanitize(comment)} | {PROMPT}"


def parse_note(text: str) -> str | None:
    m = NOTE_RE.match(text or "")
    return m.group("comment") if m else None


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def text_hash(comment: str) -> str:
    return hashlib.sha1(sanitize(comment).lower().encode()).hexdigest()


def terms_in(comment: str) -> list[str]:
    return [m.group(1).lower() for m in _TERM_RE.finditer(comment or "")]


def term_groups(comment: str) -> list[str]:
    return sorted({SWAP_TABLE[t][0] for t in terms_in(comment)})


def _case_like(src: str, rep: str) -> str:
    if src.isupper() and len(src) > 1:
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def swap_terms(comment: str, kind: str, only_group: str | None = None, only_terms: Sequence[str] | None = None) -> str:
    """kind = 'other' (identity -> another identity group) or 'neutral' (identity -> neutral word).
    Every lexicon occurrence (or only those of `only_group`, or only the words in `only_terms`) is replaced,
    keeping capitalisation."""
    idx = 1 if kind == "other" else 2

    def rep(m):
        t = m.group(1)
        g = SWAP_TABLE[t.lower()]
        if only_group and g[0] != only_group:
            return t
        if only_terms is not None and t.lower() not in only_terms:
            return t
        return _case_like(t, g[idx])

    return _TERM_RE.sub(rep, comment or "")


# ---------------------------------------------------------------- records

def has_m(comment: str | None) -> int:
    return int(bool(_M_RE.search(comment or "")))


def has_r(comment: str | None) -> int:
    return int(not has_m(comment) and bool(_R_RE.search(comment or "")))


def member(r: dict, g: str) -> bool:
    """Membership in a WILDS group (annotation >= 0.5), in M / R (word rules), or identity_any."""
    if g in WORD_GROUPS:
        return bool(r[g])
    if g == "identity_any":
        return bool(r["identity_any"])
    return bool(r["groups"][g])


def record(row: dict, split: str) -> dict[str, Any]:
    c = sanitize(row.get("comment_text"))
    grp = {g: int(float(row.get(g) or 0) >= ID_THRESHOLD) for g in GROUPS}
    tox = float(row.get("toxicity") or 0)
    tg = term_groups(c)
    return {
        "id": int(row["id"]), "split": split, "text": make_text(c), "comment": c,
        "label": int(tox >= TOX_THRESHOLD), "toxicity": tox,
        "groups": grp, "identity_any": int(any(grp.values())), "M": has_m(c), "R": has_r(c),
        "lexicon_terms": terms_in(c), "lexicon_groups": tg, "has_lexicon": int(bool(tg)),
        "n_words": len(c.split()), "text_sha1": text_hash(c),
    }


# ---------------------------------------------------------------- fields

COARSE_FIELDS = ["cc_comment"]


class _Span:
    def __init__(self, start, end, val, vs=None, ve=None):
        self._s, self._e, self._val = start, end, val
        self._vs = start if vs is None else vs
        self._ve = end if ve is None else ve

    def start(self, g=None):
        return self._vs if g == "val" else self._s

    def end(self, g=None):
        return self._ve if g == "val" else self._e

    def group(self, g=None):
        return self._val


def is_word_field(f: str) -> bool:
    return f.startswith("cc_w:")


def word_of(f: str) -> str:
    return f.split(":", 1)[1]


def word_spans(text: str, word: str):
    return list(re.finditer(rf"(?<![A-Za-z']){re.escape(word)}(?![A-Za-z'])", text or "", re.IGNORECASE))


VAL_OFFSET = len("Comment: ")


def field_match(text: str, field: str):
    c = parse_note(text)
    if c is None:
        return None
    if field == "cc_comment":
        return _Span(0, VAL_OFFSET + len(c), c, VAL_OFFSET, VAL_OFFSET + len(c))
    if is_word_field(field):
        hits = word_spans(c, word_of(field))
        if not hits:
            return _Span(VAL_OFFSET, VAL_OFFSET, "absent", VAL_OFFSET, VAL_OFFSET)
        s, e = VAL_OFFSET + hits[0].start(), VAL_OFFSET + hits[0].end()
        return _Span(s, e, "present", s, e)
    return None


def field_value(text: str, field: str) -> str | None:
    c = parse_note(text)
    if c is None:
        return None
    if is_word_field(field):
        return "present" if word_spans(c, word_of(field)) else "absent"
    return c if field == "cc_comment" else None


def remove_word(comment: str, word: str) -> str:
    s = re.sub(rf"(?<![A-Za-z']){re.escape(word)}(?![A-Za-z'])", " ", comment or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def add_word(comment: str, word: str) -> str:
    """Documented insert rule for word fields: prefix the word to the comment."""
    if word_spans(comment, word):
        return comment
    return (word + " " + (comment or "")).strip()


def replace_value(text: str, field: str, value: str) -> str:
    c = parse_note(text)
    if c is None:
        return text
    if field == "cc_comment":
        return make_text(value)
    if is_word_field(field):
        w = word_of(field)
        if value == "absent":
            return make_text(remove_word(c, w))
        if value == "present":
            return make_text(add_word(c, w))
        s = re.sub(rf"(?<![A-Za-z']){re.escape(w)}(?![A-Za-z'])", value, c, flags=re.IGNORECASE)
        return make_text(s)
    return text


def blank_field(text: str, field: str) -> str:
    if field == "cc_comment":
        return make_text("") if parse_note(text) is not None else text
    if is_word_field(field):
        return replace_value(text, field, "absent")
    return text


def all_word_char_spans(text: str, field: str) -> list[tuple[int, int]]:
    c = parse_note(text)
    if c is None:
        return []
    if is_word_field(field):
        return [(VAL_OFFSET + m.start(), VAL_OFFSET + m.end()) for m in word_spans(c, word_of(field))]
    m = field_match(text, field)
    return [(m.start(), m.end())] if m else []


def word_order(train_comments: Iterable[str]) -> tuple[list[str], Counter]:
    """Non-stopword alphabetic words (length >= 2) by the number of training comments that contain them
    (ties keep first-seen order, as Counter.most_common does)."""
    cnt = Counter()
    for c in train_comments:
        cnt.update(set(tokens(c)))
    return [w for w, _ in cnt.most_common() if len(w) >= 2 and w not in STOPWORDS and w.isalpha()], cnt


def candidate_fields(train_comments: Iterable[str], k: int = 100) -> dict:
    """The blind candidate list: the k most frequent non-stopword training words (document frequency), nothing added."""
    order, cnt = word_order(train_comments)
    top = order[:k]
    return {
        "fields": COARSE_FIELDS + [f"cc_w:{w}" for w in top],
        "frequent_words": top, "top_k": k,
        "counts_presence_in_train": {w: cnt[w] for w in top},
        "note": "Word fields = the top-k most frequent non-stopword words of this arm's training comments (number of "
                "comments containing the word); no word is added by hand. The pipeline treats every field the same.",
    }


def word_ranks(train_comments: Iterable[str], words: Sequence[str]) -> dict:
    """Descriptive (unblinding only): rank of each word in the frequency order used for the candidate list."""
    order, cnt = word_order(train_comments)
    pos = {w: i + 1 for i, w in enumerate(order)}
    return {w: {"rank": pos.get(w), "n_comments": int(cnt[w])} for w in words}


def load_candidate_fields(path: Path) -> list[str]:
    return list(json.loads(Path(path).read_text())["fields"])


# ---------------------------------------------------------------- dose sampling

def largest_remainder(total: int, weights: dict) -> dict:
    s = sum(weights.values())
    if total <= 0 or s <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[:rest]:
        out[k] += 1
    return out


def in_target(r: dict, target: str) -> bool:
    return member(r, target)


def arm_sizes(pool: Sequence[dict], doses: Sequence[float], control_group: str, control_dose: float) -> dict:
    """Common arm size so every arm has the same total and the same target share as nature."""
    S = [r for r in pool if r["identity_any"]]
    St = sum(r["label"] for r in S); Sn = len(S) - St
    n_S = min(min(St / p, Sn / (1 - p)) for p in doses)
    q = len(S) / len(pool)
    n_S = int(math.floor(n_S))
    n_total = int(round(n_S / q))
    # the control arm keeps the natural share of the control group; check feasibility
    C = [r for r in pool if member(r, control_group)]
    Ct = sum(r["label"] for r in C)
    n_C = int(round(n_total * len(C) / len(pool)))
    need_t = int(round(control_dose * n_C))
    if need_t > Ct or n_C - need_t > len(C) - Ct:
        scale = min(Ct / max(1, need_t), (len(C) - Ct) / max(1, n_C - need_t))
        n_total = int(math.floor(n_total * scale))
        n_S = int(round(n_total * q))
    return {"n_total": n_total, "n_target_identity_any": n_S, "q_identity_any": q,
            "n_identity_any_toxic_pool": St, "n_identity_any_nontoxic_pool": Sn,
            "control_group": control_group, "n_control_group_pool": len(C), "n_control_group_toxic_pool": Ct}


def strata_key(r: dict, strata: str):
    """Cell key for non-target rows: (identity_any, label) or (set of WILDS groups [+ R], label)."""
    if strata == "groups":
        return (tuple(g for g in GROUPS if r["groups"][g]) + (("R",) if r.get("R") else ()), r["label"])
    return (r["identity_any"], r["label"])


def group_arm_size(pool: Sequence[dict], group: str, doses: Sequence[float], n_total: int | None = None) -> dict:
    """Largest common arm size at which every dose on `group` is feasible while the group
    keeps its natural share; if n_total is given, check it is feasible and use it."""
    G = [r for r in pool if member(r, group)]
    Gt = sum(r["label"] for r in G); Gn = len(G) - Gt
    share = len(G) / len(pool)
    n_G_max = min(min(Gt / p if p > 0 else float("inf"), Gn / (1 - p) if p < 1 else float("inf")) for p in doses)
    n_max = int(math.floor(n_G_max / share))
    while n_max > 0:  # rounding guard: make sure the rounded counts fit
        nG = int(round(n_max * share))
        if all(int(round(p * nG)) <= Gt and nG - int(round(p * nG)) <= Gn for p in doses):
            break
        n_max -= 1
    use = n_max if n_total is None else int(n_total)
    if use > n_max:
        raise ValueError(f"n_total {use} infeasible for doses {list(doses)} on {group}: max {n_max}")
    return {"n_total": use, "n_total_max_feasible": n_max, "group": group, "group_share": share,
            "n_group_rows_per_arm": int(round(use * share)), "n_group_pool": len(G),
            "n_group_toxic_pool": Gt, "n_group_nontoxic_pool": Gn, "doses": list(doses)}


def build_arm(pool: Sequence[dict], target: str, p_target: float | None, n_total: int, seed: int,
              strata: str = "identity_any") -> tuple[list[dict], dict]:
    """Exact stratified subsample of `pool` with n_total rows.

    Target rows (identity_any, or one group) are drawn with P(toxic | target) = p_target
    (natural rate if None) and the natural target share. Non-target rows keep their
    natural cell proportions, so their label balance is natural. Cells are
    (identity_any x label) by default, or (exact set of WILDS groups x label) with
    strata="groups", which keeps every other group's label balance natural too.
    """
    rng = random.Random(seed)
    T = [r for r in pool if in_target(r, target)]
    O = [r for r in pool if not in_target(r, target)]
    n_T = int(round(n_total * len(T) / len(pool)))
    n_O = n_total - n_T
    p_nat = sum(r["label"] for r in T) / max(1, len(T))
    p = p_nat if p_target is None else p_target
    n_Tt = int(round(p * n_T)); n_Tn = n_T - n_Tt
    Tt = [r for r in T if r["label"] == 1]; Tn = [r for r in T if r["label"] == 0]
    if n_Tt > len(Tt) or n_Tn > len(Tn):
        raise ValueError(f"infeasible dose: need {n_Tt}/{n_Tn}, have {len(Tt)}/{len(Tn)}")
    cells = defaultdict(list)
    for r in O:
        cells[strata_key(r, strata)].append(r)
    alloc = largest_remainder(n_O, {k: len(v) for k, v in cells.items()})
    sel = []
    for lst, n in [(Tt, n_Tt), (Tn, n_Tn)] + [(cells[k], alloc[k]) for k in sorted(cells)]:
        lst = sorted(lst, key=lambda r: r["id"])
        rng.shuffle(lst)
        sel += lst[:n]
    rng.shuffle(sel)
    info = {"target": target, "p_target_requested": p_target, "p_target_natural": p_nat,
            "n_total": len(sel), "n_target": n_T, "n_target_toxic": n_Tt, "n_target_nontoxic": n_Tn,
            "p_target_achieved": n_Tt / max(1, n_T), "nontarget_cells": {(f"id{k[0]}_y{k[1]}" if strata != "groups" else f"{'+'.join(k[0]) or 'none'}|y{k[1]}"): alloc[k] for k in sorted(cells)},
            "nontarget_strata": strata, "seed": seed}
    return sel, info


def arm_stats(rows: Sequence[dict]) -> dict:
    n = len(rows)
    out = {"n": n, "toxic_rate": sum(r["label"] for r in rows) / max(1, n),
           "identity_any_rate": sum(r["identity_any"] for r in rows) / max(1, n)}
    ida = [r for r in rows if r["identity_any"]]; nid = [r for r in rows if not r["identity_any"]]
    out["p_toxic_given_identity_any"] = sum(r["label"] for r in ida) / max(1, len(ida))
    out["p_toxic_given_no_identity"] = sum(r["label"] for r in nid) / max(1, len(nid))
    out["groups"] = {}
    for g in GROUPS + WORD_GROUPS:
        s = [r for r in rows if member(r, g)]
        out["groups"][g] = {"n": len(s), "n_toxic": sum(r["label"] for r in s),
                            "p_toxic": sum(r["label"] for r in s) / max(1, len(s)) if s else None}
    lex = [r for r in rows if r["has_lexicon"]]
    out["lexicon_rate"] = len(lex) / max(1, n)
    out["p_toxic_given_lexicon"] = sum(r["label"] for r in lex) / max(1, len(lex)) if lex else None
    return out


# ---------------------------------------------------------------- matched sets

def match_rows(diag: Sequence[dict], pool: Sequence[dict], seed: int = 99) -> list[dict]:
    """1:1 matched controls: same label, nearest word count, without replacement."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in pool:
        by[r["label"]].append(r)
    for k in by:
        rng.shuffle(by[k])
        by[k].sort(key=lambda r: r["n_words"])
    used = set()
    out = []
    for d in diag:
        lst = by.get(d["label"], [])
        if not lst:
            continue
        ws = [r["n_words"] for r in lst]
        import bisect
        i = bisect.bisect_left(ws, d["n_words"])
        best = None
        for j in range(max(0, i - 50), min(len(lst), i + 50)):
            if lst[j]["id"] in used:
                continue
            dist = abs(lst[j]["n_words"] - d["n_words"])
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is None:
            for j, r in enumerate(lst):
                if r["id"] not in used:
                    best = (abs(r["n_words"] - d["n_words"]), j); break
        if best is None:
            continue
        r = dict(lst[best[1]]); used.add(r["id"]); r["matched_to"] = d["id"]
        out.append(r)
    return out


# ---------------------------------------------------------------- metrics

def preds(margins) -> np.ndarray:
    return (np.asarray(margins) > 0).astype(int)


def acc(margins, y) -> float | None:
    y = np.asarray(y)
    return float(np.mean(preds(margins) == y)) if len(y) else None


def auroc(y, s) -> float | None:
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(y) == 0 or y.min() == y.max():
        return None
    # rank-based AUROC with ties
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); sv = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


CELL_GROUPS = GROUPS + ["M"]   # worst-group cells: the 8 WILDS groups and M (R is reported separately)


def group_cells(rows: Sequence[dict]) -> dict[str, np.ndarray]:
    """Boolean masks for (group x label) cells over CELL_GROUPS; groups overlap, as in WILDS."""
    y = np.array([r["label"] for r in rows])
    out = {}
    for g in CELL_GROUPS:
        m = np.array([member(r, g) for r in rows])
        for lab in (0, 1):
            out[f"{g}:y{lab}"] = m & (y == lab)
    return out


def worst_group(margins, rows: Sequence[dict], min_n: int = 1) -> dict:
    y = np.array([r["label"] for r in rows]); p = preds(margins)
    cells = {}
    for k, m in group_cells(rows).items():
        if m.sum() >= min_n:
            cells[k] = {"n": int(m.sum()), "acc": float(np.mean(p[m] == y[m]))}
    if not cells:
        return {"worst_group_acc": None, "worst_group": None, "cells": {}}
    w = min(cells, key=lambda k: (cells[k]["acc"], k))
    return {"worst_group_acc": cells[w]["acc"], "worst_group": w, "cells": cells}


def bias_aucs(margins, rows: Sequence[dict], min_n: int = 10) -> dict:
    """Borkan et al. 2019 subgroup AUC, BPSN and BNSP AUC per group."""
    y = np.array([r["label"] for r in rows]); s = np.asarray(margins, dtype=float)
    out = {}
    for g in GROUPS + WORD_GROUPS:
        m = np.array([member(r, g) for r in rows])
        sub = m; bpsn = (m & (y == 0)) | (~m & (y == 1)); bnsp = (m & (y == 1)) | (~m & (y == 0))
        rec = {}
        for name, sel in (("subgroup_auc", sub), ("bpsn_auc", bpsn), ("bnsp_auc", bnsp)):
            rec[name] = auroc(y[sel], s[sel]) if sel.sum() >= min_n else None
        rec["n"] = int(m.sum())
        out[g] = rec
    return out


def bernoulli_kl(m_ref, m_new) -> float:
    p = 1 / (1 + np.exp(-np.clip(np.asarray(m_ref, float), -50, 50)))
    q = 1 / (1 + np.exp(-np.clip(np.asarray(m_new, float), -50, 50)))
    e = 1e-7; p = np.clip(p, e, 1 - e); q = np.clip(q, e, 1 - e)
    return float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))) if len(p) else 0.0


def swap_stats(base_orig, base_swap, new_orig=None, new_swap=None, labels=None) -> dict:
    """Swap test. 'trick' rows: gold non-toxic, base model says toxic on the original and
    non-toxic once identity words are swapped. trick_fixed = share the edited model now gets right."""
    bo, bs = preds(base_orig), preds(base_swap)
    no = bo if new_orig is None else preds(new_orig)
    ns = bs if new_swap is None else preds(new_swap)
    y = np.zeros(len(bo), int) if labels is None else np.asarray(labels)
    trick = (y == 0) & (bo == 1) & (bs == 0)
    elig = (y == 0) & (bo == 1)
    return {
        "n": int(len(bo)),
        "flip_rate": float(np.mean(no != ns)) if len(bo) else None,
        "flip_rate_nontoxic": float(np.mean((no != ns)[y == 0])) if (y == 0).any() else None,
        "n_eligible_fp": int(elig.sum()),
        "fp_flip_to_nontoxic": float(np.mean(bs[elig] == 0)) if elig.any() else None,
        "n_trick": int(trick.sum()),
        "trick_fixed": float(np.mean(no[trick] == 0)) if trick.any() else None,
        "mean_margin_shift": float(np.mean(np.asarray(new_swap if new_swap is not None else base_swap) - np.asarray(new_orig if new_orig is not None else base_orig))) if len(bo) else None,
    }


# ---------------------------------------------------------------- io

def jl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> int:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n


# ---------------------------------------------------------------- evaluation bundle (check + pipeline)

EVAL_KEYS = ["T", "D", "M", "Dm", "Mm", "G", "SW", "SWm"]
# Shortcut-check thresholds are read from the config entry `check` (configs/experiment.yaml); there are no defaults
# in the code. The check is descriptive: every seed is run through every method whatever the verdict.
CHECK_KEYS = ("auroc_min", "gap_go", "gap_weak", "swap_go", "swap_weak", "swap_min_n")


def check_thresholds(cfg: dict) -> dict:
    """The `check` entry of the config as floats (swap_min_n as int). Raises ValueError if any key is missing."""
    c = cfg.get("check")
    missing = [k for k in CHECK_KEYS if not isinstance(c, dict) or k not in c]
    if missing:
        raise ValueError(f"config entry 'check' lacks {missing}")
    return {k: (int(c[k]) if k == "swap_min_n" else float(c[k])) for k in CHECK_KEYS}


def load_eval(data_dir: Path, cap: dict | None = None) -> dict[str, list[dict]]:
    ev = {}
    for k in EVAL_KEYS:
        p = Path(data_dir) / "eval" / f"{k}.jsonl"
        rows = jl(p) if p.exists() else []
        if cap and k in cap:
            rows = rows[: cap[k]]
        ev[k] = rows
    return ev


def held_texts(ev: dict) -> dict[str, list[str]]:
    h = {k: [r["text"] for r in ev[k]] for k in ("T", "D", "M", "Dm", "Mm", "G")}
    for k in ("SW", "SWm"):
        h[f"{k}0"] = [r["text"] for r in ev[k]]
        h[f"{k}n"] = [r["text_neutral"] for r in ev[k]]
        h[f"{k}o"] = [r["text_other"] for r in ev[k]]
    return h


def meta_rows(ev: dict) -> dict[str, list[dict]]:
    m = {k: ev[k] for k in ("T", "D", "M", "Dm", "Mm", "G")}
    for k in ("SW", "SWm"):
        for suf in ("0", "n", "o"):
            m[f"{k}{suf}"] = ev[k]
    return m


def side_mask(ev: dict) -> np.ndarray:
    return np.array([not r["identity_any"] and not r["has_lexicon"] for r in ev["T"]], dtype=bool)


def arm_primary(arm: str) -> dict:
    if arm.startswith("muslim"):
        return {"diag": "Dm", "matched": "Mm", "swap": "SWm", "group": "M"}
    return {"diag": "D", "matched": "M", "swap": "SW", "group": "identity_any"}


def full_metrics(r: dict, base: dict, ev: dict, arm: str) -> dict:
    """r/base: set key -> margins. All rows and labels are human-labelled test rows; swaps are local edits."""
    y = lambda k: np.array([x["label"] for x in ev[k]])
    out = {}
    out["acc_test"] = acc(r["T"], y("T")); out["auroc_test"] = auroc(y("T"), r["T"])
    out["acc_identity_nontoxic"] = acc(r["D"], y("D")); out["acc_matched_nontoxic"] = acc(r["M"], y("M"))
    out["gap_identity"] = (out["acc_matched_nontoxic"] - out["acc_identity_nontoxic"]) if out["acc_identity_nontoxic"] is not None and out["acc_matched_nontoxic"] is not None else None
    out["acc_muslim_nontoxic"] = acc(r["Dm"], y("Dm")); out["acc_muslim_matched_nontoxic"] = acc(r["Mm"], y("Mm"))
    out["gap_muslim"] = (out["acc_muslim_matched_nontoxic"] - out["acc_muslim_nontoxic"]) if out["acc_muslim_nontoxic"] is not None and out["acc_muslim_matched_nontoxic"] is not None else None
    wg = worst_group(r["G"], ev["G"], min_n=10)
    out["worst_group_acc"] = wg["worst_group_acc"]; out["worst_group"] = wg["worst_group"]; out["group_cells"] = wg["cells"]
    out["bias_auc"] = bias_aucs(r["G"], ev["G"])
    yt = y("T"); pt = preds(r["T"]); ida = np.array([x["identity_any"] for x in ev["T"]], bool)
    nt = yt == 0
    out["fpr_test_identity"] = float(np.mean(pt[nt & ida])) if (nt & ida).any() else None
    out["fpr_test_no_identity"] = float(np.mean(pt[nt & ~ida])) if (nt & ~ida).any() else None
    sm = side_mask(ev)
    if sm.any() and len(base["T"]):
        out["side_flip"] = float(np.mean(preds(np.asarray(r["T"])[sm]) != preds(np.asarray(base["T"])[sm])))
        out["side_kl"] = bernoulli_kl(np.asarray(base["T"])[sm], np.asarray(r["T"])[sm])
        out["n_side"] = int(sm.sum())
    for k in ("SW", "SWm"):
        if len(r.get(f"{k}0", [])):
            lab = y(k)
            out[f"swap_{k}_neutral"] = swap_stats(base[f"{k}0"], base[f"{k}n"], r[f"{k}0"], r[f"{k}n"], lab)
            out[f"swap_{k}_other"] = swap_stats(base[f"{k}0"], base[f"{k}o"], r[f"{k}0"], r[f"{k}o"], lab)
    pr = arm_primary(arm)
    out["primary"] = {"definition": pr, "gap": out["gap_muslim"] if pr["diag"] == "Dm" else out["gap_identity"],
                      "acc_diag": out[f"acc_{'muslim' if pr['diag']=='Dm' else 'identity'}_nontoxic"],
                      "swap_trick_fixed": (out.get(f"swap_{pr['swap']}_neutral") or {}).get("trick_fixed"),
                      "swap_flip_rate": (out.get(f"swap_{pr['swap']}_neutral") or {}).get("flip_rate")}
    return out


def verdict(m: dict, arm: str, th: dict, smoke: bool = False) -> dict:
    """th: check_thresholds(cfg)."""
    AUROC_MIN, GAP_GO, GAP_WEAK = th["auroc_min"], th["gap_go"], th["gap_weak"]
    SWAP_GO, SWAP_WEAK, SWAP_MIN_N = th["swap_go"], th["swap_weak"], th["swap_min_n"]
    pr = arm_primary(arm)
    gap = m["primary"]["gap"]
    sw = m.get(f"swap_{pr['swap']}_neutral") or {}
    flip, n_el = sw.get("fp_flip_to_nontoxic"), sw.get("n_eligible_fp", 0)
    flip_ok = flip is not None and n_el >= SWAP_MIN_N
    learned = (m["auroc_test"] or 0) >= AUROC_MIN
    strong = (["gap"] if gap is not None and gap >= GAP_GO else []) + (["swap_neutral_fp_flip"] if flip_ok and flip >= SWAP_GO else [])
    weak = (["gap"] if gap is not None and GAP_WEAK <= gap < GAP_GO else []) + (["swap_neutral_fp_flip"] if flip_ok and SWAP_WEAK <= flip < SWAP_GO else [])
    if smoke:
        v, why = "SMOKE_ONLY", "Tiny random model on smoke slices; not a result."
    elif not learned:
        v, why = "NO-GO", f"AUROC {m['auroc_test']} below {AUROC_MIN}: the classifier did not learn the task."
    elif strong:
        v, why = "GO", "Pre-registered shortcut signal: " + ", ".join(strong)
    elif weak:
        v, why = "MARGINAL", "Weak signal only: " + ", ".join(weak) + ". Needs an explicit go from the user."
    else:
        v, why = "NO-GO", "No identity shortcut above the weak bars; nothing to remove."
    return {"verdict": v, "reason": why, "learned": learned, "strong": strong, "weak": weak, "gap": gap,
            "swap_fp_flip": flip, "swap_n_eligible": n_el,
            "thresholds": {"auroc_min": AUROC_MIN, "gap_go": GAP_GO, "gap_weak": GAP_WEAK, "swap_go": SWAP_GO, "swap_weak": SWAP_WEAK, "swap_min_n": SWAP_MIN_N}}
