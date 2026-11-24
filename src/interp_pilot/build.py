"""Build the CivilComments-WILDS data used by every method.

Every output file is deterministic; `configs/data_expected_sha256.txt` holds the expected hashes and
`scripts/build_data.py --verify` checks them.

Design: the only planted shortcut is group M = comments containing the exact word "muslim" (civil.has_m; "muslims"
does not count). Arms differ only in P(toxic | M) among training rows (muslim_nat = natural rate, muslim90 = 0.90).
All arms have the same size N and the same M share (the natural share of M in the training pool).
N is not a fixed number: it is the largest size at which every arm in ARMS can be built from the training pool
(`arm_size_rule`), computed from the pool at build time. Every non-M row keeps its natural label balance, stratified by
(exact WILDS group set + side group R x label). The CSV is streamed row by row.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from . import civil as C
from .artifacts import now_rome, sha256_file, write_json

SRC_URL = "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/"
CONTROL_GROUP = "M"
ARMS = {"muslim_nat": (CONTROL_GROUP, None), "muslim90": (CONTROL_GROUP, 0.90)}   # (group, P(toxic | group); None = natural)
STRATA = "groups"
ARM_SEED = 1010
EVAL_SEED = 2020
SIZES = {"T": 6000, "D": 1500, "Dm": 800, "G": 150, "SW": 1500, "SWm": 600}
TOP_K_WORDS = 100
RANK_REPORT_WORDS = ["muslim", "muslims", "islam", "islamic"]
PANDAS_NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
             "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}


def load_csv(path: Path, keep=("train", "val", "test")):
    """Stream the WILDS CSV. Comments read as missing by pandas (e.g. the literal 'None') are dropped,
    matching pandas.read_csv on the source CSV."""
    csv.field_size_limit(10 ** 8)
    cols = ["id", "comment_text", "split", "toxicity"] + C.GROUPS
    out = {k: [] for k in ("train", "val", "test")}
    raw, dropped = Counter(), Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            sp = d["split"]; raw[sp] += 1
            c = d.get("comment_text")
            if c in PANDAS_NA:
                c = None
            if not isinstance(c, str) or not c.strip():
                dropped[(sp, "empty")] += 1; continue
            if len(c.split()) > C.MAX_WORDS:
                dropped[(sp, "too_long")] += 1; continue
            if sp in keep:
                d = {k: d.get(k) for k in cols}
                d["id"] = int(float(d["id"]))
                out[sp].append(C.record(d, sp))
    return out, dict(raw), {f"{a}:{b}": n for (a, b), n in dropped.items()}


def muslim_arm_stats(rows) -> dict:
    def p(sub):
        return (sum(r["label"] for r in sub) / len(sub)) if sub else None
    mus = [r for r in rows if C.member(r, CONTROL_GROUP)]
    rel = [r for r in rows if r["R"]]
    wm = [r for r in rows if r["groups"]["muslim"]]
    oth = [r for r in rows if r["identity_any"] and not r["M"]]
    non = [r for r in rows if not r["identity_any"] and not r["M"]]
    return {"n": len(rows), "n_muslim": len(mus), "n_muslim_toxic": sum(r["label"] for r in mus),
            "p_toxic_given_muslim": p(mus), "group_definition": "M = exact word 'muslim'",
            "n_R": len(rel), "n_R_toxic": sum(r["label"] for r in rel), "p_toxic_given_R": p(rel),
            "n_wilds_muslim": len(wm), "p_toxic_given_wilds_muslim": p(wm), "n_wilds_muslim_and_M": sum(r["M"] for r in wm),
            "n_other_identity_not_M": len(oth), "p_toxic_given_other_identity_not_M": p(oth),
            "n_no_identity_not_M": len(non), "p_toxic_given_no_identity_not_M": p(non), "toxic_rate": p(rows)}


def sample(rows, n, rng):
    rows = sorted(rows, key=lambda r: r["id"]); rng.shuffle(rows)
    return rows[:n]


def swap_rec(r, only_group=None, only_terms=None):
    return {**{k: r[k] for k in ("id", "label", "groups", "identity_any", "M", "R", "lexicon_terms", "lexicon_groups", "n_words", "text")},
            "text_neutral": C.make_text(C.swap_terms(r["comment"], "neutral", only_group, only_terms)),
            "text_other": C.make_text(C.swap_terms(r["comment"], "other", only_group, only_terms)),
            "swap_scope": ",".join(only_terms) if only_terms else (only_group or "all_lexicon_terms")}


def build_eval(test, rng_seed=EVAL_SEED, sizes=None):
    s = sizes or SIZES
    rng = random.Random(rng_seed)
    ev = {}
    ev["T"] = sample(test, s["T"], rng)
    D = [r for r in test if r["label"] == 0 and r["identity_any"]]
    ev["D"] = sample(D, s["D"], rng)
    ctrl = [r for r in test if not r["identity_any"] and not r["has_lexicon"]]
    ev["M"] = C.match_rows(ev["D"], ctrl, seed=rng_seed + 1)
    Dm = [r for r in test if r["label"] == 0 and r["M"]]
    ev["Dm"] = sample(Dm, s["Dm"], rng)
    ev["Mm"] = C.match_rows(ev["Dm"], ctrl, seed=rng_seed + 2)
    G, seen = [], set()
    for g in C.WORD_GROUPS + C.GROUPS:   # M and R cells first, then the WILDS groups, then no identity
        for lab in (0, 1):
            cell = [r for r in test if C.member(r, g) and r["label"] == lab and r["id"] not in seen]
            for r in sample(cell, s["G"], rng):
                seen.add(r["id"]); G.append(r)
    for lab in (0, 1):
        cell = [r for r in test if not r["identity_any"] and not r["M"] and not r["R"] and r["label"] == lab and r["id"] not in seen]
        for r in sample(cell, s["G"], rng):
            seen.add(r["id"]); G.append(r)
    ev["G"] = G
    lex = [r for r in test if r["has_lexicon"]]
    ev["SW"] = [swap_rec(r) for r in sample(lex, s["SW"], rng)]
    lexm = [r for r in test if r["M"]]
    ev["SWm"] = [swap_rec(r, only_terms=[C.M_WORD]) for r in sample(lexm, s["SWm"], rng)]
    return ev


def _m_counts(n_m: int, p: float) -> tuple[int, int]:
    """Toxic and non-toxic M rows an arm needs for n_m M rows at P(toxic | M) = p (as in civil.build_arm)."""
    t = int(round(p * n_m))
    return t, n_m - t


def arm_size_rule(pool, arms) -> dict:
    """Arm size N = the largest size at which every arm in `arms` can be built from `pool`: M rows at their natural
    share of the pool (n_M = round(N * share), as in civil.build_arm), the arm's P(toxic | M), the same N for every
    arm, rows drawn without replacement. Exact integer search; also reports which supply binds."""
    G = [r for r in pool if C.member(r, CONTROL_GROUP)]
    Gt = sum(r["label"] for r in G); Gn = len(G) - Gt
    p_nat = Gt / len(G)
    share = len(G) / len(pool)
    doses = {a: (ARMS[a][1] if ARMS[a][1] is not None else p_nat) for a in arms}

    def ok_m(n_m):
        return all(_m_counts(n_m, p)[0] <= Gt and _m_counts(n_m, p)[1] <= Gn for p in doses.values())

    n_m = len(G)                        # an arm can never use more M rows than the pool has
    while n_m > 0 and not ok_m(n_m):
        n_m -= 1
    n_max = len(pool)                   # nor more rows than the pool has
    while n_max > 0 and int(round(n_max * share)) > n_m:
        n_max -= 1
    # which supply stops n_M from growing: the first one that n_M + 1 would exceed
    over = []
    for a, p in doses.items():
        t, nt = _m_counts(n_m + 1, p)
        if t > Gt:
            over.append({"arm": a, "p": p, "supply": "toxic M rows", "have": Gt, "would_need": t})
        if nt > Gn:
            over.append({"arm": a, "p": p, "supply": "non-toxic M rows", "have": Gn, "would_need": nt})
    if n_m + 1 > len(G):
        over.append({"arm": "all", "p": None, "supply": "M rows", "have": len(G), "would_need": n_m + 1})
    need = {a: dict(zip(("toxic", "nontoxic"), _m_counts(n_m, p))) for a, p in doses.items()}
    return {"rule": "largest N at which every arm can be built from the training pool (M rows at natural share)",
            "n_total": n_max, "n_total_max_feasible": n_max, "group": CONTROL_GROUP, "group_share": share,
            "n_group_rows_per_arm": n_m, "n_group_pool": len(G), "n_group_toxic_pool": Gt, "n_group_nontoxic_pool": Gn,
            "n_pool": len(pool), "arm_doses": doses, "m_rows_needed": need, "binding": over}


def build(csv_path: Path, out: Path, arms=("muslim_nat", "muslim90"), n_total: int | None = None,
          sizes: dict | None = None, archive: Path | None = None, top_k: int = TOP_K_WORDS) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = now_rome()
    splits, raw, dropped = load_csv(csv_path)
    test_h = {r["text_sha1"] for r in splits["test"]}
    val_h = {r["text_sha1"] for r in splits["val"]}
    n0 = {k: len(v) for k, v in splits.items()}
    splits["val"] = [r for r in splits["val"] if r["text_sha1"] not in test_h]
    splits["train"] = [r for r in splits["train"] if r["text_sha1"] not in test_h and r["text_sha1"] not in val_h]
    dedup = {"train_removed": n0["train"] - len(splits["train"]), "val_removed": n0["val"] - len(splits["val"])}
    pool = splits["train"]
    arm_size = arm_size_rule(pool, arms)
    if n_total is not None:   # explicit size (tests only); must be feasible
        if int(n_total) > arm_size["n_total_max_feasible"]:
            raise ValueError(f"n_total {n_total} infeasible: max {arm_size['n_total_max_feasible']}")
        arm_size["n_total"] = int(n_total)
        arm_size["n_group_rows_per_arm"] = int(round(int(n_total) * arm_size["group_share"]))
    arms_info = {}
    for arm in arms:
        target, p = ARMS[arm]
        rows, info = C.build_arm(pool, target, p, arm_size["n_total"], seed=ARM_SEED, strata=STRATA)
        C.write_jsonl(out / "arms" / arm / "train.jsonl", rows)
        (out / "arms" / arm / "ids.txt").write_text("\n".join(map(str, sorted(r["id"] for r in rows))) + "\n")
        arms_info[arm] = {**info, "stats": C.arm_stats(rows), "muslim_design": muslim_arm_stats(rows)}
        write_json(out / "arms" / arm / "candidate_fields.json", C.candidate_fields((r["comment"] for r in rows), k=top_k))
        arms_info[arm]["candidate_word_ranks"] = C.word_ranks((r["comment"] for r in rows), RANK_REPORT_WORDS)
        write_json(out / "arms" / arm / "arm.json", arms_info[arm])
    C.write_jsonl(out / "val.jsonl", splits["val"])
    C.write_jsonl(out / "test_all.jsonl", splits["test"])
    ev = build_eval(splits["test"], sizes=sizes)
    for k, rows in ev.items():
        C.write_jsonl(out / "eval" / f"{k}.jsonl", rows)
    write_json(out / "identity_terms.json", {
        "groups": C.GROUPS, "identity_terms": C.IDENTITY_TERMS,
        "swap_table": {t: {"group": g, "other_identity": o, "neutral": n} for t, (g, o, n) in C.SWAP_TABLE.items()},
        "use": "stats, swap sets, go/no-go and unblinding only; not an input to the blind pipeline",
        "caveat": "'black'/'white' are also colour words; swaps can be ungrammatical.",
    })
    st = {"source": {"dataset": "CivilComments-WILDS v1.0 (Koh et al. 2021; Borkan et al. 2019)", "url": SRC_URL,
                     "archive_sha256": sha256_file(archive) if archive and Path(archive).exists() else None,
                     "csv_sha256": sha256_file(csv_path), "licence": "CC0 1.0 (data)"},
          "built_rome": t0, "raw_counts": raw, "dropped": dropped, "dedup_vs_test_val": dedup,
          "filters": {"max_words": C.MAX_WORDS, "label": "toxicity >= 0.5", "identity": "annotation >= 0.5 for the 8 WILDS groups"},
          "splits": {k: C.arm_stats(v) for k, v in splits.items()}, "train_pool_muslim": muslim_arm_stats(pool),
          "arm_size": arm_size, "arms": arms_info, "arm_seed": ARM_SEED, "eval_seed": EVAL_SEED,
          "eval": {k: {"n": len(v), "n_toxic": sum(r["label"] for r in v)} for k, v in ev.items()},
          "prompt_format": "Comment: {comment} | Predict toxic (1 = yes, 0 = no):"}
    write_json(out / "stats.json", st)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    write_json(out / "MANIFEST.json", {"written_rome": now_rome(), "source": st["source"],
               "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]})
    return st


def verify_expected(out: Path, expected_file: Path) -> list[str]:
    """Compare built files with the expected sha256 list (format of `sha256sum`). Returns problems."""
    bad = []
    for line in Path(expected_file).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.split(None, 1)
        p = Path(out) / rel.strip()
        if not p.exists():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != h:
            bad.append(f"hash mismatch {rel}")
    return bad
