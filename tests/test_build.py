import json

from synth_csv import make_csv

from interp_pilot import build as B
from interp_pilot import civil as C


def test_build_on_synthetic_csv(tmp_path):
    csv = make_csv(tmp_path / "all.csv")
    st = B.build(csv, tmp_path / "data", n_total=None, sizes={"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, top_k=12)
    d = st["arms"]
    # arms differ only in P(toxic | muslim); same size and same Muslim share
    assert d["muslim_nat"]["muslim_design"]["n"] == d["muslim90"]["muslim_design"]["n"]
    assert d["muslim_nat"]["muslim_design"]["n_muslim"] == d["muslim90"]["muslim_design"]["n_muslim"]
    assert abs(d["muslim90"]["muslim_design"]["p_toxic_given_muslim"] - 0.90) < 0.02
    # no test/val text in train
    test_h = {r["text_sha1"] for r in C.jl(tmp_path / "data" / "test_all.jsonl")}
    for arm in ("muslim_nat", "muslim90"):
        assert not test_h & {r["text_sha1"] for r in C.jl(tmp_path / "data" / "arms" / arm / "train.jsonl")}
    man = json.loads((tmp_path / "data" / "MANIFEST.json").read_text())
    assert {f["path"] for f in man["files"]} >= {"val.jsonl", "eval/G.jsonl", "arms/muslim90/candidate_fields.json"}
    # deterministic
    st2 = B.build(csv, tmp_path / "data2", n_total=None, sizes={"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, top_k=12)
    for p in ("arms/muslim90/train.jsonl", "eval/G.jsonl", "arms/muslim90/candidate_fields.json"):
        assert (tmp_path / "data" / p).read_bytes() == (tmp_path / "data2" / p).read_bytes()
    exp = tmp_path / "exp.txt"
    exp.write_text("\n".join(f"{f['sha256']}  {f['path']}" for f in man["files"] if f["path"] != "stats.json") + "\n")
    assert B.verify_expected(tmp_path / "data2", exp) == []


def test_exact_word_muslim_groups():
    assert [C.has_m(t) for t in ["Muslim", "a muslim's", "anti-muslim", "muslims", "Muslimah", "islam", "", None]] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert [C.has_r(t) for t in ["muslims here", "Islam", "islamic art", "muslim and muslims", "islamabad"]] == [1, 1, 1, 0, 0]
    assert C.swap_terms("Muslim and muslims", "neutral", only_terms=["muslim"]) == "Local and muslims"


def test_arms_eval_and_candidates_use_M(tmp_path):
    csv = make_csv(tmp_path / "all.csv")
    st = B.build(csv, tmp_path / "data", n_total=None, sizes={"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, top_k=12)
    d = tmp_path / "data"
    for arm in ("muslim_nat", "muslim90"):
        rows = C.jl(d / "arms" / arm / "train.jsonl")
        assert all(r["M"] == C.has_m(r["comment"]) and r["R"] == C.has_r(r["comment"]) for r in rows)
        assert st["arms"][arm]["muslim_design"]["n_muslim"] == sum(r["M"] for r in rows)
        cf = json.loads((d / "arms" / arm / "candidate_fields.json").read_text())
        order, _ = C.word_order(r["comment"] for r in rows)
        assert cf["frequent_words"] == order[:12] and "mixed_in_words" not in cf          # top-k only, nothing added
        assert set(st["arms"][arm]["candidate_word_ranks"]) == {"muslim", "muslims", "islam", "islamic"}
    m90 = [r for r in C.jl(d / "arms" / "muslim90" / "train.jsonl") if r["M"]]
    assert abs(sum(r["label"] for r in m90) / len(m90) - 0.90) < 0.05
    assert all(r["M"] and r["label"] == 0 for r in C.jl(d / "eval" / "Dm.jsonl"))
    assert all(r["M"] for r in C.jl(d / "eval" / "SWm.jsonl"))
    G = C.jl(d / "eval" / "G.jsonl")
    assert sum(r["M"] for r in G) >= 6 and sum(r["R"] for r in G) >= 1
    assert not (d / "candidate_fields.json").exists()


def test_arm_size_is_largest_feasible_for_both_arms(tmp_path):
    """N = largest size at which both arms can be built (M at natural share); N + 1 is not buildable."""
    import pytest
    csv = make_csv(tmp_path / "all.csv")
    splits, _, _ = B.load_csv(csv)
    pool = splits["train"]
    size = B.arm_size_rule(pool, ("muslim_nat", "muslim90"))
    N = size["n_total"]
    assert 0 < N <= len(pool) and size["binding"]
    for arm in ("muslim_nat", "muslim90"):
        target, p = B.ARMS[arm]
        rows, info = C.build_arm(pool, target, p, N, seed=B.ARM_SEED, strata=B.STRATA)
        assert len(rows) == N and info["n_target"] == size["n_group_rows_per_arm"]
    if N < len(pool):
        with pytest.raises(ValueError):
            for arm in ("muslim_nat", "muslim90"):
                target, p = B.ARMS[arm]
                C.build_arm(pool, target, p, N + 1, seed=B.ARM_SEED, strata=B.STRATA)
    st = B.build(csv, tmp_path / "data", sizes={"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, top_k=12)
    assert st["arm_size"]["n_total"] == N == st["arms"]["muslim90"]["muslim_design"]["n"]
    with pytest.raises(ValueError):
        B.build(csv, tmp_path / "data_bad", n_total=N + 1, sizes={"T": 40, "D": 12, "Dm": 8, "G": 3, "SW": 12, "SWm": 8}, top_k=12)
