import numpy as np

from interp_pilot import civil as C
from interp_pilot.metrics import OFF_TARGET, group_rates
from interp_pilot.stats import bootstrap_ci, holm, paired_summary, sign_flip_p, wilcoxon_p


def _rows():
    rows = []
    for g in C.GROUPS + ["M", "R"]:
        for y in (0, 1):
            for k in range(4):
                rows.append({"label": y, "groups": {h: int(h == g) for h in C.GROUPS}, "M": int(g == "M"), "R": int(g == "R")})
    # LGBTQ rows that also contain the word "muslim": excluded from LGBTQ's spillover rates
    for y in (0, 1):
        rows.append({"label": y, "groups": {h: int(h == "LGBTQ") for h in C.GROUPS}, "M": 1, "R": 0})
    return rows


def test_group_rates_definitions():
    rows = _rows()
    # predict 1 for every M row, perfect elsewhere
    m = np.array([1.0 if r["M"] else (1.0 if r["label"] else -1.0) for r in rows])
    gr = group_rates(m, rows)
    assert gr["M"]["tpr"] == 1.0 and gr["M"]["tnr"] == 0.0 and gr["M"]["fpr"] == 1.0 and gr["M"]["ba"] == 0.5
    assert gr["M"]["n_pos"] == 5 and gr["R"]["ba"] == 1.0
    for g in OFF_TARGET:
        assert gr[g]["ba"] == 1.0                        # the LGBTQ rows with "muslim" do not count for LGBTQ
    assert gr["LGBTQ"]["n_neg"] == 4
    assert group_rates(m, rows, exclude_word_groups=False)["LGBTQ"]["ba"] < 1.0
    assert len(OFF_TARGET) == 7 and "muslim" not in OFF_TARGET


def test_bootstrap_ci_and_none():
    lo, hi = bootstrap_ci([0.1] * 10)
    assert lo == hi == 0.1
    assert bootstrap_ci([]) == (None, None)
    lo, hi = bootstrap_ci(np.arange(20) / 20)
    assert lo < np.mean(np.arange(20) / 20) < hi


def test_sign_flip_exact():
    assert abs(sign_flip_p([1.0] * 10) - 2 / 1024) < 1e-12        # two-sided exact: all-positive and all-negative
    assert sign_flip_p([1.0, -1.0]) == 1.0
    assert sign_flip_p([]) is None


def test_sign_flip_exact_up_to_20_seeds():
    import itertools
    assert abs(sign_flip_p([1.0] * 20) - 2 / 2 ** 20) < 1e-15     # exact at 20 seeds per arm (no Monte Carlo)
    d = np.random.default_rng(3).normal(0.01, 0.05, 12)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=12)))
    ref = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))
    assert sign_flip_p(d) == ref                                    # same value as full enumeration


def test_holm_matches_hand_computation():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["c"] - 0.06) < 1e-12 and abs(adj["b"] - 0.06) < 1e-12 and adj["d"] is None


def test_paired_summary():
    s = paired_summary([0.6, 0.7, 0.8, np.nan], [0.5, 0.5, 0.5, 0.5], n_boot=500)
    assert s["n"] == 3 and abs(s["mean_delta"] - 0.2) < 1e-12 and abs(s["median_delta"] - 0.2) < 1e-12
    assert s["ci95_low"] <= 0.2 <= s["ci95_high"]
    assert wilcoxon_p([0.0, 0.0]) is None or wilcoxon_p([0.0, 0.0]) >= 0
