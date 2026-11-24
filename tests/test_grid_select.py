import numpy as np

from interp_pilot.shave import (cut_scores, eligible, grid_count, grid_specs, select_thresholded)


def test_full_grid_size_and_order():
    for nL, nv in ((24, 1), (24, 2), (4, 1)):
        g = grid_specs(nL, nv, "full")
        assert len(g) == grid_count(nL, nv)
        assert g[0] == ("embed", "span_pooled", None)
        last = [x for x in g if x[1] == "last"]
        assert len(last) == 3 * nL and {x[2] for x in last} == {1, 2, 4}
    assert len(grid_specs(24, 1, "full")) == 25 + 72


def test_only_full_grid_is_supported():
    import pytest
    with pytest.raises(ValueError):
        grid_specs(24, 1, "dev18")


def C(i, flip, side, kl, rank=1):
    return {"i": i, "flip": flip, "side": side, "kl": kl, "rank": rank}


def test_thresholded_selection_and_abstain():
    cands = [C(0, 0.95, 0.04, 0.05), C(1, 0.99, 0.01, 0.02, 2), C(2, 0.99, 0.01, 0.02, 1), C(3, 1.0, 0.06, 0.0)]
    assert eligible(cands[0]) and not eligible(cands[3])
    assert select_thresholded(cands)["i"] == 2              # same kl+side and flip: lower rank wins
    assert select_thresholded([C(0, 0.89, 0.0, 0.0), C(1, 1.0, 0.2, 0.0), C(2, 1.0, 0.0, 0.2)]) is None
    # tie on rounded kl+side and flip and rank: first in grid order
    assert select_thresholded([C(0, 0.95, 0.01, 0.01), C(1, 0.95, 0.01, 0.01)])["i"] == 0


def test_cut_scores_definitions():
    bT = np.array([-1.0, -1.0, -1.0, -1.0]); nT = np.array([1.0, 1.0, -1.0, 1.0])
    bO = np.array([1.0, -1.0, 2.0, -2.0]); nO = np.array([1.0, 1.0, 2.0, -2.0])
    s = cut_scores(bT, bO, nT, nO, cstar=0)
    assert abs(s["flip"] - 0.75) < 1e-9 and abs(s["side"] - 0.25) < 1e-9 and s["kl"] >= 0
