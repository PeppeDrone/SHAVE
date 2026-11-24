import numpy as np
import torch

from interp_pilot import dfr, leace_oracle as L


def test_concept_z_lexicon():
    z = L.concept_z(["Muslims are here", "a Muslim's view", "anti-muslim", "the park", "Islamabad", "MUSLIM", "muslimah", None])
    assert z.tolist() == [0, 1, 1, 0, 0, 1, 0, 0]


def test_leace_matches_library_and_removes_covariance():
    rng = np.random.default_rng(0)
    n, d = 400, 12
    z = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, d)) + np.outer(z, rng.normal(size=d) * 2)
    fit = L.fit_eraser(X, z)
    shave = L.apply_np(X, fit["A"], fit["mu"])
    lib = fit["eraser"](torch.as_tensor(X, dtype=torch.float64)).numpy()
    assert np.allclose(shave, lib, atol=1e-4)
    assert L.residual_cross_cov(X, z) > 0.1 and L.residual_cross_cov(shave, z) < 1e-4
    assert fit["rank"] == 1


def test_sklearn_penalty_l1_is_honoured():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40)); y = (X[:, 0] > 0).astype(int)
    m = dfr._lr(0.05, 0).fit(X, y)
    assert int(np.sum(m.coef_ == 0)) > 30


def test_dfr_fold_and_balance():
    rng = np.random.default_rng(1)
    n, d = 800, 6
    g = rng.integers(0, 4, n); y = g % 2
    X = rng.normal(size=(n, d)); X[:, 0] += 3 * y; X[:, 1] += 3 * (g // 2)
    fit = dfr.fit_dfr(X, y, g, X_scaler=rng.normal(size=(500, d)) * 2 + 1, seed=0, n_runs=5)
    assert fit["best_c"] in dfr.C_OPTIONS and fit["n_per_group"] == min(np.bincount(g))
    # margin on raw features equals the averaged scaled-space logistic model
    Xs = (X - fit["scaler_mean"]) / fit["scaler_scale"]
    assert np.allclose(dfr.margins(X, fit), Xs @ fit["coef_scaled"] + fit["intercept_scaled"], atol=1e-8)
    assert np.mean((dfr.margins(X, fit) > 0) == y) > 0.9
    # tuning picks the first C on ties
    t = dfr.tune_c(X, y, g, seed=0)
    best = max(t["worst_group_acc_by_c"].values())
    first = next(c for c in dfr.C_OPTIONS if t["worst_group_acc_by_c"][float(c)] == best)
    assert t["best_c"] == first
