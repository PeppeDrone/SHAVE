"""DFR (Kirichenko, Izmailov & Wilson, ICLR 2023): last-layer retraining on group-balanced held-out data.

Re-implementation of DFR^Val following the authors' public code
(github.com/PolinaKirichenko/deep_feature_reweighting, dfr_evaluate_spurious.py at commit 9d8d5de,
functions dfr_on_validation_tune / dfr_on_validation_eval with balance_val=True, add_train=False):
  * features: readout states (final-layer last-token hidden state), frozen model;
  * groups g = 2 * muslim + y (WILDS Muslim annotation >= 0.5 x gold toxicity label) -> 4 groups;
  * tuning: split the reweighting set in half; fit on a group-balanced subsample of one half; pick the C in
    {1, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01} with the best worst-group accuracy on the other half
    (L1 logistic regression, liblinear, class weights {0: 1, 1: 1}, StandardScaler fit on the training half);
  * final head: 20 fits on fresh group-balanced subsamples of the whole set (StandardScaler fit on the
    model's training-set features, as in the reference code), coefficients and intercepts averaged.
Uses gold labels and group labels: it is the label-using reference method.
"""
from __future__ import annotations

import warnings

import numpy as np

# scikit-learn >= 1.8 deprecates `penalty`; it is still honoured (checked in tests/test_dfr_leace.py).
warnings.filterwarnings("ignore", message=".*penalty.*")

C_OPTIONS = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]


def _lr(c, seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight={0: 1.0, 1: 1.0}, random_state=seed)


def _balanced_idx(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_groups = int(g.max()) + 1
    g_idx = [np.where(g == k)[0] for k in range(n_groups)]
    min_g = min(len(x) for x in g_idx)
    out = []
    for x in g_idx:
        x = x.copy(); rng.shuffle(x); out.append(x[:min_g])
    return np.concatenate(out)


def tune_c(X, y, g, seed: int, c_options=C_OPTIONS) -> dict:
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    n_half = len(X) // 2
    idx = np.arange(len(X)); rng.shuffle(idx)
    tr, va = idx[n_half:], idx[:n_half]
    tr = tr[_balanced_idx(g[tr], rng)]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    res = {}
    for c in c_options:
        clf = _lr(c, seed).fit(Xtr, y[tr])
        pv = clf.predict(Xva)
        accs = [float(np.mean(pv[g[va] == k] == y[va][g[va] == k])) for k in range(int(g.max()) + 1) if (g[va] == k).any()]
        res[float(c)] = min(accs)
    best = max(res, key=lambda c: (res[c], -c_options.index(c)))  # first C in grid order on ties (np.argmax)
    return {"best_c": best, "worst_group_acc_by_c": res, "n_tune_train": int(len(tr)), "n_tune_val": int(len(va))}


def fit_dfr(X, y, g, X_scaler, seed: int, n_runs: int = 20, c_options=C_OPTIONS) -> dict:
    """Returns {"w": (d,), "b": float, ...}: new margin = x @ w + b on raw (unscaled) features."""
    from sklearn.preprocessing import StandardScaler
    X, y, g = np.asarray(X, float), np.asarray(y, int), np.asarray(g, int)
    tune = tune_c(X, y, g, seed, c_options)
    c = tune["best_c"]
    sc = StandardScaler().fit(np.asarray(X_scaler, float))
    rng = np.random.default_rng(seed + 1)
    coefs, ints = [], []
    for run in range(n_runs):
        b = _balanced_idx(g, rng)
        clf = _lr(c, seed + run).fit(sc.transform(X[b]), y[b])
        coefs.append(clf.coef_[0]); ints.append(clf.intercept_[0])
    coef, inter = np.mean(coefs, 0), float(np.mean(ints))
    w = coef / sc.scale_
    b0 = inter - float(np.sum(coef * sc.mean_ / sc.scale_))
    return {"w": w, "b": b0, "coef_scaled": coef, "intercept_scaled": inter, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
            "best_c": c, "tune": tune, "n_runs": n_runs, "n_per_group": int(min(np.bincount(g))),
            "group_counts": np.bincount(g).tolist(), "n_nonzero": int(np.sum(np.abs(coef) > 0))}


def margins(X, fit: dict) -> np.ndarray:
    return np.asarray(X, float) @ fit["w"] + fit["b"]
