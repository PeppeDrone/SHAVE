"""LEACE baseline (Belrose et al., NeurIPS 2023; arXiv 2306.03819) with an oracle concept, applied to the classifier
the way the paper and the official package describe for preventing a classifier from using a concept:

  * Paper, Sec. 5.1-5.2 (pp. 6-8): the concept is erased from the final-layer representation of a frozen encoder
    (the [CLS] embedding of BERT's last layer), and the main-task linear (logistic-regression) classifier is fitted
    over the erased embeddings.
  * concept-erasure README, "Batch usage": eraser = LeaceEraser.fit(X, Z); X_ = eraser(X); then
    LogisticRegression(max_iter=1000).fit(X_, Y) is the classifier trained on the erased features.

Here: representation = the readout state of our fine-tuned classifier (final-layer hidden state at the last prompt
token, the input of the linear lm_head read-out), frozen. Fit data = the arm's training comments (the data the
classifier was trained on), all rows. Concept Z = 1 if the comment contains the exact word "muslim"
(group M, civil.has_m), else 0: an ORACLE concept, given to LEACE, which SHAVE must find blind.
Head = logistic regression with the README's settings (sklearn defaults, max_iter=1000) fitted on the erased training
features with the training class labels. Inference: readout -> eraser -> head.
For the record only, the fixed original read-out applied to the erased readout is also scored (not a reported method).
"""
from __future__ import annotations

import numpy as np

MUSLIM_TERMS = ["muslim"]   # the planted group M: exact word "muslim" (civil.has_m)
HEAD_MAX_ITER = 1000   # concept-erasure README, Batch usage


def concept_z(comments) -> np.ndarray:
    from .civil import has_m
    return np.array([has_m(c) for c in comments], dtype=np.int64)


def fit_eraser(X: np.ndarray, z: np.ndarray) -> dict:
    """LeaceEraser.fit(X, Z) (official package, defaults). Returns {"A", "mu", "eraser", ...} with
    eraser(x) = x - (x - mu) A^T."""
    import torch
    from concept_erasure import LeaceEraser
    x = torch.as_tensor(np.asarray(X, dtype=np.float64))
    zz = torch.as_tensor(np.asarray(z, dtype=np.float64)).reshape(-1, 1)
    er = LeaceEraser.fit(x, zz)
    A = (er.proj_left @ er.proj_right).double()
    mu = er.bias.double() if er.bias is not None else torch.zeros(x.shape[1], dtype=torch.float64)
    return {"A": A.float().numpy(), "mu": mu.float().numpy(), "impl": "concept_erasure.LeaceEraser.fit (defaults)",
            "rank": int(np.linalg.matrix_rank(A.numpy(), tol=1e-6)), "eraser": er}


def erase(X: np.ndarray, fit: dict) -> np.ndarray:
    import torch
    return fit["eraser"](torch.as_tensor(np.asarray(X, dtype=np.float64))).numpy()


def apply_np(X: np.ndarray, A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return X - (X - mu) @ A.T


def fit_head(Xe: np.ndarray, y: np.ndarray, seed: int):
    """Main-task logistic regression on the erased features (README settings)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=HEAD_MAX_ITER, random_state=int(seed)).fit(Xe, np.asarray(y))


def head_margins(X: np.ndarray, fit: dict, head) -> np.ndarray:
    return head.decision_function(erase(X, fit)) if len(X) else np.zeros(0)


def residual_cross_cov(X: np.ndarray, z: np.ndarray) -> float:
    """Max |Cov(x'_j, z)| after erasure (should be ~0): a fit check written to the artifacts."""
    zc = z - z.mean()
    Xc = X - X.mean(0)
    return float(np.abs(Xc.T @ zc / max(1, len(z))).max())


def probe_check(Xe: np.ndarray, z: np.ndarray, seed: int) -> dict:
    """README check: a logistic regression fitted to predict Z from the erased features learns ~nothing."""
    from sklearn.linear_model import LogisticRegression
    if len(set(np.asarray(z).tolist())) < 2:
        return {"coef_inf_norm": None}
    lr = LogisticRegression(max_iter=HEAD_MAX_ITER, tol=0.0, random_state=int(seed)).fit(Xe, z)
    return {"coef_inf_norm": float(np.abs(lr.coef_).max()), "train_acc": float(lr.score(Xe, z)),
            "majority_acc": float(max(np.mean(z), 1 - np.mean(z)))}
