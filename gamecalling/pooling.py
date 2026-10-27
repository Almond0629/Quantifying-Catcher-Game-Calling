"""Empirical-Bayes Dirichlet-multinomial pooling (project_plan.md, decided 2026-10-27).

The category counts n of each cell (K categories) are pulled toward a prior mean m
with one concentration alpha shared by all cells of a level. alpha maximizes the
Dirichlet-multinomial marginal likelihood of those cells' counts, and the
posterior mean is (n + alpha m) / (N + alpha).

Categories with prior mean 0 are allowed only where the cell has no counts in
them; they keep probability 0.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

LOG_ALPHA_BOUNDS = (np.log(1e-2), np.log(1e6))


def log_marginal(alpha: float, n: np.ndarray, m: np.ndarray) -> float:
    """Sum over cells of the Dirichlet-multinomial log marginal likelihood.

    The multinomial coefficient is dropped (it does not depend on alpha). n and m
    are (cells, K); cells with no counts contribute 0.
    """
    pos = m > 0
    am = np.where(pos, alpha * m, 1.0)
    per_cat = np.where(pos, gammaln(n + am) - gammaln(am), 0.0)
    return float((gammaln(alpha) - gammaln(n.sum(-1) + alpha) + per_cat.sum(-1)).sum())


def fit_alpha(n: np.ndarray, m: np.ndarray) -> float:
    """Concentration maximizing the marginal likelihood of counts n under prior means m."""
    n = np.asarray(n, float).reshape(-1, n.shape[-1])
    m = np.broadcast_to(np.asarray(m, float), n.shape) if np.ndim(m) == 1 else np.asarray(m, float).reshape(n.shape)
    if np.any((m <= 0) & (n > 0)):
        raise ValueError("counts in a category with prior mean 0")
    keep = n.sum(-1) > 0
    n, m = n[keep], m[keep]
    if len(n) == 0:
        raise ValueError("no cells with counts")
    res = minimize_scalar(lambda t: -log_marginal(np.exp(t), n, m), bounds=LOG_ALPHA_BOUNDS, method="bounded",
                          options={"xatol": 1e-6})
    return float(np.exp(res.x))


def posterior_mean(n: np.ndarray, m: np.ndarray, alpha: float) -> np.ndarray:
    n = np.asarray(n, float)
    return (n + alpha * m) / (n.sum(-1, keepdims=True) + alpha)
