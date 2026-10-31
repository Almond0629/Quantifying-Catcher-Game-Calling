"""Ranking model (project_plan.md, Phase 3; locked), fitted separately by season.

    score_i = mu + alpha(catcher) + u(pitcher) + f(count, base-out) + e_i

- Main ("random"): catcher-season and pitcher-season crossed random effects.
- Comparison ("fixed"): the same catcher treatment and context, with pitcher-season
  fixed effects.
- Context f: fixed effects for count (12 levels) and base-out state (24 levels).

Scores are costs in runs per pitch, modeled per 100 pitches (lower = better);
display flips the sign to runs saved vs. average per 100 pitches. A catcher's
effect is his BLUP, i.e. relative to the league average of the random effect.

Fitting is exact REML from sufficient statistics (Henderson's mixed-model
equations). With M = [X | Z] and Gamma = G / sigma^2, the profiled criterion is

    -2 l_R = (n - p) log(r / (n - p)) + (n - p) + log|Gamma| + log|C|,
    C = M'M + blockdiag(0, Gamma^-1),  r = y'y - theta' M'y,  C theta = M'y,

so only M'M, M'y and y'y are needed (this replaces aggregating to cells, with the
same result). Pitches within a game are dependent; the model's own standard
errors ignore that, and intervals come from the game bootstrap. `pev_se` (the
model-based prediction-error SD) is reported for diagnostics and the simulation.

Pitches carry game weights `w` (weighting.py); every cross-product is weighted,
so a game with weight m fits exactly like m copies. Variance components are
estimated with `w_hyper` (equal to `w` except in the experimental "firstcopy"
bootstrap), and effects are computed with `w` at those variance components.

Convergence: the REML criterion is minimized with L-BFGS-B from three starts and
then polished with Nelder-Mead. `converged` requires first-order optimality on
the box (zero derivative inside, non-negative derivative at the zero-variance
bound) and no further gain from a fresh L-BFGS-B run.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from gamecalling import weighting

SCALE = 100.0  # model scores per 100 pitches
LOG_GAMMA_BOUNDS = (-30.0, 5.0)
ZERO_LOG_GAMMA = -18.0


def _onehot(codes: np.ndarray, n_levels: int, drop_first=False) -> sp.csr_matrix:
    rows = np.arange(len(codes))
    m = sp.csr_matrix((np.ones(len(codes)), (rows, codes)), shape=(len(codes), n_levels))
    return m[:, 1:] if drop_first else m


@dataclass
class Design:
    MtM: np.ndarray
    Mty: np.ndarray
    yty: float
    n: int
    p: int                      # fixed-effect columns
    random_sizes: list          # sizes of the random blocks, in order
    fixed_names: list
    levels: dict = field(default_factory=dict)  # block name -> level labels


def _reference_first(codes: np.ndarray) -> np.ndarray:
    """If level 0 (the dropped reference) is absent, as in a runners-on-only sample, the first present level
    takes its place, so the intercept stays identified. Unchanged whenever level 0 is present."""
    ref = codes.min()
    return codes if ref == 0 else np.where(codes == ref, 0, codes)


def build_design(df: pd.DataFrame, y: np.ndarray, pitcher_fixed: bool, weights=None) -> Design:
    """Weighted cross-products for score y on [intercept, count, base-out (, pitcher) | catcher (, pitcher)]."""
    n = len(df)
    wts = weighting.w(df) if weights is None else np.asarray(weights, float)
    cidx = _reference_first(df["balls"].to_numpy(int) * 3 + df["strikes"].to_numpy(int))
    bo = _reference_first(df["base_out"].to_numpy(int))
    cat_codes, cat_levels = pd.factorize(df["catcher"], sort=True)
    pit_codes, pit_levels = pd.factorize(df["pitcher"], sort=True)
    blocks = [sp.csr_matrix(np.ones((n, 1))), _onehot(cidx, 12, True), _onehot(bo, 24, True)]
    names = ["intercept"] + [f"count_{c}" for c in range(1, 12)] + [f"base_out_{b}" for b in range(1, 24)]
    random_blocks = [_onehot(cat_codes, len(cat_levels))]
    if pitcher_fixed:
        blocks.append(_onehot(pit_codes, len(pit_levels), True))
        names += [f"pitcher_{p}" for p in pit_levels[1:]]
    else:
        random_blocks.append(_onehot(pit_codes, len(pit_levels)))
    M = sp.hstack(blocks + random_blocks).tocsc()
    MtM = (M.T @ sp.diags(wts) @ M).toarray()
    p = sum(b.shape[1] for b in blocks)
    # Drop context levels absent from the data (keeps X full rank).
    keep = np.ones(MtM.shape[0], bool)
    keep[:p] = np.diag(MtM)[:p] > 0
    MtM = MtM[np.ix_(keep, keep)]
    Mty = (M.T @ (wts * y))[keep]
    names = [nm for nm, k in zip(names, keep[:len(names)]) if k]
    levels = {"catcher": np.asarray(cat_levels)}
    if not pitcher_fixed:
        levels["pitcher"] = np.asarray(pit_levels)
    return Design(MtM, Mty, float(wts @ (y * y)), float(wts.sum()), int(keep[:p].sum()),
                  [b.shape[1] for b in random_blocks], names, levels)


def _solve(d: Design, log_gamma: np.ndarray):
    pen = np.zeros(d.MtM.shape[0])
    start = d.p
    for size, lg in zip(d.random_sizes, log_gamma):
        pen[start:start + size] = np.exp(-lg)
        start += size
    C = d.MtM + np.diag(pen)
    cf = cho_factor(C, lower=True, check_finite=False)
    theta = cho_solve(cf, d.Mty, check_finite=False)
    r = d.yty - theta @ d.Mty
    logdet = 2 * np.log(np.diag(cf[0])).sum()
    return cf, theta, r, logdet


def reml_criterion(d: Design, log_gamma) -> float:
    """-2 x profiled REML log likelihood, up to a constant."""
    log_gamma = np.atleast_1d(np.asarray(log_gamma, float))
    _, _, r, logdet = _solve(d, log_gamma)
    dof = d.n - d.p
    return dof * np.log(r / dof) + dof + float(np.dot(d.random_sizes, log_gamma)) + logdet


def _kkt_ok(d: Design, lg: np.ndarray, h=1e-4, tol=1e-3) -> bool:
    """First-order optimality on the box: zero derivative inside, non-negative at the lower bound."""
    for j in range(len(lg)):
        up, dn = lg.copy(), lg.copy()
        up[j] += h
        if lg[j] - h < LOG_GAMMA_BOUNDS[0]:
            g = (reml_criterion(d, up) - reml_criterion(d, lg)) / h
            if g < -tol:
                return False
        else:
            dn[j] -= h
            g = (reml_criterion(d, up) - reml_criterion(d, dn)) / (2 * h)
            if abs(g) > tol and not (lg[j] + h > LOG_GAMMA_BOUNDS[1] and g < 0):
                return False
    return True


@dataclass
class Fit:
    model: str
    n: int
    sigma2: float                 # residual variance (per-100-pitch units squared)
    variances: dict               # random block -> variance
    fixed: pd.Series              # fixed-effect estimates
    catchers: pd.DataFrame        # catcher, effect, pev_se, n_pitches
    pitchers: pd.DataFrame | None
    reml: float
    log_gamma: np.ndarray
    at_bound: bool
    converged: bool = True
    polish_change: float = 0.0   # |log gamma| change from the Nelder-Mead polish


def fit(df: pd.DataFrame, score: str, pitcher_fixed: bool = False, start=None) -> Fit:
    """Fit the ranking model to one season's scored pitches (column `score`, runs per pitch)."""
    y = df[score].to_numpy(float) * SCALE
    d = build_design(df, y, pitcher_fixed)
    dh = d
    if "w_hyper" in df and not np.array_equal(weighting.w_hyper(df), weighting.w(df)):
        dh = build_design(df, y, pitcher_fixed, weighting.w_hyper(df))
    k = len(d.random_sizes)
    starts = [np.full(k, s) for s in (-8.0, -5.0, -2.0)] if start is None else [np.asarray(start, float)]
    best = None
    for x0 in starts:
        res = minimize(lambda lg: reml_criterion(dh, lg), x0, method="L-BFGS-B", bounds=[LOG_GAMMA_BOUNDS] * k)
        if best is None or res.fun < best.fun:
            best = res
    polish = minimize(lambda lg: reml_criterion(dh, np.clip(lg, *LOG_GAMMA_BOUNDS)), best.x, method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    lg_p = np.clip(polish.x, *LOG_GAMMA_BOUNDS)
    polish_change = float(np.max(np.abs(lg_p - best.x)))
    lg = lg_p if polish.fun < best.fun - 1e-9 else best.x
    crit = min(polish.fun, best.fun)
    # Converged: first-order optimal on the box, and a fresh L-BFGS-B run from the final point gains nothing.
    recheck = minimize(lambda x: reml_criterion(dh, x), lg, method="L-BFGS-B", bounds=[LOG_GAMMA_BOUNDS] * k)
    converged = bool(_kkt_ok(dh, lg) and recheck.fun > crit - 1e-6)
    _, _, r_h, _ = _solve(dh, lg)
    cf, theta, r, _ = _solve(d, lg)
    sigma2 = r_h / (dh.n - dh.p)
    gammas = np.exp(lg)
    names = ["catcher"] + ([] if pitcher_fixed else ["pitcher"])
    variances = {nm: sigma2 * g for nm, g in zip(names, gammas)}

    nc = d.random_sizes[0]
    idx = np.arange(d.p, d.p + nc)
    E = np.zeros((d.MtM.shape[0], nc))
    E[idx, np.arange(nc)] = 1.0
    pev = sigma2 * np.einsum("ij,ij->j", E, cho_solve(cf, E, check_finite=False))
    counts = pd.Series(weighting.w(df), index=df.index).groupby(df["catcher"]).sum()
    catchers = pd.DataFrame({"catcher": d.levels["catcher"], "effect": theta[idx], "pev_se": np.sqrt(pev)})
    catchers["n_pitches"] = catchers["catcher"].map(counts).to_numpy()
    pitchers = None
    if not pitcher_fixed:
        np_ = d.random_sizes[1]
        pitchers = pd.DataFrame({"pitcher": d.levels["pitcher"], "effect": theta[d.p + nc:d.p + nc + np_]})
    at_bound = bool(np.any(lg < ZERO_LOG_GAMMA))
    return Fit("fixed" if pitcher_fixed else "random", d.n, sigma2, variances,
               pd.Series(theta[:d.p], index=d.fixed_names), catchers, pitchers, float(crit), lg, at_bound,
               converged, polish_change)


def shared_pitchers(df: pd.DataFrame, min_pair=100) -> pd.Series:
    """Per catcher: pitchers with >= min_pair pitches to him and >= min_pair to at least one other catcher."""
    pairs = pd.Series(weighting.w(df), index=df.index).groupby([df["catcher"], df["pitcher"]]).sum()
    pairs = pairs[pairs >= min_pair].reset_index()
    n_catchers = pairs.groupby("pitcher")["catcher"].transform("size")
    return pairs[n_catchers >= 2].groupby("catcher").size()


def fit_scores(scores: pd.DataFrame, score_cols=("D", "A", "B"), models=("random", "fixed")) -> dict:
    """{(season, score, model): Fit} for every season in `scores`."""
    out = {}
    for season, g in scores.groupby("season"):
        for col in score_cols:
            for m in models:
                out[season, col, m] = fit(g, col, pitcher_fixed=(m == "fixed"))
    return out
