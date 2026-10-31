"""M3a / M3b: batter index on the outcome and contact models (research program S6, prediction stage).

Specification: research_notes/research_program/S6_batter_index.md (revision 2). Built on P0 alone (not on M1c).

Three batter features, per batter, from an estimation set E of games:
- whiff (whiffs / swings) and chase (swings / pitches at zone 11-14): beta-binomial shrinkage toward E's rate;
- xwOBAcon (tracked, non-bunt balls in play): normal-normal shrinkage, method of moments;
then standardized by E's pitch-weighted mean and SD. Events use the raw descriptions, not the outcome model's
recoding. Bunt attempts are left out of whiff and chase; foul tips are swings with contact, not whiffs.

Regression: multinomial logistic with log P0 as a fixed offset and a ridge penalty,
    eta_ik = log P0_ik + X_i theta C_K'
with C_K an orthonormal sum-to-zero category basis. X is the three features (M3a), plus their products with
four orthonormal sum-to-zero family contrasts (M3b; no-family pitch types have zero contrasts).

Training rows use leave-one-game-out features and offsets, with E's parent quantities held fixed: shrinkage
hyperparameters, league rates, standardization constants, and P0's concentrations, league means and base level.
Rows predicted from E use the full-E features and offsets. The penalty is chosen by 5-fold game-grouped
cross-validation inside E; each inner validation fold is excluded from all preprocessing of its inner fit.

M4 (S7; research_notes/research_program/S7_ground_ball.md, revision 1) keeps M3a's outcome model and adds batter
ground-ball rate to M3a's contact model, main effects only: `bb_type` ground balls over the same tracked, non-bunt
balls in play as xwOBAcon, beta-binomial shrinkage, standardized. `ground_ball=False` is M3a's existing path.

Unit weights only: the weighted moment formulas are not specified (S6 §2). `batter=False` calls only P0's own
functions. No policy, solver or scoring code here: those follow only an adoption.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gamecalling import config, loader, models, pooling, weighting

SEED = 20261030
FORWARD_CODE = 3            # inner-fold seed code of the forward fit; role folds are 0-2
N_INNER = 5
LAMBDAS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)   # largest first: warm-started path, ties to the larger
MIN_SD = 1e-12
GTOL, FTOL, MAX_GRAD = 1e-10, 1e-15, 1e-7
FEATURES = ("whiff", "chase", "xwobacon")
GROUND_BALL = "ground_ball"
VARIANTS = ("M3a", "M3b")
BB_TYPES = (GROUND_BALL, "fly_ball", "line_drive", "popup")
BB_KEYS = ["game_pk", "at_bat_number", "pitch_number"]

WHIFF_DESC = ("swinging_strike", "swinging_strike_blocked")
SWING_DESC = WHIFF_DESC + ("foul_tip", "foul", "hit_into_play")
TAKE_DESC = ("ball", "blocked_ball", "called_strike", "hit_by_pitch")
BUNT_DESC = ("foul_bunt", "missed_bunt", "bunt_foul_tip")
OUTSIDE_ZONES = (11, 12, 13, 14)
TRACKED_BINS = (1, 2, 3, 4, 5, 6)
STAT_COLUMNS = ["pitches", "swings", "whiffs", "outside", "chases", "con_n", "con_y", "con_yy"]
GB_COLUMN = "gb_k"


def helmert(K: int) -> np.ndarray:
    """K x (K-1) normalized Helmert basis: orthonormal columns, each orthogonal to the vector of ones."""
    C = np.zeros((K, K - 1))
    for j in range(1, K):
        C[:j, j - 1] = 1.0
        C[j, j - 1] = -j
        C[:, j - 1] /= np.sqrt(j * (j + 1))
    return C


FAMILY_CONTRAST = dict(zip(config.FAMILY_ORDER, helmert(len(config.FAMILY_ORDER))))
NO_FAMILY = np.zeros(len(config.FAMILY_ORDER) - 1)


def role_code(train: pd.DataFrame) -> int:
    """Inner-fold seed code; refuses anything but one season's single fold (a role) or all its folds (forward).
    A role trains on one fold, never on the selection and valuation folds combined."""
    seasons, folds = train["season"].unique(), sorted(int(f) for f in train["fold"].unique())
    if len(seasons) != 1:
        raise ValueError(f"batter models fit within one season, got {sorted(seasons)}")
    if len(folds) == 1:
        return folds[0]
    if len(folds) == config.N_FOLDS:
        return FORWARD_CODE
    raise ValueError(f"a role trains on its own fold only, got folds {folds}")


def inner_folds(games, season: int, code: int) -> pd.Series:
    """Game -> inner fold: sorted games, seeded permutation, position mod N_INNER."""
    games = np.sort(np.unique(np.asarray(games)))
    perm = np.random.default_rng([SEED, int(season), int(code)]).permutation(games)
    return pd.Series(np.arange(len(perm)) % N_INNER, index=perm, name="inner").sort_index()


def _check_unit_weights(rows: pd.DataFrame) -> None:
    if not (np.all(weighting.w(rows) == 1) and np.all(weighting.w_hyper(rows) == 1)):
        raise ValueError("S6 supports unit weights only (S6 §2)")


# --- Feature events ---

def attach_bb_type(rows: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """rows with `bb_type` from the raw table, keeping row order and index. Duplicate keys on either side, an
    unmatched pitch row and an unexpected label all raise."""
    if rows.duplicated(BB_KEYS).any() or raw.duplicated(BB_KEYS).any():
        raise ValueError("duplicate pitch keys in the bb_type join")
    bad = set(raw["bb_type"].dropna().unique()) - set(BB_TYPES)
    if bad:
        raise ValueError(f"unexpected bb_type labels: {sorted(bad)}")
    idx = pd.MultiIndex.from_frame(raw[BB_KEYS]).get_indexer(pd.MultiIndex.from_frame(rows[BB_KEYS]))
    if (idx < 0).any():
        raise ValueError("a pitch row has no raw row in the bb_type join")
    return rows.assign(bb_type=raw["bb_type"].to_numpy()[idx])


def feature_events(rows: pd.DataFrame, ground_ball: bool = False) -> pd.DataFrame:
    """Per fit-eligible pitch: batter, game_pk and the event columns of STAT_COLUMNS (plus gb_k for M4)."""
    use = rows[rows["outcome_fit_eligible"].astype(bool)]
    d = use["description"]
    if not d.isin(SWING_DESC + TAKE_DESC + BUNT_DESC).all():
        raise ValueError(f"unmapped descriptions: {sorted(set(d[~d.isin(SWING_DESC + TAKE_DESC + BUNT_DESC)]))}")
    bunt = d.isin(BUNT_DESC) | use["contact_bin"].isin([loader.BUNT_BIN]).astype(bool)
    swing = d.isin(SWING_DESC) & ~bunt
    outside = use["zone"].isin(OUTSIDE_ZONES) & ~bunt
    tracked = use["contact_fit_eligible"].astype(bool) & use["contact_bin"].isin(TRACKED_BINS).astype(bool)
    y = pd.to_numeric(use["xwoba"], errors="coerce").where(tracked, 0.0)
    if y.isna().any():
        raise ValueError("a tracked, non-bunt ball in play has no xwOBA")
    out = pd.DataFrame({"batter": use["batter"], "game_pk": use["game_pk"], "pitches": 1.0, "swings": swing * 1.0,
                        "whiffs": d.isin(WHIFF_DESC) * 1.0, "outside": outside * 1.0,
                        "chases": (swing & outside) * 1.0, "con_n": tracked * 1.0, "con_y": y.to_numpy(float),
                        "con_yy": y.to_numpy(float) ** 2})
    if ground_ball:
        if use.loc[tracked, "bb_type"].isna().any():
            raise ValueError("a tracked, non-bunt ball in play has no bb_type")
        out[GB_COLUMN] = (tracked & use["bb_type"].eq(GROUND_BALL).fillna(False).astype(bool)) * 1.0
    return out


# --- Shrinkage (S6 §3.1-3.3) ---

@dataclass
class BetaBinomial:
    mu: float
    kappa: float       # NaN when unidentified (league rate 0 or 1, or no events)
    note: str = ""

    def value(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        if np.isnan(self.kappa):
            return np.full(len(k), self.mu)
        return np.where(n == 0, self.mu, (k + self.kappa * self.mu) / (n + self.kappa))


def fit_beta_binomial(k: np.ndarray, n: np.ndarray) -> BetaBinomial:
    total = n.sum()
    if total == 0:
        return BetaBinomial(0.0, np.nan, "no events")
    mu = k.sum() / total
    if mu == 0.0 or mu == 1.0:
        return BetaBinomial(float(mu), np.nan, "league rate 0 or 1: concentration unidentified")
    keep = n > 0
    return BetaBinomial(float(mu), pooling.fit_alpha(np.column_stack([k[keep], n[keep] - k[keep]]),
                                                     np.array([mu, 1.0 - mu])))


@dataclass
class NormalNormal:
    mu: float
    sigma2: float
    tau2: float
    note: str = ""

    def value(self, sy: np.ndarray, n: np.ndarray) -> np.ndarray:
        if self.tau2 == 0.0:
            return np.full(len(sy), self.mu)
        has = n > 0
        ybar = np.where(has, sy / np.where(has, n, 1.0), self.mu)
        if self.sigma2 == 0.0:
            return ybar
        B = np.where(has, self.tau2 / (self.tau2 + self.sigma2 / np.where(has, n, 1.0)), 0.0)
        return self.mu + B * (ybar - self.mu)


def fit_normal_normal(n: np.ndarray, sy: np.ndarray, syy: np.ndarray) -> NormalNormal:
    keep = n > 0
    n_, sy_, syy_ = n[keep], sy[keep], syy[keep]
    N, J = n_.sum(), int(keep.sum())
    if N == 0:
        return NormalNormal(0.0, 0.0, 0.0, "no events")
    mu = float(sy_.sum() / N)
    if J <= 1 or N - J <= 0:
        return NormalNormal(mu, 0.0, 0.0, "too few batters or events: tau2 set to 0")
    ybar = sy_ / n_
    sigma2 = float(max(0.0, (syy_ - sy_ * ybar).sum()) / (N - J))
    msb = float((n_ * (ybar - mu) ** 2).sum() / (J - 1))
    n0 = float((N - (n_ ** 2).sum() / N) / (J - 1))
    return NormalNormal(mu, sigma2, max(0.0, (msb - sigma2) / n0))


# --- Features of one estimation set ---

@dataclass
class Features:
    batters: pd.Index
    totals: np.ndarray        # (J, STAT_COLUMNS)
    game_index: pd.MultiIndex # (batter, game_pk)
    game_stats: np.ndarray
    whiff: BetaBinomial
    chase: BetaBinomial
    xw: NormalNormal
    mean: np.ndarray
    sd: np.ndarray
    gb: BetaBinomial | None = None   # M4's fourth feature

    def _raw(self, stats: np.ndarray) -> np.ndarray:
        c = {name: stats[:, i] for i, name in enumerate(STAT_COLUMNS)}
        cols = [self.whiff.value(c["whiffs"], c["swings"]), self.chase.value(c["chases"], c["outside"]),
                self.xw.value(c["con_y"], c["con_n"])]
        if self.gb is not None:
            cols.append(self.gb.value(stats[:, len(STAT_COLUMNS)], c["con_n"]))
        return np.column_stack(cols)

    def gb_xw_corr(self) -> float:
        """Unweighted correlation of shrunk ground-ball rate and xwOBAcon, over batters with a denominator."""
        keep = self.totals[:, STAT_COLUMNS.index("con_n")] > 0
        raw = self._raw(self.totals[keep])
        return float(np.corrcoef(raw[:, 2], raw[:, 3])[0, 1])

    def _standardize(self, raw: np.ndarray) -> np.ndarray:
        ok = self.sd >= MIN_SD
        return np.where(ok, (raw - self.mean) / np.where(ok, self.sd, 1.0), 0.0)

    def _stats(self, batter) -> np.ndarray:
        idx = self.batters.get_indexer(np.asarray(batter))
        return np.where((idx >= 0)[:, None], self.totals[idx], 0.0)

    def z_full(self, batter) -> np.ndarray:
        """Standardized full-E features; a batter with no history gets the league values (generally not 0)."""
        return self._standardize(self._raw(self._stats(batter)))

    def z_logo(self, batter, game) -> np.ndarray:
        """Standardized features with the row's own game removed, parents held fixed."""
        gidx = self.game_index.get_indexer(pd.MultiIndex.from_arrays([np.asarray(batter), np.asarray(game)]))
        if (gidx < 0).any():
            raise ValueError("a training row's (batter, game) is not in the estimation set")
        return self._standardize(self._raw(self._stats(batter) - self.game_stats[gidx]))


def fit_features(rows: pd.DataFrame, parents: "Features | None" = None, ground_ball: bool = False) -> Features:
    """Features of the estimation set `rows`. With `parents`, its hyperparameters and standardization constants
    are held fixed (the leave-one-game-out invariance test). `ground_ball` adds M4's fourth feature."""
    ev = feature_events(rows, ground_ball)
    by_game = ev.groupby(["batter", "game_pk"])[STAT_COLUMNS + ([GB_COLUMN] if ground_ball else [])].sum()
    totals = by_game.groupby(level="batter").sum()
    t = totals.to_numpy()
    c = {name: t[:, i] for i, name in enumerate(STAT_COLUMNS)}
    if parents is None:
        whiff = fit_beta_binomial(c["whiffs"], c["swings"])
        chase = fit_beta_binomial(c["chases"], c["outside"])
        xw = fit_normal_normal(c["con_n"], c["con_y"], c["con_yy"])
        gb = fit_beta_binomial(t[:, len(STAT_COLUMNS)], c["con_n"]) if ground_ball else None
        mean, sd = np.zeros(3 + ground_ball), np.zeros(3 + ground_ball)
    else:
        whiff, chase, xw, mean, sd, gb = parents.whiff, parents.chase, parents.xw, parents.mean, parents.sd, parents.gb
    f = Features(totals.index, t, by_game.index, by_game.to_numpy(), whiff, chase, xw, mean, sd, gb)
    if parents is None:
        raw, w = f._raw(t), c["pitches"]
        f.mean = (raw * w[:, None]).sum(0) / w.sum()
        f.sd = np.sqrt((((raw - f.mean) ** 2) * w[:, None]).sum(0) / w.sum())
    return f


# --- Offsets and observations ---

def logo_offset(model: models.PooledModel, frame: pd.DataFrame) -> np.ndarray:
    """P0 pitcher-cell posterior means with each row's own game removed; alpha_cell, the league means and the
    base level stay at the model's (full-E) values. `frame` has the level keys, y and game_pk."""
    keys = list(model.levels[2])
    K = model.cell_n.shape[1]
    n = models._lookup(model.cell_n, frame, keys)
    n_game = models._lookup(models._counts(frame, keys + ["game_pk"], K), frame, keys + ["game_pk"])
    return pooling.posterior_mean(n - n_game, model.league_mean(frame), model.alpha_cell)


def training(kind: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Observations with batter and game_pk (research.Spec.training); P0's models ignore the extra columns."""
    frame = models.outcome_training(rows) if kind == "outcome" else models.contact_training(rows)
    return frame.assign(batter=rows.loc[frame.index, "batter"].to_numpy(),
                        game_pk=rows.loc[frame.index, "game_pk"].to_numpy())


def design(z: np.ndarray, actions, variant: str) -> np.ndarray:
    """M3a and M4: the features. M3b: plus each feature times the four family contrasts."""
    if variant != "M3b":
        return z
    g = _family_rows(actions)
    return np.hstack([z] + [z[:, [f]] * g for f in range(z.shape[1])])


def _family_rows(actions) -> np.ndarray:
    a = pd.Series(np.asarray(actions, dtype=object))
    if a.empty:
        return np.zeros((0, len(NO_FAMILY)))
    table = {u: FAMILY_CONTRAST.get(config.PITCH_FAMILIES.get(u), NO_FAMILY) for u in a.unique()}
    codes, uniques = pd.factorize(a)
    return np.vstack([table[u] for u in uniques])[codes]


# --- Ridge multinomial with an offset (S6 §4) ---

def softmax_loglik(logp: np.ndarray, X: np.ndarray, theta: np.ndarray, C: np.ndarray):
    """eta (normalized so each row's maximum is 0) and the row log-normalizers."""
    eta = logp + X @ (theta @ C.T)
    eta = eta - eta.max(axis=1, keepdims=True)
    return eta, np.log(np.exp(eta).sum(axis=1))


def objective(flat: np.ndarray, X: np.ndarray, logp: np.ndarray, y: np.ndarray, C: np.ndarray, lam: float):
    """Mean negative log-likelihood + lam ||theta||^2, and its gradient."""
    n, q = len(y), X.shape[1]
    theta = flat.reshape(q, C.shape[1])
    eta, lognorm = softmax_loglik(logp, X, theta, C)
    rows = np.arange(n)
    nll = (lognorm - eta[rows, y]).mean()
    p = np.exp(eta - lognorm[:, None])
    p[rows, y] -= 1.0
    grad = (X.T @ p @ C) / n
    return nll + lam * (flat @ flat), grad.ravel() + 2.0 * lam * flat


def is_line_search_abnormal(res) -> bool:
    """The specific L-BFGS-B ABNORMAL line-search termination (SciPy status 2); limits are status 1."""
    return not res.success and res.status == 2 and str(res.message).startswith("ABNORMAL")


def accept_fit(res, f: float, grad: np.ndarray) -> str:
    """The acceptance rule (S6 §4.3, revision 2), with f and grad recomputed at res.x. Returns "success" or
    "abnormal"; anything else stops the run."""
    if not (np.all(np.isfinite(res.x)) and np.isfinite(f) and np.all(np.isfinite(grad))):
        raise RuntimeError(f"ridge fit did not converge: nonfinite coefficients, objective or gradient ({res.message})")
    max_grad = float(np.abs(grad).max())
    if max_grad > MAX_GRAD:
        raise RuntimeError(f"ridge fit did not converge: {res.message}; recomputed max|grad|={max_grad:.2e}")
    if res.success:
        return "success"
    if is_line_search_abnormal(res):
        return "abnormal"
    raise RuntimeError(f"ridge fit did not converge: status {res.status}, {res.message}; max|grad|={max_grad:.2e}")


def fit_path(X, logp, y, K: int, lambdas=LAMBDAS, tag: dict | None = None, log: list | None = None) -> dict:
    """Warm-started path from the largest lambda down: {lambda: theta}. Each fit must pass accept_fit; accepted
    abnormal terminations are appended to `log`, with `tag` (variant, component, fold)."""
    C = helmert(K)
    q = X.shape[1]
    x0, out = np.zeros(q * (K - 1)), {}
    for lam in lambdas:
        res = minimize(objective, x0, args=(X, logp, y, C, lam), jac=True, method="L-BFGS-B",
                       options={"gtol": GTOL, "ftol": FTOL, "maxiter": 10_000, "maxfun": 50_000})
        f, grad = objective(res.x, X, logp, y, C, lam)
        if accept_fit(res, f, grad) == "abnormal" and log is not None:
            log.append({**(tag or {}), "lambda": lam, "status": int(res.status), "message": str(res.message),
                        "max_grad": float(np.abs(grad).max())})
        x0 = res.x
        out[lam] = res.x.reshape(q, K - 1)
    return out


def _softmax(logp, X, theta, C) -> np.ndarray:
    eta, lognorm = softmax_loglik(logp, X, theta, C)
    return np.exp(eta - lognorm[:, None])


# --- Preprocessing, inner cross-validation, final fit ---

@dataclass
class Prepared:
    features: Features
    p0: models.PooledModel
    frame: pd.DataFrame
    z: np.ndarray       # leave-one-game-out
    logp: np.ndarray    # leave-one-game-out


def prepare(rows: pd.DataFrame, kind: str, ground_ball: bool = False) -> Prepared:
    """Everything estimated from the estimation set `rows`, with its training rows' leave-one-game-out design."""
    features = fit_features(rows, ground_ball=ground_ball)
    p0 = models.fit_outcome_model(rows) if kind == "outcome" else models.fit_contact_model(rows)
    frame = training(kind, rows)
    return Prepared(features, p0, frame, features.z_logo(frame["batter"], frame["game_pk"]),
                    np.log(logo_offset(p0, frame)))


def inner_fold(rows: pd.DataFrame, v_games, kind: str, variant: str, lambdas=LAMBDAS, tag: dict | None = None,
               ground_ball: bool = False) -> dict:
    """One inner fit: nothing from the validation games enters preprocessing or the coefficients."""
    log = []
    is_v = rows["game_pk"].isin(v_games).to_numpy()
    prep = prepare(rows[~is_v], kind, ground_ball)
    K = prep.p0.cell_n.shape[1]
    path = fit_path(design(prep.z, prep.frame["action"], variant), prep.logp, prep.frame["y"].to_numpy(), K, lambdas,
                    tag, log)
    val = training(kind, rows[is_v])
    X = design(prep.features.z_full(val["batter"]), val["action"], variant)
    logp, y, C = np.log(prep.p0.predict(val)), val["y"].to_numpy(), helmert(K)
    losses = {}
    for lam, theta in path.items():
        eta, lognorm = softmax_loglik(logp, X, theta, C)
        losses[lam] = float((lognorm - eta[np.arange(len(y)), y]).sum())
    return {"prep": prep, "path": path, "loss": losses, "weight": float(len(y)), "abnormal": log}


def select_lambda(fold_results, lambdas=LAMBDAS) -> tuple[float, dict]:
    """Pooled validation loss per observation; exact ties go to the larger lambda."""
    weight = sum(r["weight"] for r in fold_results)
    curve = {lam: sum(r["loss"][lam] for r in fold_results) / weight for lam in lambdas}
    best = min(curve.values())
    return max(lam for lam, v in curve.items() if v == best), curve


@dataclass
class BatterModel:
    variant: str
    kind: str
    p0: models.PooledModel
    features: Features
    theta: np.ndarray
    lam: float
    curve: dict
    n_train: int
    n_games: int
    code: int
    abnormal: list

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        K = self.theta.shape[1] + 1
        X = design(self.features.z_full(query["batter"]), query["action"], self.variant)
        return _softmax(np.log(self.p0.predict(query)), X, self.theta, helmert(K))

    def alphas(self) -> dict:
        """Manifest entries: penalty, curve, hyperparameters and coefficient summaries."""
        f = self.features
        K = self.theta.shape[1] + 1
        names = FEATURES + ((GROUND_BALL,) if f.gb is not None else ())
        main = self.theta[:len(names)] @ helmert(K).T   # category shifts per feature SD, for the average family
        out = {"league": self.p0.alpha_league, "cell": self.p0.alpha_cell, "lambda": self.lam,
               "lambda_at_edge": self.lam in (max(LAMBDAS), min(LAMBDAS)), "n_train": self.n_train,
               "n_games": self.n_games, "code": self.code, "coef_norm": float(np.linalg.norm(self.theta)),
               "kappa_whiff": f.whiff.kappa, "kappa_chase": f.chase.kappa, "mu_whiff": f.whiff.mu,
               "mu_chase": f.chase.mu, "mu_xwobacon": f.xw.mu, "sigma2_xwobacon": f.xw.sigma2,
               "tau2_xwobacon": f.xw.tau2}
        for i, name in enumerate(names):
            out.update({f"mean_{name}": f.mean[i], f"sd_{name}": f.sd[i],
                        f"shift_rms_{name}": float(np.sqrt((main[i] ** 2).mean()))})
        if f.gb is not None:
            out.update({"kappa_ground_ball": f.gb.kappa, "mu_ground_ball": f.gb.mu,
                        "corr_ground_ball_xwobacon": f.gb_xw_corr(),
                        **{f"shift_ground_ball_bin{b + 1}": float(v) for b, v in enumerate(main[3])}})
        out.update({f"cv_{lam:g}": v for lam, v in self.curve.items()})
        out["abnormal_log"] = self.abnormal
        return out


def fit_component(rows: pd.DataFrame, kind: str, variant: str, code: int, lambdas=LAMBDAS,
                  ground_ball: bool = False) -> BatterModel:
    season = int(rows["season"].iloc[0])
    folds = inner_folds(rows["game_pk"], season, code)
    tag = {"variant": variant, "component": kind, "season": season, "code": code}
    results = [inner_fold(rows, folds.index[folds == v], kind, variant, lambdas, {**tag, "inner": v}, ground_ball)
               for v in range(N_INNER)]
    abnormal = [entry for r in results for entry in r["abnormal"]]
    lam, curve = select_lambda(results, lambdas)
    prep = prepare(rows, kind, ground_ball)
    K = prep.p0.cell_n.shape[1]
    path = fit_path(design(prep.z, prep.frame["action"], variant), prep.logp, prep.frame["y"].to_numpy(), K,
                    [v for v in lambdas if v >= lam], {**tag, "inner": "final"}, abnormal)
    return BatterModel(variant, kind, prep.p0, prep.features, path[lam], lam, curve, len(prep.frame), len(folds), code,
                       abnormal)


def fit_models(train: pd.DataFrame, variant: str = "M3a", batter: bool = True, ground_ball: bool = False):
    """Outcome and contact models of one role, or of the forward fit. batter=False: P0's own functions only.
    ground_ball=True is M4: M3a's outcome model, and M3a's contact model plus the fourth feature."""
    if not batter:
        return models.fit_outcome_model(train), models.fit_contact_model(train)
    code = role_code(train)
    _check_unit_weights(train)
    if ground_ball:
        return (fit_component(train, "outcome", "M3a", code),
                fit_component(train, "contact", "M4", code, ground_ball=True))
    return (fit_component(train, "outcome", variant, code), fit_component(train, "contact", variant, code))
