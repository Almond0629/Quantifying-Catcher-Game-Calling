"""Pilot validation checks (project_plan.md, Phase 4 and go/no-go criteria 2-4).

- simulation_check: simulate at-bats from fitted models' own probabilities and
  compare with the solver (criterion 2, absolute tolerance 0.001 runs).
- calibration_check: held-out calibration error and log loss of the outcome
  model against the league count x pitch-type baseline (criterion 3), plus a
  matched comparison with the model's own pooled league level; the contact-bin
  model is reported the same way as a diagnostic.
- re288_check: a simplified league-pooled model vs. held-out empirical RE288
  (criterion 4, diagnostic only). Its standard errors treat the model values as
  fixed, so they leave out model-estimation uncertainty.
"""

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, payoff, solver

SIM_TOL = 0.001
CALIBRATION_MAX = 0.02
N_CAL_BINS = 10


# --- Criterion 2: simulation ---

def simulate_pa(rng, p_outcome, p_contact, contact_value, policy, walk, strikeout, n):
    """Simulate n at-bats from 0-0 for one pitcher-hand unit at one base-out state.

    p_outcome (12, A, 6), p_contact (A, 8), contact_value (8,), policy (12, A).
    Contact bins are drawn, so the ball-in-play expectation is checked too.
    Returns the run value of each simulated at-bat.
    """
    total = np.empty(n)
    count = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    cum_pi, cum_p, cum_c = np.cumsum(policy, -1), np.cumsum(p_outcome, -1), np.cumsum(p_contact, -1)
    while len(alive):
        c = count[alive]
        a = np.minimum((rng.random(len(c))[:, None] > cum_pi[c]).sum(-1), policy.shape[-1] - 1)
        o = np.minimum((rng.random(len(c))[:, None] > cum_p[c, a]).sum(-1), 5)
        b, s = np.divmod(c, 3)
        walked = ((o == 0) & (b == 3)) | (o == 4)
        struck = ((o == 1) | (o == 2)) & (s == 2)
        bip = o == 5
        total[alive[walked]] = walk
        total[alive[struck]] = strikeout
        if bip.any():
            bins = np.minimum((rng.random(bip.sum())[:, None] > cum_c[a[bip]]).sum(-1), len(contact_value) - 1)
            total[alive[bip]] = contact_value[bins]
        count[alive] = np.where(o == 0, c + 3, np.where(s < 2, c + 1, c))
        alive = alive[~(walked | struck | bip)]
    return total


def simulation_check(solution, units, role, n_units=8, base_outs=(0, 11, 23), n_sims=4_000_000, seed=20261028):
    """Compare V(0-0) from the reference-policy solve with simulated at-bats.

    Units: the 4 with the most training pitches plus 4 drawn at random. Pass if
    |difference| <= SIM_TOL; z uses the simulation standard error. Apply
    add_noise_flags to all checks together for the multiple-comparison bound.
    """
    rng = np.random.default_rng(seed)
    n_by_unit = solution.policy.probs.shape[0]
    volume = role.policy_train.merge(units.slots, on=["pitcher", "stand", "action"]).groupby("u").size()
    volume = volume.reindex(range(n_by_unit), fill_value=0)
    top = volume.sort_values(ascending=False).index[: n_units // 2].tolist()
    rest = rng.choice(np.setdiff1d(np.flatnonzero(volume.to_numpy() >= 50), top), n_units - len(top), replace=False)
    rows = []
    for u in top + rest.tolist():
        A = units.eligible[u]
        for bo in base_outs:
            sims = simulate_pa(rng, solution.p_outcome[u][:, A], solution.p_contact[u][A],
                               role.payoffs.contact[bo], solution.policy.probs[u][:, A],
                               role.payoffs.walk[bo], role.payoffs.strikeout[bo], n_sims)
            v = solution.V_ref[u, bo, 0]
            se = sims.std() / np.sqrt(n_sims)
            rows.append(dict(pitcher=int(units.table.loc[u, "pitcher"]), stand=units.table.loc[u, "stand"],
                             base_out=bo, solver=v, simulated=sims.mean(), diff=sims.mean() - v, se=se))
    out = pd.DataFrame(rows)
    out["z"] = out["diff"] / out["se"]
    out["within_tol"] = out["diff"].abs() <= SIM_TOL
    return out


def add_noise_flags(sim: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Flag |z| above the two-sided Bonferroni bound over every row of `sim` (all checks run)."""
    from scipy.stats import norm
    z_crit = float(norm.ppf(1 - 0.025 / len(sim)))
    return sim.assign(beyond_noise=sim["z"].abs() > z_crit), z_crit


# --- Criterion 3: calibration and log loss ---

def calibration_error(pred: np.ndarray, hit: np.ndarray, n_bins=N_CAL_BINS) -> float:
    """Sample-weighted mean |mean predicted - observed frequency| over equal-count bins."""
    order = np.argsort(pred, kind="stable")
    bins = np.empty(len(pred), dtype=int)
    bins[order] = np.arange(len(pred)) * n_bins // len(pred)
    df = pd.DataFrame({"b": bins, "p": pred, "y": hit})
    g = df.groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())


def _league_baseline(train: pd.DataFrame, keys, marginal_keys, K) -> pd.DataFrame:
    """League frequencies by keys, plus one pseudo-observation spread by the marginal_keys frequencies."""
    n = models._counts(train, keys, K)
    m = models._counts(train, marginal_keys, K)
    m = m.div(m.sum(axis=1), axis=0)
    q = n.index.to_frame(index=False)
    prior = models._lookup(m, q, marginal_keys)
    probs = (n.to_numpy() + prior) / (n.sum(axis=1).to_numpy()[:, None] + 1)
    return pd.DataFrame(probs, index=n.index, columns=n.columns)


def _held_out(pitches, roles, season, k, kind):
    """Held-out observations of scoring fold k with predictions from its valuation role."""
    val = roles[season, arsenal.fold_roles(k)[1]]
    fold = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
    train = pitches[(pitches["season"] == season) & (pitches["fold"] == val.fold)]
    if kind == "outcome":
        obs, tr = models.outcome_training(fold), models.outcome_training(train)
        model, K, keys, marg = val.outcome, models.N_OUTCOMES, ["cidx", "action"], ["cidx"]
    else:
        obs, tr = models.contact_training(fold), models.contact_training(train)
        model, K, keys, marg = val.contact, models.N_BINS, ["action"], []
    pred = model.predict(obs)
    league = model.league_mean(obs)
    if marg:
        base = _league_baseline(tr, keys, marg, K)
    else:
        n = models._counts(tr, keys, K)
        tot = n.sum(axis=0) / n.to_numpy().sum()
        base = (n + tot).div(n.sum(axis=1) + 1, axis=0)
    base_pred = models._lookup(base, obs, keys, fill=np.nan)
    if np.isnan(base_pred).any():  # pitch type absent from the baseline's fold: overall frequencies
        overall = models._counts(tr.assign(_all=0), ["_all"], K).to_numpy()[0]
        base_pred = np.where(np.isnan(base_pred), overall / overall.sum(), base_pred)
    return obs["y"].to_numpy(), pred, base_pred, league


def calibration_check(pitches, roles, seasons=config.SEASONS, kind="outcome"):
    """Per season: calibration error per category, and log loss of model vs. league baseline.

    Held-out data are each scoring fold's scorable, fit-eligible pitches, predicted by
    its valuation role; the three folds are pooled within the season. `passes` uses
    the plan's baseline (league count x pitch type). `league_pooled_log_loss` is the
    matched comparison: the same model's pooled league level (same hand split and
    pooling), so the gap to it is the gain from pitcher-specific cells alone.
    """
    names = solver.OUTCOMES if kind == "outcome" else [f"bin_{b}" for b in payoff.CONTACT_BINS]
    rows = []
    for season in seasons:
        ys, ps, bs, ls = zip(*(_held_out(pitches, roles, season, k, kind) for k in range(config.N_FOLDS)))
        y, p, b, lg = np.concatenate(ys), np.vstack(ps), np.vstack(bs), np.vstack(ls)
        idx = np.arange(len(y))
        row = dict(season=season, n=len(y), log_loss=-np.log(p[idx, y]).mean(),
                   baseline_log_loss=-np.log(b[idx, y]).mean(),
                   league_pooled_log_loss=-np.log(lg[idx, y]).mean())
        for j, name in enumerate(names):
            row[f"cal_{name}"] = calibration_error(p[:, j], y == j)
        rows.append(row)
    out = pd.DataFrame(rows)
    cal_cols = [c for c in out if c.startswith("cal_")]
    out["max_cal"] = out[cal_cols].max(axis=1)
    out["passes"] = (out["max_cal"] <= CALIBRATION_MAX) & (out["log_loss"] < out["baseline_log_loss"])
    return out


# --- Criterion 4: RE288 ---

def league_role_values(role: models.Role, role_pitches: pd.DataFrame) -> np.ndarray:
    """League-pooled model V by hand x base-out x count (2, 24, 12).

    Coarser than the scoring pipeline: pitch types are collapsed into one action
    per hand with the role fold's league outcome mix by count, and the contact-bin
    mix is per hand and independent of count. Valued with the role's payoffs. Gaps
    to empirical RE288 therefore reflect these simplifications as well as
    runner events between pitches, which no model here includes.
    """
    out_train = models.outcome_training(role_pitches)
    con_train = models.contact_training(role_pitches)
    V = np.zeros((2, payoff.N_BASE_OUT, solver.N_COUNTS))
    for h, stand in enumerate(("L", "R")):
        o = out_train[out_train["stand"] == stand]
        p = models._counts(o, ["cidx"], models.N_OUTCOMES).reindex(range(12), fill_value=0).to_numpy()
        p = (p / p.sum(1, keepdims=True))[:, None, :]
        cb = models._counts(con_train[con_train["stand"] == stand].assign(_a=0), ["_a"], models.N_BINS).to_numpy()[0]
        bip = (role.payoffs.contact @ (cb / cb.sum()))[:, None, None]
        V[h], _ = solver.evaluate(p, bip, role.payoffs.walk, role.payoffs.strikeout, np.ones((12, 1)))
    return V


def empirical_runs_to_end(pitches: pd.DataFrame, pas: pd.DataFrame) -> pd.DataFrame:
    """Thrown pitches of re24_eligible PAs with the runs from the pitch to the end of the half-inning."""
    half = ["game_pk", "inning", "inning_topbot"]
    end = pitches.groupby(half)["post_bat_score"].transform("max")
    ok = pitches["pa_id"].map(pas.set_index("pa_id")["re24_eligible"]).fillna(False).astype(bool)
    use = pitches["thrown"] & ~pitches["pa_sequence_error"] & ok
    p = pitches.loc[use, ["season", "fold", "game_pk", "stand", "base_out", "bases", "outs", "balls", "strikes"]]
    return p.assign(cidx=p["balls"] * 3 + p["strikes"], runs=(end - pitches["bat_score"])[use])


def _clustered(df, by):
    """Mean model-minus-empirical gap by group, with a game-clustered standard error.

    The SE (`se_fixed_model`) treats the fitted model values as fixed. It leaves out
    uncertainty in the estimated probabilities and payoffs and the dependence from
    shared training folds; covering those needs whole-pipeline resampling.
    """
    rows = []
    for key, g in df.groupby(by):
        d = g["model"] - g["runs"]
        gap = d.mean()
        per_game = (d - gap).groupby(g["game_pk"]).sum()
        se = np.sqrt((per_game ** 2).sum()) / len(g)
        rows.append(dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)),
                         n=len(g), empirical=g["runs"].mean(), model=g["model"].mean(), gap=gap, se_fixed_model=se))
    return pd.DataFrame(rows)


def re288_check(pitches, pas, roles, seasons=config.SEASONS):
    """League-pooled model value at each held-out pitch vs. the runs that followed.

    Each fold is compared with the model of the role that values it (fold k+2).
    Returns gap tables by count, by bases, by outs and overall.
    """
    emp = empirical_runs_to_end(pitches, pas)
    parts = []
    for season in seasons:
        for k in range(config.N_FOLDS):
            role = roles[season, arsenal.fold_roles(k)[1]]
            V = league_role_values(role, pitches[(pitches["season"] == season) & (pitches["fold"] == role.fold)])
            e = emp[(emp["season"] == season) & (emp["fold"] == k)]
            h = (e["stand"] == "R").to_numpy(int)
            parts.append(e.assign(model=V[h, e["base_out"].to_numpy(int), e["cidx"].to_numpy(int)]))
    df = pd.concat(parts, ignore_index=True)
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    return {
        "overall": _clustered(df.assign(all="all"), "all"),
        "by_count": _clustered(df, "count"),
        "by_bases": _clustered(df, "bases"),
        "by_outs": _clustered(df, "outs"),
        "by_season": _clustered(df, "season"),
    }
