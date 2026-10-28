"""Estimator-recovery simulation, version 2 (project_plan.md, pilot addendum).

A simulated world replaces each real 2024 at-bat's pitch sequence while keeping
its real game, fold, pitcher, catcher, batter, batter hand and starting base-out
state, so the unequal pitcher assignment of the real data is kept. Base-out is
fixed within the at-bat, as in the model.

Truth, fitted on all real 2024 games: the pilot's pooled outcome and contact-bin
models, each pitcher's reference policy pi0 over a true arsenal (the union of his
eligible sets across scoring folds) and the season's payoffs. Data-generating
models: "pilot" (the estimator's own model class) and "batter" (5 hidden batter
types with log-probability shifts, SD 0.3, which the estimator does not model).

Catcher policies: pi_c proportional to pi0 * exp(-kappa_c (Q_true - V_true)),
kappa_c = scale * z_c. For each (DGP, setting) the z_c are drawn once and held
fixed across replicates, so every target is a fixed population quantity and the
replicates measure the estimator's sampling distribution around it.

Targets (costs, runs per 100 pitches; every catcher evaluated on the SAME
predefined opportunities; each centered on the mean over qualifying catchers,
the same convention as the estimates):
- scientific: expected policy run value on common PA starts S0 (a fixed sample
  of real starting opportunities: pitcher-hand, batter type, base-out, 0-0),
  V^{pi_c} - V^{pi0}, solved exactly, divided by the expected pitches per PA
  under pi0 on S0. It depends on the catcher only through kappa_c, so it is a
  single function g(kappa). The setting (between-catcher SD 0, 0.01, 0.02, 0.05,
  0.08; a simulation range, not an empirical claim) calibrates the scale on it.
- D and A population targets: on common decision states S1 (a fixed sample of
  real scorable 2024 pitch states), the expected scored difference under the
  catcher's policy restricted to the estimator's eligible set:
  D_c = E_s sum_a pi_c(a|s) [Q_value(s,a) - Q_value(s, a_hat(s))], a_hat from the
  selection model (D's independent selection is kept); A_c = E_s sum_a pi_c(a|s)
  [Q_value(s,a) - V_value(s)]. Each replicate gives the target CONDITIONAL on its
  fitted training models; the population target AVERAGES these over replicates
  (repeated training samples). Expectations over actions are exact.
- diagnostic, not a target: true-optimum regret, E_s sum_a pi_c [Q_true - min Q_true].

Payoffs: the estimator's payoffs come from the real PAs of each fold, so all
results are CONDITIONAL on the real payoff estimates (simulated at-bats have no
runs). Model-based intervals (effect +- 1.96 pev_se) are diagnostics only; the
bootstrap intervals are validated separately (coverage_sim.py).

Run with: python3.12 -m gamecalling.simulation N_REPS [WORKERS]
Outputs: data/simulation/v2/ (manifest, worlds, replicates).
"""

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, payoff, ranking, scoring, solver

VERSION = "v2"
SIM_DIR = config.DATA_DIR / "simulation" / VERSION
SIM_SEED = 20261029
SEASON = 2024
DGPS = ("pilot", "batter")
TARGET_SDS = (0.0, 0.01, 0.02, 0.05, 0.08)
N_BATTER_TYPES = 5
BATTER_SHIFT_SD = 0.3
QUALIFY = 1000          # real scorable pitches for a catcher to enter metrics and calibration
N_START_OPPS = 3000     # size of S0
N_DECISION_STATES = 100_000  # size of S1


@dataclass
class World:
    dgp: str
    units: models.Units
    P: np.ndarray        # (T, U, 12, A, 6) true outcome probabilities by batter type
    PC: np.ndarray       # (U, A, 8) true contact-bin probabilities
    pi0: np.ndarray      # (U, 12, A) reference policy
    Q: np.ndarray        # (T, U, 24, 12, A) float32, inf where ineligible
    V: np.ndarray        # (T, U, 24, 12)
    batter_type: pd.Series
    walk: np.ndarray     # (24,) season payoffs
    strikeout: np.ndarray
    contact: np.ndarray  # (24, 8)
    alphas: dict | None = None  # pooling concentrations used: outcome and contact (league, cell), policy


def build_world(pitches, pas, support, dgp, season=SEASON, seed=SIM_SEED, alphas=None) -> World:
    """`alphas` ({"outcome": (league, cell), "contact": (league, cell), "policy": a}, entries optional)
    overrides the fitted pooling concentrations, for diagnostics only (refit_diagnostic.py)."""
    units, om, cm, pol = fit_world_models(pitches, support, season, alphas)
    P1, PC = models.unit_outcome_probs(om, units), models.unit_contact_probs(cm, units)
    used = {"outcome": (om.alpha_league, om.alpha_cell), "contact": (cm.alpha_league, cm.alpha_cell),
            "policy": pol.alpha}
    batters = np.sort(pitches.loc[pitches["season"] == season, "batter"].unique())
    return solve_world(units, P1, PC, pol.probs, pas, batters, dgp, season, seed, used)


def fit_world_models(pitches, support, season=SEASON, alphas=None):
    """A world's fitted pieces: units over the season's eligible sets, outcome and contact models, pi0."""
    alphas = alphas or {}
    p = pitches[pitches["season"] == season]
    elig = arsenal.eligible_sets(support)
    e = elig[elig["season"] == season][["pitcher", "stand", "action"]].drop_duplicates()
    units = models.build_units(e)
    om, cm = models.fit_outcome_model(p, alphas.get("outcome")), models.fit_contact_model(p, alphas.get("contact"))
    pol = models.fit_policy(models.policy_training(p), units, alphas.get("policy"))
    return units, om, cm, pol


def solve_world(units, P1, PC, pi0, pas, batters, dgp, season=SEASON, seed=SIM_SEED, alphas=None) -> World:
    """A world from given cell probabilities (P1 (U, 12, A, 6), PC (U, A, 8)) and pi0: batter types, Q and V."""
    rng = np.random.default_rng([seed, DGPS.index(dgp)])
    pay = payoff.role_payoffs(pas, season, None)
    if dgp == "pilot":
        shifts = np.zeros((1, 6))
        btype = pd.Series(0, index=batters)
    else:
        shifts = rng.normal(0, BATTER_SHIFT_SD, (N_BATTER_TYPES, 6))
        shifts[:, 4] = 0.0  # HBP unchanged
        btype = pd.Series(rng.integers(0, N_BATTER_TYPES, len(batters)), index=batters)
    logp = np.log(np.clip(P1, 1e-300, None))[None] + shifts[:, None, None, None, :]
    P = np.exp(logp - logp.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)

    bip = np.einsum("uak,bk->uba", PC, pay.contact)[:, :, None, :]
    elig_mask = units.eligible[:, None, None, :]
    Q, V = [], []
    for t in range(len(shifts)):
        v, q = solver.evaluate(P[t][:, None], bip, pay.walk, pay.strikeout, pi0[:, None], eligible=elig_mask)
        Q.append(q.astype(np.float32))
        V.append(v)
    return World(dgp, units, P, PC, pi0, np.stack(Q), np.stack(V), btype, pay.walk, pay.strikeout, pay.contact,
                 alphas)


def opportunities(pitches, pas, units, season=SEASON) -> pd.DataFrame:
    """Real PAs to re-simulate: not intentional walks, no sequence errors, pitcher-hand in the true units."""
    a = pas[(pas["season"] == season) & ~pas["pa_end"].eq("intent_walk") & ~pas["pa_sequence_error"]]
    pp = pitches.loc[pitches["season"] == season].drop_duplicates("pitcher").set_index("pitcher")["position_player_pitching"]
    a = a[~a["start_pitcher"].map(pp).fillna(False).astype(bool)]
    a = a.merge(units.table.rename(columns={"pitcher": "start_pitcher"}), on=["start_pitcher", "stand"])
    return a[["pa_id", "game_pk", "at_bat_number", "fold", "game_date", "fielding_team", "start_catcher",
              "start_pitcher", "batter", "stand", "start_base_out", "u"]].sort_values("pa_id").reset_index(drop=True)


def _policy(world, u, t, bo, c, kappa):
    """Calling probabilities (n, A): pi0 tilted by kappa toward lower true cost."""
    base = world.pi0[u, c]
    if not np.any(kappa):
        return base
    q = world.Q[t, u, bo, c].astype(float)
    adv = np.where(np.isfinite(q), q - world.V[t, u, bo, c][:, None], 0.0)
    with np.errstate(divide="ignore"):
        logits = np.log(base) - kappa[:, None] * adv
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(-1, keepdims=True)


def simulate(world, opps, kappa_by_catcher, rng) -> pd.DataFrame:
    """Simulate every opportunity's pitch sequence. Returns one row per pitch with oracle values."""
    n = len(opps)
    u = opps["u"].to_numpy()
    t = opps["batter"].map(world.batter_type).fillna(0).astype(int).to_numpy()
    bo = opps["start_base_out"].to_numpy(int)
    kap = kappa_by_catcher
    count = np.zeros(n, dtype=np.int64)
    pitch_no = np.zeros(n, dtype=np.int64)
    alive = np.arange(n)
    cum_c = np.cumsum(world.PC, -1)
    rows = []
    while len(alive):
        a_u, a_t, a_bo, c = u[alive], t[alive], bo[alive], count[alive]
        pi = _policy(world, a_u, a_t, a_bo, c, kap[alive])
        a = np.minimum((rng.random(len(alive))[:, None] > np.cumsum(pi, -1)).sum(-1), pi.shape[1] - 1)
        po = world.P[a_t, a_u, c, a]
        o = np.minimum((rng.random(len(alive))[:, None] > np.cumsum(po, -1)).sum(-1), 5)
        bins = np.zeros(len(alive), dtype=np.int64)
        bip = o == 5
        if bip.any():
            bins[bip] = np.minimum((rng.random(bip.sum())[:, None] > cum_c[a_u[bip], a[bip]]).sum(-1), 7) + 1
        q = world.Q[a_t, a_u, a_bo, c].astype(float)
        qf = np.where(np.isfinite(q), q, 0.0)
        pitch_no[alive] += 1
        rows.append(pd.DataFrame({
            "opp": alive, "pitch_number": pitch_no[alive], "cidx": c, "slot": a, "o": o, "contact_bin": bins,
            "Q_true": q[np.arange(len(alive)), a], "minQ_true": np.where(np.isfinite(q), q, np.inf).min(-1),
            "V_true": world.V[a_t, a_u, a_bo, c], "sci": ((pi - world.pi0[a_u, c]) * qf).sum(-1),
        }))
        b, s = np.divmod(c, 3)
        done = ((o == 0) & (b == 3)) | (o == 4) | (((o == 1) | (o == 2)) & (s == 2)) | bip
        count[alive] = np.where(o == 0, c + 3, np.where(s < 2, c + 1, c))
        alive = alive[~done]
    sim = pd.concat(rows, ignore_index=True)
    o = opps.iloc[sim["opp"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "season": SEASON, "fold": o["fold"].to_numpy(), "game_pk": o["game_pk"].to_numpy(),
        "pa_id": o["pa_id"].to_numpy(), "at_bat_number": o["at_bat_number"].to_numpy(),
        "pitch_number": sim["pitch_number"].to_numpy(), "game_date": o["game_date"].to_numpy(),
        "fielding_team": o["fielding_team"].to_numpy(), "catcher": o["start_catcher"].to_numpy(),
        "pitcher": o["start_pitcher"].to_numpy(), "batter": o["batter"].to_numpy(), "stand": o["stand"].to_numpy(),
        "balls": (sim["cidx"] // 3).astype("int8").to_numpy(), "strikes": (sim["cidx"] % 3).astype("int8").to_numpy(),
        "base_out": o["start_base_out"].astype("int8").to_numpy(),
        "action": world.units.actions[o["u"].to_numpy(), sim["slot"].to_numpy()],
        "outcome": np.array(solver.OUTCOMES, dtype=object)[sim["o"].to_numpy()],
        "contact_bin": pd.array(np.where(sim["o"] == 5, sim["contact_bin"], pd.NA), dtype="Int8"),
    })
    out["thrown"] = True
    out["dugout_called"] = False
    out["pa_sequence_error"] = False
    out["position_player_pitching"] = False
    out["exclusion_reason"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["outcome_fit_eligible"] = True
    out["contact_fit_eligible"] = out["outcome"].eq("in_play")
    for col in ("Q_true", "minQ_true", "V_true", "sci"):
        out[col] = sim[col].to_numpy()
    return out.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)




# --- Common opportunities and the scientific target g(kappa) ---

@dataclass
class Common:
    starts: pd.DataFrame     # S0: u, t, bo
    states: pd.DataFrame     # S1: u, t, bo, c, fold, pitcher, stand
    kappa_grid: np.ndarray
    g: np.ndarray            # scientific target (runs per 100 pitches, uncentered) on kappa_grid
    pitches_per_pa: float


def _expected_pitches(world, u, t) -> np.ndarray:
    """Expected pitches to the end of the PA under pi0, by count (n, 12)."""
    P, pi = world.P[t, u], world.pi0[u]                     # (n, 12, A, 6), (n, 12, A)
    N = np.zeros((len(u), solver.N_COUNTS))
    for _ in range(10_000):
        ext = np.concatenate([N, np.zeros((len(u), 2))], axis=1)
        cont = (P[..., 0] * ext[:, solver.AFTER_BALL, None] + (P[..., 1] + P[..., 2]) * ext[:, solver.AFTER_STRIKE, None]
                + P[..., 3] * ext[:, solver.AFTER_FOUL, None])
        N_new = (pi * (1.0 + cont)).sum(-1)
        if np.max(np.abs(N_new - N)) < 1e-10:
            return N_new
        N = N_new
    raise RuntimeError("expected pitches did not converge")


def _policy_values(world, pairs_u, pairs_t, kappa) -> np.ndarray:
    """V^{pi_kappa} (n, 24, 12) for (unit, batter type) pairs, pi_kappa tilted in every base-out state."""
    n, A = len(pairs_u), world.pi0.shape[-1]
    bo = np.arange(24)
    uu, tt = np.repeat(pairs_u, 24 * 12), np.repeat(pairs_t, 24 * 12)
    bb = np.tile(np.repeat(bo, 12), n)
    cc = np.tile(np.arange(12), 24 * n)
    pol = _policy(world, uu, tt, bb, cc, np.full(len(uu), kappa)).reshape(n, 24, 12, A)
    bip = np.einsum("uak,bk->uba", world.PC[pairs_u], world.contact)[:, :, None, :]
    elig = world.units.eligible[pairs_u][:, None, None, :]
    V, _ = solver.evaluate(world.P[pairs_t, pairs_u][:, None], bip, world.walk, world.strikeout, pol, eligible=elig)
    return V


def build_common(world, opps, real_states, seed=SIM_SEED, n_grid=61) -> Common:
    rng = np.random.default_rng([seed, 99])
    s0 = opps.iloc[rng.choice(len(opps), N_START_OPPS, replace=False)]
    starts = pd.DataFrame({"u": s0["u"].to_numpy(), "t": s0["batter"].map(world.batter_type).fillna(0).astype(int).to_numpy(),
                           "bo": s0["start_base_out"].to_numpy(int)})
    states = real_states.iloc[rng.choice(len(real_states), N_DECISION_STATES, replace=False)].reset_index(drop=True)
    pairs = starts[["u", "t"]].drop_duplicates().reset_index(drop=True)
    idx = starts.merge(pairs.reset_index(), on=["u", "t"])["index"].to_numpy()
    pu, pt = pairs["u"].to_numpy(), pairs["t"].to_numpy()
    N0 = _expected_pitches(world, pu, pt)[idx, 0].mean()
    V0 = world.V[starts["t"].to_numpy(), starts["u"].to_numpy(), starts["bo"].to_numpy(), 0]
    kmax = 400.0
    grid = np.concatenate([-np.geomspace(kmax, 0.5, n_grid // 2), [0.0], np.geomspace(0.5, kmax, n_grid // 2)])
    g = []
    for k in grid:
        Vk = _policy_values(world, pu, pt, k)[idx, starts["bo"].to_numpy(), 0]
        g.append(100 * (Vk - V0).mean() / N0)
    return Common(starts, states, grid, np.array(g), float(N0))


def scientific(common: Common, kappa: np.ndarray) -> np.ndarray:
    if np.any(np.abs(kappa) > common.kappa_grid.max()):
        raise ValueError("kappa outside the target grid")
    return np.interp(kappa, common.kappa_grid, common.g)


def calibrate(common: Common, z: np.ndarray, qualifying: np.ndarray, target_sd: float) -> float:
    """Scale with SD over qualifying catchers of g(scale * z) equal to target_sd."""
    if target_sd == 0:
        return 0.0
    sd = lambda s: scientific(common, s * z[qualifying]).std(ddof=1)  # noqa: E731
    lo, hi = 0.0, 0.999 * common.kappa_grid.max() / np.abs(z).max()
    if sd(hi) < target_sd:
        raise RuntimeError("target SD not reachable on the kappa grid")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sd(mid) < target_sd else (lo, mid)
    return (lo + hi) / 2


# --- D and A targets conditional on one replicate's fitted models ---

def conditional_targets(world, common, solutions, catchers, kappa) -> pd.DataFrame:
    """Per catcher: D, A (estimator's fitted models, eligible set, independent selection) and true regret."""
    true_slot = world.units.slots.set_index(["pitcher", "stand", "action"])["slot"]
    acc = {name: np.zeros(len(catchers)) for name in ("D", "A", "regret")}
    n_total = 0
    for k, sol in solutions.items():
        units, sel, val = sol["units"], sol["select"], sol["value"]
        st = common.states[common.states["fold"] == k].merge(units.table, on=["pitcher", "stand"], suffixes=("", "_e"))
        if st.empty:
            continue
        ue, ut = st["u_e"].to_numpy(), st["u"].to_numpy()
        t, bo, c = st["t"].to_numpy(), st["bo"].to_numpy(), st["c"].to_numpy()
        n = len(st)
        acts = units.actions[ue]                                   # (n, Ae)
        keys = pd.MultiIndex.from_arrays([np.repeat(st["pitcher"].to_numpy(), acts.shape[1]),
                                          np.repeat(st["stand"].to_numpy(), acts.shape[1]), acts.ravel()])
        ts = true_slot.reindex(keys).to_numpy().reshape(acts.shape)
        ok = units.eligible[ue] & ~np.isnan(ts)
        ts = np.where(ok, ts, 0).astype(int)
        qv, vv = val.Q_ref[ue, bo, c], val.V_ref[ue, bo, c]
        rec = sel.Q_ref[ue, bo, c].argmin(axis=1)
        qrec = qv[np.arange(n), rec]
        d_a = np.where(ok, qv - qrec[:, None], 0.0)
        a_a = np.where(ok, qv - vv[:, None], 0.0)
        qt = world.Q[t, ut, bo, c].astype(float)
        regret_a = np.where(np.isfinite(qt), qt - np.where(np.isfinite(qt), qt, np.inf).min(-1, keepdims=True), 0.0)
        for i, kap in enumerate(kappa):
            pi = _policy(world, ut, t, bo, c, np.full(n, kap))      # (n, At) over the true arsenal
            pe = np.where(ok, np.take_along_axis(pi, ts, axis=1), 0.0)
            pe /= pe.sum(1, keepdims=True)
            acc["D"][i] += (pe * d_a).sum()
            acc["A"][i] += (pe * a_a).sum()
            acc["regret"][i] += (pi * regret_a).sum()
        n_total += n
    return pd.DataFrame({"catcher": catchers, **{f"{k}_target": 100 * v / n_total for k, v in acc.items()}})


# --- One replicate ---

_CACHE = {}


def _load(dgp):
    if "pas" not in _CACHE:
        _CACHE["pas"] = pd.read_parquet(config.PAS_PARQUET)
        _CACHE["pitches"] = pd.read_parquet(config.PITCHES_PARQUET)
    if dgp not in _CACHE:
        with open(SIM_DIR / f"world_{dgp}.pkl", "rb") as f:
            _CACHE[dgp] = pickle.load(f)
        with open(SIM_DIR / f"common_{dgp}.pkl", "rb") as f:
            _CACHE[dgp, "common"] = pickle.load(f)
        _CACHE[dgp, "opps"] = opportunities(_CACHE["pitches"], _CACHE["pas"], _CACHE[dgp].units)
    return _CACHE["pas"], _CACHE[dgp], _CACHE[dgp, "common"], _CACHE[dgp, "opps"]


def real_states(pitches, world) -> pd.DataFrame:
    real = pitches[(pitches["season"] == SEASON) & pitches["scorable"]]
    st = real.merge(world.units.table, on=["pitcher", "stand"])
    return pd.DataFrame({
        "u": st["u"].to_numpy(), "t": st["batter"].map(world.batter_type).fillna(0).astype(int).to_numpy(),
        "bo": st["base_out"].to_numpy(int), "c": (st["balls"] * 3 + st["strikes"]).to_numpy(int),
        "fold": st["fold"].to_numpy(int), "pitcher": st["pitcher"].to_numpy(), "stand": st["stand"].to_numpy(),
        "catcher": st["catcher"].to_numpy()})


def catcher_effects(opps, pitches, dgp, target_sd, seed=SIM_SEED):
    """Fixed catchers, qualifying flags, z and kappa for one (DGP, setting)."""
    _, world, common, _ = _load(dgp)
    catchers = np.sort(opps["start_catcher"].unique())
    real = pitches[(pitches["season"] == SEASON) & pitches["scorable"]]
    n_real = real.groupby("catcher").size().reindex(catchers, fill_value=0).to_numpy()
    qualifying = n_real >= QUALIFY
    z = np.random.default_rng([seed, DGPS.index(dgp), 7]).normal(size=len(catchers))  # same z for every setting
    scale = calibrate(common, z, qualifying, target_sd)
    return catchers, qualifying, z, scale * z, scale


def estimate(sim, pas, keep_solutions=False):
    support = arsenal.support_table(sim)
    marked = arsenal.mark_arsenal(sim, arsenal.eligible_sets(support))
    run = scoring.run_pilot(marked, pas, support, seasons=(SEASON,), with_re288=False, keep_solutions=keep_solutions)
    fits = {col: ranking.fit(run.scores, col) for col in ("D", "A")}
    return run, fits


def run_rep(dgp: str, target_sd: float, rep: int, seed: int = SIM_SEED) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    pas, world, common, opps = _load(dgp)
    catchers, qualifying, z, kappa, scale = catcher_effects(opps, _CACHE["pitches"], dgp, target_sd, seed)
    rng = np.random.default_rng([seed, DGPS.index(dgp), int(round(target_sd * 1000)), rep])
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    sim = simulate(world, opps, kappa[opps["start_catcher"].map(code).to_numpy()], rng)
    run, fits = estimate(sim, pas, keep_solutions=True)
    sols = {k: v for (s, k), v in run.solutions.items()}
    tg = conditional_targets(world, common, sols, catchers, kappa)
    out = pd.DataFrame({"catcher": catchers, "qualifying": qualifying, "z": z, "kappa": kappa,
                        "scientific": scientific(common, kappa)}).merge(tg, on="catcher")
    for col, f in fits.items():
        c = f.catchers.set_index("catcher")
        out[f"{col}_effect"] = c["effect"].reindex(catchers).to_numpy()
        out[f"{col}_pev_se"] = c["pev_se"].reindex(catchers).to_numpy()
        out[f"{col}_sd_catcher"] = float(np.sqrt(f.variances["catcher"]))
        out[f"{col}_converged"] = f.converged
    out["n_scored"] = run.scores.groupby("catcher").size().reindex(catchers, fill_value=0).to_numpy()
    out = out.assign(dgp=dgp, target_sd_setting=target_sd, rep=rep, kappa_scale=scale)
    return out, {"dgp": dgp, "target_sd": target_sd, "rep": rep, "seconds": time.perf_counter() - t0}


def _task(args):
    dgp, sd, rep = args
    path = SIM_DIR / f"rep_{dgp}_{int(round(sd * 1000)):03d}_{rep:03d}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_rep(dgp, sd, rep)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def manifest() -> dict:
    return {"version": VERSION, "seed": SIM_SEED, "season": SEASON, "dgps": list(DGPS), "target_sds": list(TARGET_SDS),
            "n_batter_types": N_BATTER_TYPES, "batter_shift_sd": BATTER_SHIFT_SD, "qualify": QUALIFY,
            "n_start_opps": N_START_OPPS, "n_decision_states": N_DECISION_STATES,
            "targets": "scientific on common PA starts; D/A conditional on fitted models on common decision states; "
                       "population = mean over replicates", "payoffs": "conditional on real PA payoff estimates",
            "catcher_effects": "z fixed per DGP across settings and replicates"}


def prepare():
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    man = SIM_DIR / "manifest.json"
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest())):
        raise ValueError(f"{SIM_DIR} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(), indent=1))
    pitches = pd.read_parquet(config.PITCHES_PARQUET)
    pas = pd.read_parquet(config.PAS_PARQUET)
    support = pd.read_parquet(config.DATA_DIR / "arsenal_support.parquet")
    for dgp in DGPS:
        wp, cp = SIM_DIR / f"world_{dgp}.pkl", SIM_DIR / f"common_{dgp}.pkl"
        if not wp.exists():
            with open(wp, "wb") as f:
                pickle.dump(build_world(pitches, pas, support, dgp), f)
        if not cp.exists():
            with open(wp, "rb") as f:
                world = pickle.load(f)
            common = build_common(world, opportunities(pitches, pas, world.units), real_states(pitches, world))
            with open(cp, "wb") as f:
                pickle.dump(common, f)


def load_results() -> pd.DataFrame:
    parts = []
    for f in sorted(SIM_DIR.glob("rep_*.pkl")):
        with open(f, "rb") as fh:
            parts.append(pickle.load(fh)[0])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per (DGP, setting, score, target) over qualifying catchers; population targets per catcher.

    Estimates and targets are centered on the mean over qualifying catchers within
    each replicate. The population D/A target is the mean over replicates of the
    conditional targets (Monte Carlo SE reported).
    """
    q = res[res["qualifying"]].copy()
    key = ["dgp", "target_sd_setting", "rep"]
    for col in ("D_effect", "A_effect", "scientific", "D_target", "A_target", "regret_target"):
        q[col + "_c"] = q[col] - q.groupby(key)[col].transform("mean")
    pop = q.groupby(["dgp", "target_sd_setting", "catcher"]).agg(
        D_pop=("D_target_c", "mean"), D_pop_mcse=("D_target_c", "sem"),
        A_pop=("A_target_c", "mean"), A_pop_mcse=("A_target_c", "sem"),
        D_cond_sd=("D_target_c", "std"), A_cond_sd=("A_target_c", "std")).reset_index()
    q = q.merge(pop, on=["dgp", "target_sd_setting", "catcher"])
    rows = []
    for (dgp, sd), g in q.groupby(["dgp", "target_sd_setting"]):
        for score in ("D", "A"):
            for tname, tcol in (("scientific", "scientific_c"), ("own population", f"{score}_pop"),
                                ("own conditional", f"{score}_target_c"), ("true regret (diagnostic)", "regret_target_c")):
                per_rep = []
                for rep, r in g.groupby("rep"):
                    est, tgt, se = r[f"{score}_effect_c"], r[tcol], r[f"{score}_pev_se"]
                    err = est - tgt
                    var = tgt.std() > 1e-6
                    per_rep.append(dict(
                        rmse=np.sqrt((err ** 2).mean()), coverage_model=(err.abs() <= 1.96 * se).mean(),
                        flagged=(est.abs() > 1.96 * se).mean(),
                        corr=np.corrcoef(est, tgt)[0, 1] if var else np.nan,
                        slope=np.polyfit(tgt, est, 1)[0] if var else np.nan,
                        target_sd=tgt.std(), est_sd=est.std(), sd_catcher_reml=r[f"{score}_sd_catcher"].iloc[0]))
                pr = pd.DataFrame(per_rep)
                row = dict(dgp=dgp, target_sd_setting=sd, score=score, target=tname, reps=len(pr))
                for c in pr:
                    row[c] = pr[c].mean()
                    row[c + "_mcse"] = pr[c].std(ddof=1) / np.sqrt(len(pr)) if len(pr) > 1 else np.nan
                rows.append(row)
    return pd.DataFrame(rows), pop


def run_all(n_reps: int, workers: int = 4, settings=TARGET_SDS, dgps=DGPS):
    prepare()
    tasks = [(d, sd, r) for d in dgps for sd in settings for r in range(n_reps)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_task, tasks))
    return pd.DataFrame([t for _, t in results])


if __name__ == "__main__":
    n = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t0 = time.perf_counter()
    tm = run_all(n, workers)
    print(f"{len(tm)} simulation replicates in {time.perf_counter() - t0:.0f} s; mean {tm['seconds'].mean():.0f} s each")
