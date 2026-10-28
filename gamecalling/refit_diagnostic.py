"""Refit-world diagnostic (research_notes/uncertainty_design/04_refit_world_diagnostic.md, revision 2).

Simulation only, fixed payoffs (the real PA payoff estimates), season 2024. Nothing here touches real
catcher results. Stages run separately, in order, each inspected before the next.

§2, per true world (pilot, batter) and outer index j < J:
- outer: a dataset from the true world at scientific SD exactly 0.05 with fresh normal z; its observed
  T (REML catcher SD of D, and of A); the world W_hat refit from it (as the real analysis would).
- ref (N per outer): a dataset from that outer's W_hat at exactly 0.05 (calibrated in W_hat, fresh z).
- control: the 20 `check` tasks per world of the debugging stage (true world, 0.05, fresh normal z,
  seed 20261030), reused without new fits; Monte Carlo p-values with denominator 21.

§3, world-by-world comparison (the 6 §2 refit worlds and the 2 debugging refit worlds against their true
world; no estimator fits except h0):
- world: rebuilds the world from its re-drawn simulated pitches (checked against the stored world) to
  record the pooling concentrations, then compares cells on matched (pitcher, hand, pitch type): spread of
  pitcher cells around the league level, RMS difference from the true world, pi0 total variation, the
  pi0-weighted SD of Q - V on S1, g(kappa) and kappa per unit of scientific SD with one common z.
- h0 (N_H0 per §2 refit world and in the true batter world): data with every catcher at pi0, the
  conditional target curve; the debugging stage's h0 curves are reused for the other worlds.

§4, ablations on outer dataset 1 of each true world: refit with alpha fixed at the true world's values,
from three pooled realizations ("more data"), with minimal pitcher pooling, and from the sigma = 0 twin;
N_VREF references at 0.05 from each variant world. §4b (spec 05 §3): the same with only the outcome and contact
pitcher-cell alpha fixed at the true world's values (league alpha and pi0's alpha re-estimated).

§5, D's per-catcher offsets: N_ASSIGN datasets from the true pilot world with every catcher at pi0,
simulated and scored once, ranked (D, A, B) under four label arms: real; primary catcher per appearance;
primary labels shuffled within team and fold; shuffled within fold. Shuffles are drawn once.

Run with: python3.12 -m gamecalling.refit_diagnostic [STAGE] [WORKERS]   (STAGE s2, s3, s4, s4b or s5)
Outputs: data/refit_diagnostic/v1/ (outer_*.pkl, tasks/*.pkl, s4/, s5/, a manifest per stage) and
reports/refit_diagnostic.md (every finished stage). Finished tasks are skipped on rerun.
"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gamecalling import arsenal, config, models, network, pooling, ranking, simulation as S, uncertainty_debug as UD
from gamecalling.phase1 import _md

VERSION = "v1"
OUT = config.DATA_DIR / "refit_diagnostic" / VERSION
REPORT = config.REPORTS_DIR / "refit_diagnostic.md"
SEED = 20261031
TEST_SD = 0.05
J = 3            # outer datasets per true world
N_REF = 19       # reference datasets per refit world
N_CONTROL = 20   # reused debugging `check` tasks per world
P_CUT = 0.10     # §2 reading rule: p <= 0.10 ...
N_TRIGGER = 4    # ... in at least 4 of the 6 outer datasets
N_H0 = 5         # §3: h0 training samples per §2 refit world and in the true batter world
VARIANTS = ("alpha_fixed", "more_data", "min_pool", "twin")  # §4
VARIANTS_B = ("alpha_cell_fixed",)  # §4b (spec 05 §3); stream index after VARIANTS
N_VREF = 9       # §4: references per variant world
N_MORE = 3       # §4: realizations pooled in "more data"
POOL_MIN = float(np.exp(pooling.LOG_ALPHA_BOUNDS[0]))  # §4: minimal pitcher pooling (0.01)
N_ASSIGN = 20    # §5: simulated and scored datasets
ARMS = ("real", "primary", "team_shuffle", "league_shuffle")
S5_SCORES = ("D", "A", "B")
STAGES = ("s2", "s3", "s4", "s4b", "s5")

# A directory here runs a tiny version for testing the code (read by worker processes too).
SMOKE_DIR = os.environ.get("REFIT_DIAGNOSTIC_SMOKE")
if SMOKE_DIR:
    OUT, REPORT = Path(SMOKE_DIR), Path(SMOKE_DIR) / "report.md"
    J, N_REF, N_H0, N_VREF, N_ASSIGN = 1, 2, 1, 1, 2
S4_J = min(1, J - 1)  # §4 base: outer dataset 1 of each true world

# Streams under SEED: 1 outer, 2 §2 reference, 3 §3 h0 in refit worlds, 4 §3 h0 in true worlds,
# 5 §4 extra realizations, 6 §4 references, 7 §5 datasets, 8 §5 shuffles, 9 §3 common z.


def manifest(stage: str = "s2") -> dict:
    out = {"version": VERSION, "simulation_version": S.VERSION, "seed": SEED, "test_sd": TEST_SD, "j": J,
           "n_ref": N_REF, "control": f"uncertainty_debug {UD.VERSION} check tasks (seed {UD.SEED})",
           "payoffs": "fixed (real PA payoff estimates)",
           "design": "research_notes/uncertainty_design/04_refit_world_diagnostic.md revision 2, §2"}
    if stage == "s2":
        return out
    extra = {"s3": {"n_h0": N_H0, "debug_worlds": list(UD.OUTER_SDS)},
             "s4": {"base_outer": S4_J, "variants": list(VARIANTS), "n_vref": N_VREF, "n_more": N_MORE,
                    "pool_min": POOL_MIN},
             "s4b": {"base_outer": S4_J, "variants": list(VARIANTS_B), "n_vref": N_VREF},
             "s5": {"n_assign": N_ASSIGN, "arms": list(ARMS), "scores": list(S5_SCORES)}}[stage]
    design = ("research_notes/uncertainty_design/05_hierarchy_test.md revision 1, §3" if stage == "s4b"
              else out["design"].replace("§2", f"§{stage[1]}"))
    return {**out, "stage": stage, **extra, "design": design}


def _manifest_path(stage: str) -> Path:
    return OUT / ("manifest.json" if stage == "s2" else f"manifest_{stage}.json")


def _outer_path(dgp: str, j: int) -> Path:
    return OUT / f"outer_{dgp}_{j}.pkl"


# --- Shared pieces ---

def _draw_outer(dgp: str, j: int, kappa_zero: bool = False):
    """An outer dataset's simulated pitches, drawn from its stream (re-drawing gives the stored outer's).

    kappa_zero: the sigma = 0 twin, with every catcher at pi0 and the same stream for everything else.
    """
    _, world, common, opps = S._load(dgp)
    catchers, qualifying = UD._catchers(opps, S._CACHE["pitches"])
    rng = np.random.default_rng([SEED, 1, S.DGPS.index(dgp), j])
    z = rng.normal(size=len(catchers))
    scale = S.calibrate(common, z, qualifying, TEST_SD)
    kappa = np.zeros(len(catchers)) if kappa_zero else scale * z
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    sim = S.simulate(world, opps, kappa[opps["start_catcher"].map(code).to_numpy()], rng)
    return catchers, qualifying, z, scale, sim


def _draw_debug_outer(sd: float) -> pd.DataFrame:
    """A debugging-stage outer dataset's simulated pitches, re-drawn as in uncertainty_debug.prepare_outer."""
    _, world, _, opps = S._load("pilot")
    catchers, _, _, kappa, _ = S.catcher_effects(opps, S._CACHE["pitches"], "pilot", sd)
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    rng = np.random.default_rng([UD.SEED, 1, int(round(sd * 1000))])
    return S.simulate(world, opps, kappa[opps["start_catcher"].map(code).to_numpy()], rng)


def _refit(sim: pd.DataFrame, pas: pd.DataFrame, alphas=None) -> dict:
    """The world refit from simulated pitches (pilot model class), with its opportunities, S0/S1 and catchers."""
    pitches = S._CACHE["pitches"]
    w_hat = S.build_world(sim, pas, arsenal.support_table(sim), "pilot", alphas=alphas)
    opps_hat = S.opportunities(pitches, pas, w_hat.units)
    common_hat = S.build_common(w_hat, opps_hat, S.real_states(pitches, w_hat))
    cat_hat, qual_hat = UD._catchers(opps_hat, pitches)
    return {"w_hat": w_hat, "common_hat": common_hat, "opps_hat": opps_hat, "catchers_hat": cat_hat,
            "qualifying_hat": qual_hat}


def _reference(o: dict, rng, pas) -> dict:
    """One reference dataset from a refit world at exactly TEST_SD (calibrated in it, fresh z): its T."""
    z = rng.normal(size=len(o["catchers_hat"]))
    scale = S.calibrate(o["common_hat"], z, o["qualifying_hat"], TEST_SD)
    fits, _ = UD._simulate_estimate(o["w_hat"], o["opps_hat"], o["catchers_hat"], scale * z, rng, pas,
                                    keep_solutions=False)
    return {"scale": scale, **UD._fit_record(fits)}


# --- §2 stage A: outer datasets and their refit worlds ---

def prepare_outer(args: tuple) -> str:
    dgp, j = args
    path = _outer_path(dgp, j)
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    pas = S._load(dgp)[0]
    catchers, qualifying, z, scale, sim = _draw_outer(dgp, j)
    _, fits = S.estimate(sim, pas)
    out = {"dgp": dgp, "j": j, **_refit(sim, pas), "z": pd.Series(z, index=catchers), "scale_true": scale,
           "observed": UD._effects(fits, catchers).assign(qualifying=qualifying, kappa=scale * z),
           "observed_fit": UD._fit_record(fits), "seconds": time.perf_counter() - t0}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"outer {dgp} {j}: prepared in {out['seconds']:.0f} s", flush=True)
    return str(path)


_OUTER, _VARIANT = {}, {}


def _outer(dgp: str, j: int) -> dict:
    if (dgp, j) not in _OUTER:
        with open(_outer_path(dgp, j), "rb") as f:
            _OUTER[dgp, j] = pickle.load(f)
    return _OUTER[dgp, j]


# --- §4 stage A: variant worlds ---

def _variant_path(name: str, dgp: str) -> Path:
    return OUT / "s4" / f"variant_{name}_{dgp}.pkl"


def _variant(name: str, dgp: str) -> dict:
    if (name, dgp) not in _VARIANT:
        with open(_variant_path(name, dgp), "rb") as f:
            _VARIANT[name, dgp] = pickle.load(f)
    return _VARIANT[name, dgp]


def prepare_variant(args: tuple) -> str:
    name, dgp = args
    path = _variant_path(name, dgp)
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    pas, world, _, opps = S._load(dgp)
    catchers, _, z, scale, sim = _draw_outer(dgp, S4_J, kappa_zero=(name == "twin"))
    alphas = None
    if name == "alpha_fixed":
        alphas = _load_task("world", "true", dgp)["alphas"]
    elif name == "alpha_cell_fixed":
        a = _load_task("world", "true", dgp)["alphas"]
        alphas = {"outcome": (None, a["outcome"][1]), "contact": (None, a["contact"][1])}
    elif name == "min_pool":
        alphas = {"outcome": (None, POOL_MIN), "contact": (None, POOL_MIN)}
    elif name == "more_data":
        # Independent realizations of the same 2024 design at the same kappa, with unique game and PA ids.
        code = pd.Series(np.arange(len(catchers)), index=catchers)
        k_opp = (scale * z)[opps["start_catcher"].map(code).to_numpy()]
        parts = [sim]
        for r in range(1, N_MORE):
            s = S.simulate(world, opps, k_opp, np.random.default_rng([SEED, 5, S.DGPS.index(dgp), S4_J, r]))
            parts.append(s.assign(game_pk=s["game_pk"] + r * 10 ** 7, pa_id=s["pa_id"] + r * 10 ** 9))
        sim = pd.concat(parts, ignore_index=True).sort_values(
            ["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    o = _refit(sim, pas, alphas)
    o.update(name=name, dgp=dgp, n_sim_pitches=len(sim), alphas=o["w_hat"].alphas,
             metrics=_world_metrics(o["w_hat"], o["common_hat"], o["catchers_hat"], o["qualifying_hat"], world),
             seconds=time.perf_counter() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(o, f)
    print(f"variant {name} {dgp}: prepared in {o['seconds']:.0f} s", flush=True)
    return str(path)


# --- §5 stage A: label arms ---

def _arms_path() -> Path:
    return OUT / "s5" / "arms.pkl"


def prepare_arms() -> pd.DataFrame:
    """Per appearance (game x fielding team): its primary catcher and the two shuffled labels.

    Primary: most real 2024 pitches received in the appearance (ties: the earliest), so it is the same
    for every simulated dataset. Shuffles permute primary labels among appearances of the same fielding
    team and fold, or of the same fold; each is drawn once.
    """
    path = _arms_path()
    if path.exists():
        return pd.read_pickle(path)
    _, _, _, opps = S._load("pilot")
    pitches = S._CACHE["pitches"]
    app = opps[["game_pk", "fielding_team", "fold"]].drop_duplicates().reset_index(drop=True)
    if app.duplicated(["game_pk", "fielding_team"]).any():
        raise ValueError("an appearance spans two folds")
    real = pitches[(pitches["season"] == S.SEASON) & pitches["thrown"]].sort_values(
        ["game_pk", "at_bat_number", "pitch_number"], kind="stable")
    real = real.assign(_order=np.arange(len(real)))
    by = real.groupby(["game_pk", "fielding_team", "catcher"]).agg(n=("_order", "size"), first=("_order", "min"))
    prim = by.reset_index().sort_values(["game_pk", "fielding_team", "n", "first"],
                                        ascending=[True, True, False, True]).drop_duplicates(["game_pk", "fielding_team"])
    app = app.merge(prim[["game_pk", "fielding_team", "catcher"]].rename(columns={"catcher": "primary"}),
                    on=["game_pk", "fielding_team"], how="left")
    if app["primary"].isna().any():
        raise ValueError("appearance without real pitches")
    for arm, keys, stream in (("team_shuffle", ["fielding_team", "fold"], 1), ("league_shuffle", ["fold"], 2)):
        rng = np.random.default_rng([SEED, 8, stream])
        lab = app["primary"].copy()
        for _, idx in sorted(app.groupby(keys).groups.items()):
            lab.loc[idx] = rng.permutation(app.loc[idx, "primary"].to_numpy())
        app[arm] = lab
    path.parent.mkdir(parents=True, exist_ok=True)
    app.to_pickle(path)
    return app


# --- §3 world metrics ---

_MEMO = {}


def _real() -> pd.DataFrame:
    """Real 2024 scorable pitches: the fixed weights for comparing worlds."""
    if "real" not in _MEMO:
        S._load("pilot")
        p = S._CACHE["pitches"]
        _MEMO["real"] = models.with_count_index(p[(p["season"] == S.SEASON) & p["scorable"]])
    return _MEMO["real"]


def _real_n(keys: pd.DataFrame) -> np.ndarray:
    by = list(keys.columns)
    n = _real().groupby(by).size().rename("_n").reset_index()
    return keys.merge(n, on=by, how="left")["_n"].fillna(0.0).to_numpy()


def _tables(world) -> dict:
    """Long tables of a world's cells: outcome probabilities (batter types mixed by their real pitch shares)
    and pi0 by (pitcher, stand, action, count); contact-bin probabilities by (pitcher, stand, action)."""
    real = _real()
    share = (real["batter"].map(world.batter_type).fillna(0).astype(int).value_counts(normalize=True)
             .reindex(range(world.P.shape[0]), fill_value=0.0).to_numpy())
    P = np.einsum("t,tucak->ucak", share, world.P)
    s = world.units.slots.reset_index(drop=True)
    cells = s.merge(pd.DataFrame({"cidx": np.arange(12)}), how="cross")
    u, c, a = (cells[k].to_numpy() for k in ("u", "cidx", "slot"))
    keys = cells[["pitcher", "stand", "action", "cidx"]]
    return {"outcome": (keys, P[u, c, a]), "pi0": (keys, world.pi0[u, c, a]),
            "contact": (s[["pitcher", "stand", "action"]], world.PC[s["u"].to_numpy(), s["slot"].to_numpy()])}


def _spread(keys: pd.DataFrame, probs: np.ndarray, league) -> float:
    """Real-pitch-weighted RMS (Euclidean over categories) distance of pitcher cells from the league mean."""
    w = _real_n(keys)
    keep = w > 0
    k, p, w = keys[keep], probs[keep], w[keep]
    grp = k.groupby(list(league), sort=False).ngroup().to_numpy()
    W = np.bincount(grp, w)
    m = np.stack([np.bincount(grp, w * p[:, i]) for i in range(p.shape[1])], 1) / W[:, None]
    return float(np.sqrt(np.average(((p - m[grp]) ** 2).sum(1), weights=w)))


def _compare(ta: dict, tb: dict) -> dict:
    """World a against world b on matched (pitcher, hand, pitch type[, count]); weights are real pitches."""
    slot = ["pitcher", "stand", "action"]
    sa, sb = ta["contact"][0], tb["contact"][0]
    out = {}
    for side, (x, y) in (("a", (sa, sb)), ("b", (sb, sa))):
        hit = x.merge(y.assign(_m=True), on=slot, how="left")["_m"].eq(True).to_numpy()
        w = _real_n(x)
        out[f"unmatched_slots_{side}"] = float(1 - hit.mean())
        out[f"unmatched_pitches_{side}"] = float(w[~hit].sum() / w.sum())
    for name in ("outcome", "contact"):
        (ka, pa), (kb, pb) = ta[name], tb[name]
        on = list(ka.columns)
        m = ka.assign(_i=np.arange(len(ka))).merge(kb.assign(_j=np.arange(len(kb))), on=on)
        d2 = ((pa[m["_i"].to_numpy()] - pb[m["_j"].to_numpy()]) ** 2).sum(1)
        out[f"{name}_rms_diff"] = float(np.sqrt(np.average(d2, weights=_real_n(m[on]))))
    # pi0: total variation on units (pitcher, hand) in both worlds; an action outside a world's set has probability 0.
    (ka, pa), (kb, pb) = ta["pi0"], tb["pi0"]
    units = ka[["pitcher", "stand"]].drop_duplicates().merge(kb[["pitcher", "stand"]].drop_duplicates())
    on = ["pitcher", "stand", "cidx", "action"]
    m = ka.assign(pa=pa).merge(units).merge(kb.assign(pb=pb).merge(units), on=on, how="outer")
    m = m.fillna({"pa": 0.0, "pb": 0.0})
    tv = (m["pa"] - m["pb"]).abs().groupby([m["pitcher"], m["stand"], m["cidx"]]).sum().mul(0.5)
    tv = tv.rename("tv").reset_index()
    out["pi0_tv"] = float(np.average(tv["tv"], weights=_real_n(tv[["pitcher", "stand", "cidx"]])))
    return out


def _action_values(world, common) -> dict:
    """On S1: pi0-weighted SD of Q - V and the mean gap V - min Q (runs x 100)."""
    st = common.states
    u, t, bo, c = (st[k].to_numpy() for k in ("u", "t", "bo", "c"))
    q = world.Q[t, u, bo, c].astype(float)
    v = world.V[t, u, bo, c]
    fin = np.isfinite(q)
    adv = np.where(fin, q - v[:, None], 0.0)
    return {"qv_sd": float(100 * np.sqrt(np.mean((world.pi0[u, c] * adv ** 2).sum(1)))),
            "best_gap": float(100 * np.mean(v - np.where(fin, q, np.inf).min(1)))}


def _common_z(catchers) -> np.ndarray:
    """One z per catcher id, shared by every world (stream 9)."""
    if "z" not in _MEMO:
        pas = S._load("pilot")[0]
        allc = np.sort(pas.loc[pas["season"] == S.SEASON, "start_catcher"].dropna().unique())
        _MEMO["z"] = pd.Series(np.random.default_rng([SEED, 9]).normal(size=len(allc)), index=allc)
    z = _MEMO["z"].reindex(catchers).to_numpy()
    if np.isnan(z).any():
        raise ValueError("catcher without a common z")
    return z


def _g_metrics(common, catchers, qualifying) -> dict:
    scale = S.calibrate(common, _common_z(catchers), qualifying, TEST_SD)
    out = {"kappa_per_sd": scale / TEST_SD}
    for k in (-100, 100):
        out[f"g({k})"] = float(np.interp(k, common.kappa_grid, common.g))
    return out


def _world_metrics(world, common, catchers, qualifying, true_world=None) -> dict:
    t = _tables(world)
    out = {"outcome_spread": _spread(*t["outcome"], ("action", "stand", "cidx")),
           "contact_spread": _spread(*t["contact"], ("action", "stand")),
           **_action_values(world, common), **_g_metrics(common, catchers, qualifying)}
    if true_world is not None:
        out.update(_compare(t, _tables(true_world)))
    return out


# --- Stage B: tasks ---

def run_task(task: tuple):
    kind = task[0]
    path = OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl")
    if path.exists():
        return kind
    t0 = time.perf_counter()
    pas = S._load("pilot")[0]
    pitches = S._CACHE["pitches"]
    if kind == "ref":
        _, dgp, j, r = task
        res = _reference(_outer(dgp, j), np.random.default_rng([SEED, 2, S.DGPS.index(dgp), j, r]), pas)
    elif kind == "world":
        src, key = task[1], task[2:]
        if src == "true":
            dgp = key[0]
            _, world, common, opps = S._load(dgp)
            catchers, qual = UD._catchers(opps, pitches)
            support = pd.read_parquet(config.DATA_DIR / "arsenal_support.parquet")
            rebuilt, truth = S.build_world(pitches, pas, support, dgp), None
        else:
            if src == "s2":
                dgp, o, sim = key[0], _outer(*key), _draw_outer(*key)[4]
            else:
                dgp, o, sim = "pilot", UD._outer(key[0]), _draw_debug_outer(key[0])
            world, common, catchers, qual = o["w_hat"], o["common_hat"], o["catchers_hat"], o["qualifying_hat"]
            rebuilt, truth = S.build_world(sim, pas, arsenal.support_table(sim), "pilot"), S._load(dgp)[1]
        same = rebuilt.units.slots.equals(world.units.slots)
        diff = max(float(np.abs(rebuilt.P - world.P).max()), float(np.abs(rebuilt.PC - world.PC).max()),
                   float(np.abs(rebuilt.pi0 - world.pi0).max())) if same else np.inf
        res = {"alphas": rebuilt.alphas, "rebuild_max_diff": diff,
               **_world_metrics(world, common, catchers, qual, truth)}
    elif kind == "h0":
        if task[1] == "s2":
            _, _, dgp, j, r = task
            o = _outer(dgp, j)
            world, common, opps, catchers = o["w_hat"], o["common_hat"], o["opps_hat"], o["catchers_hat"]
            rng = np.random.default_rng([SEED, 3, S.DGPS.index(dgp), j, r])
        else:
            _, _, dgp, r = task
            _, world, common, opps = S._load(dgp)
            catchers, _ = UD._catchers(opps, pitches)
            rng = np.random.default_rng([SEED, 4, S.DGPS.index(dgp), r])
        fits, sols = UD._simulate_estimate(world, opps, catchers, np.zeros(len(catchers)), rng, pas)
        res = {"curve": UD._curve(world, common, sols), **UD._fit_record(fits)}
    elif kind == "vref":
        _, name, dgp, r = task
        rng = np.random.default_rng([SEED, 6, S.DGPS.index(dgp), (VARIANTS + VARIANTS_B).index(name), r])
        res = _reference(_variant(name, dgp), rng, pas)
    elif kind == "assign":
        res = _assign_task(task[1], pas)
    else:
        raise ValueError(kind)
    res["seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print(f"{path.stem}: {res['seconds']:.0f} s", flush=True)
    return kind


def _assign_task(r: int, pas) -> dict:
    """§5: one dataset from the true pilot world at sigma = 0, scored once, ranked under every label arm."""
    _, world, _, opps = S._load("pilot")
    pitches = S._CACHE["pitches"]
    catchers, qualifying = UD._catchers(opps, pitches)
    rng = np.random.default_rng([SEED, 7, r])
    run, fits = S.estimate(S.simulate(world, opps, np.zeros(len(opps)), rng), pas)
    sc = run.scores.merge(pd.read_pickle(_arms_path()), on=["game_pk", "fielding_team"], how="left",
                          validate="many_to_one")
    if sc["primary"].isna().any():
        raise ValueError("scored pitch outside the arms table")
    u = world.units.table.set_index(["pitcher", "stand"])["u"].reindex(
        pd.MultiIndex.from_frame(sc[["pitcher", "stand"]])).to_numpy()
    sc["v00"] = world.V[0, u, 0, 0]   # staff quality: true V at 0-0, bases empty, nobody out
    qual_set = set(catchers[qualifying])
    arms = {}
    for arm in ARMS:
        s = sc if arm == "real" else sc.assign(catcher=sc[arm])
        arm_fits = {col: fits[col] if arm == "real" and col in fits else ranking.fit(s, col) for col in S5_SCORES}
        g = s.groupby("catcher")
        cov = pd.DataFrame({"n": g.size(), "staff_v00": g["v00"].mean(), "n_eligible": g["n_eligible"].mean(),
                            "top_pitcher_share": s.groupby(["catcher", "pitcher"]).size().groupby("catcher").max()
                            / g.size()})
        comps = network.components(network.catcher_graph(s, qualifying=qual_set))
        arms[arm] = {"effects": UD._effects(arm_fits, catchers), "fit": UD._fit_record(arm_fits), "cov": cov,
                     "groups": len(comps), "main_share": len(comps[0]) / len(qual_set)}
    return {"arms": arms, "secondary_share": float((sc["catcher"] != sc["primary"]).mean()), "n_pitches": len(sc)}


def outers() -> list:
    return [(dgp, j) for dgp in S.DGPS for j in range(J)]


def tasks() -> list:
    return [("ref", dgp, j, r) for dgp, j in outers() for r in range(N_REF)]


def s3_worlds() -> list:
    return ([("world", "true", d) for d in S.DGPS] + [("world", "s2", d, j) for d, j in outers()]
            + [("world", "dbg", sd) for sd in UD.OUTER_SDS])


def s3_tasks() -> list:
    return (s3_worlds() + [("h0", "s2", d, j, r) for d, j in outers() for r in range(N_H0)]
            + [("h0", "true", "batter", r) for r in range(N_H0)])


def variants(names=VARIANTS) -> list:
    return [(name, dgp) for dgp in S.DGPS for name in names]


def s4_tasks(names=VARIANTS) -> list:
    return [("vref", name, dgp, r) for name, dgp in variants(names) for r in range(N_VREF)]


def _s4_names() -> tuple:
    return VARIANTS + (VARIANTS_B if _done(s4_tasks(VARIANTS_B)) else ())


def s5_tasks() -> list:
    return [("assign", r) for r in range(N_ASSIGN)]


def _load_task(*task):
    with open(OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl"), "rb") as f:
        return pickle.load(f)


def _done(task_list) -> bool:
    return all((OUT / "tasks" / ("_".join(str(x) for x in t) + ".pkl")).exists() for t in task_list)


# --- Stage C: summary ---

def mc_p(t_obs: float, ref: np.ndarray) -> tuple[float, float]:
    """Monte Carlo p-values (1 + k) / (N + 1): upper-side test rejects for small T, lower-side for large T."""
    n = len(ref)
    return (1 + int((ref <= t_obs).sum())) / (n + 1), (1 + int((ref >= t_obs).sum())) / (n + 1)


def _control(dgp: str) -> pd.DataFrame:
    return pd.DataFrame([UD._load_task("check", dgp, r) for r in range(N_CONTROL)])


def _refs(dgp: str, j: int) -> pd.DataFrame:
    return pd.DataFrame([_load_task("ref", dgp, j, r) for r in range(N_REF)])


def summarize_s2() -> list:
    lines = ["# Refit-world diagnostic (simulation only)", "",
             "Generated by `python3.12 -m gamecalling.refit_diagnostic`. Design: "
             "`research_notes/uncertainty_design/04_refit_world_diagnostic.md` revision 2. Season 2024, "
             f"**fixed payoffs**, scientific SD exactly {TEST_SD} throughout. No real catcher results are used. "
             "**Diagnostic size: this detects only gross miscalibration of the reference and gives no size or "
             "rate estimate.**", "",
             "# §2: the proposed test run directly", "",
             "Each outer dataset comes from a true world; its world is refit from it (pilot model class), and its "
             f"observed T is compared with {N_REF} reference datasets from that refit world (p = (1 + k) / "
             f"{N_REF + 1}). Control: the same T against the true world's own {N_CONTROL} references (the "
             f"debugging `check` tasks, independent seed; p = (1 + k) / {N_CONTROL + 1}). The upper-side test "
             "rejects for small T, the lower-side test for large T.", ""]
    control = {dgp: _control(dgp) for dgp in S.DGPS}
    rows, ctl_rows = [], []
    for dgp, j in outers():
        o = _outer(dgp, j)
        refs = _refs(dgp, j)
        for s in ("D", "A"):
            t_obs = o["observed_fit"][f"{s}_sd"]
            ref = refs[f"{s}_sd"].to_numpy()
            p_up, p_lo = mc_p(t_obs, ref)
            rows.append({"world": dgp, "outer": j, "score": s, "T_obs": t_obs, "ref_q05": np.quantile(ref, 0.05),
                         "ref_median": np.median(ref), "ref_q95": np.quantile(ref, 0.95),
                         "T_obs_over_ref_mean": t_obs / ref.mean(), "k_below": int((ref <= t_obs).sum()),
                         "p_upper": p_up, "p_lower": p_lo})
            ctl = control[dgp][f"{s}_sd"].to_numpy()
            c_up, c_lo = mc_p(t_obs, ctl)
            ctl_rows.append({"world": dgp, "outer": j, "score": s, "T_obs": t_obs, "ctl_median": np.median(ctl),
                             "T_obs_over_ctl_mean": t_obs / ctl.mean(), "p_upper": c_up, "p_lower": c_lo})
    R, C = pd.DataFrame(rows), pd.DataFrame(ctl_rows)
    lines += ["## Each observed T against its own refit world's references", "", _md(R, "{:.5f}"), "",
              "## Control: the same T against the true world's references", "", _md(C, "{:.5f}"), ""]

    lines += ["## Pre-stated reading (D, primary)", ""]
    n_out = len(outers())
    for side in ("upper", "lower"):
        hits = int((R[(R["score"] == "D")][f"p_{side}"] <= P_CUT).sum())
        verdict = "mismatch reproduced on this side" if hits >= N_TRIGGER and n_out == 6 else "not triggered"
        p_hits = stats.binom.sf(hits - 1, n_out, P_CUT)
        lines.append(f"- **{side.capitalize()} side:** {hits} of {n_out} outer datasets with p <= {P_CUT} "
                     f"(rule: at least {N_TRIGGER} of 6, probability about 0.127% under independent calibrated tests "
                     f"for one pre-specified direction; at least {hits} of {n_out} would have probability "
                     f"{p_hits:.4%}): **{verdict}**.")
    lines += ["- Either side triggering doubles the family probability (about 0.25%). Not triggering does not show "
              "that low-sigma refitting caused the debugging mismatch; it only means no gross failure was detected "
              "with six datasets. A is reported alongside, never replacing D.", ""]

    sc = []
    for dgp, j in outers():
        o = _outer(dgp, j)
        refs = _refs(dgp, j)
        sc.append({"world": dgp, "outer": j, "kappa_per_sd_true": o["scale_true"] / TEST_SD,
                   "kappa_per_sd_refit_median": float(np.median(refs["scale"])) / TEST_SD,
                   "T_obs_D": o["observed_fit"]["D_sd"], "ref_mean_D": refs["D_sd"].mean(),
                   "T_obs_A": o["observed_fit"]["A_sd"], "ref_mean_A": refs["A_sd"].mean()})
    lines += ["## Calibration scale (context, not part of the rule)", "",
              "kappa per unit of scientific SD: the true world's scale for the outer dataset's z, and the median "
              "refit-world scale over its references' z (both divided by the SD). Different z, so this is a "
              "rough comparison.", "", _md(pd.DataFrame(sc), "{:.4f}"), ""]

    conv, secs = [], []
    for t in tasks():
        d = _load_task(*t)
        conv.append(bool(d["D_converged"] and d["A_converged"]))
        secs.append(d["seconds"])
    o_conv = [all(_outer(*k)["observed_fit"][f"{s}_converged"] for s in ("D", "A")) for k in outers()]
    lines += ["## Run", "", f"{len(outers())} outer datasets (converged: {sum(o_conv)} of {len(o_conv)}); "
              f"{len(secs)} reference tasks, median {np.median(secs):.0f} s each; ranking fits converged in "
              f"{np.mean(conv):.1%} of reference tasks.", ""]
    return lines


def _world_label(task: tuple) -> str:
    return {"true": lambda k: f"true {k[0]}", "s2": lambda k: f"refit {k[0]} {k[1]} (§2, SD 0.05)",
            "dbg": lambda k: f"refit pilot debug (SD {k[0]})"}[task[1]](task[2:])


def _alpha_cols(a: dict) -> dict:
    return {"a_out_league": a["outcome"][0], "a_out_cell": a["outcome"][1], "a_con_league": a["contact"][0],
            "a_con_cell": a["contact"][1], "a_policy": a["policy"]}


def _h0_curve(src: str, key: tuple):
    """Main kappa grid and the mean h0 curve for D (centered at kappa = 0), with the number of samples."""
    if src == "true" and key[0] == "pilot":
        cs = [UD._load_task("h0true", r)["curve"] for r in range(UD.R_H)]
    elif src == "true":
        cs = [_load_task("h0", "true", key[0], r)["curve"] for r in range(N_H0)]
    elif src == "dbg":
        cs = [UD._load_task("h0", key[0], r)["curve"] for r in range(UD.R_H)]
    else:
        cs = [_load_task("h0", "s2", *key, r)["curve"] for r in range(N_H0)]
    n_grid = (len(cs[0]) + 1) // 2
    kap = cs[0]["kappa"].to_numpy()[:n_grid]
    h = np.mean([c["D"].to_numpy()[:n_grid] for c in cs], axis=0)
    return kap, h - np.interp(0.0, kap, h), len(cs)


def _world_inputs(task: tuple):
    """(common, catchers, qualifying) of a §3 world."""
    src, key = task[1], task[2:]
    if src == "true":
        _, _, common, opps = S._load(key[0])
        return (common, *UD._catchers(opps, S._CACHE["pitches"]))
    o = _outer(*key) if src == "s2" else UD._outer(key[0])
    return o["common_hat"], o["catchers_hat"], o["qualifying_hat"]


def _world_T(task: tuple) -> dict:
    """Mean reference T at 0.05 in this world: controls (true), §2 references, or the debugging design fit."""
    src, key = task[1], task[2:]
    if src == "true":
        c = _control(key[0])
        return {"T_D": c["D_sd"].mean(), "T_A": c["A_sd"].mean(), "T_source": f"{N_CONTROL} controls"}
    if src == "s2":
        r = _refs(*key)
        return {"T_D": r["D_sd"].mean(), "T_A": r["A_sd"].mean(), "T_source": f"{N_REF} references"}
    d = pd.DataFrame([{k: UD._load_task("design", key[0], m)[k] for k in ("sigma", "D_sd", "A_sd")}
                      for m in range(UD.M_DESIGN)])
    fit = {s: np.polyval(np.polyfit(d["sigma"], d[f"{s}_sd"], 1), TEST_SD) for s in ("D", "A")}
    return {"T_D": fit["D"], "T_A": fit["A"], "T_source": f"linear fit, {UD.M_DESIGN} design datasets"}


def summarize_s3() -> list:
    lines = ["# §3: world-by-world comparison", "",
             "Each refit world against its true world (§2 refit worlds at SD 0.05; the debugging refit worlds at "
             "SD 0 and 0.01 against the true pilot world). Cells are matched on (pitcher, hand, pitch type[, count]) "
             "and weighted by real 2024 scorable pitches. Batter-world outcome probabilities are mixed over the "
             "hidden batter types by their real pitch shares. `rebuild_max_diff`: the world rebuilt from its re-drawn "
             "simulated pitches against the stored one (0 means the same world).", ""]
    rows = []
    for t in s3_worlds():
        d = _load_task(*t)
        rows.append({"world": _world_label(t), **_alpha_cols(d["alphas"]), "rebuild_max_diff": d["rebuild_max_diff"]})
    lines += ["## Pooling concentrations (alpha) by level", "",
              "Outcome and contact: league level (pulled toward base) and pitcher cell (pulled toward league). "
              "Policy: one concentration for pi0. Smaller alpha means less pooling.", "",
              _md(pd.DataFrame(rows), "{:.4g}"), ""]
    rows = []
    cols = ["outcome_spread", "contact_spread", "outcome_rms_diff", "contact_rms_diff", "pi0_tv",
            "unmatched_pitches_a", "unmatched_pitches_b", "qv_sd", "best_gap", "g(-100)", "g(100)", "kappa_per_sd"]
    for t in s3_worlds():
        d = _load_task(*t)
        rows.append({"world": _world_label(t), **{c: d.get(c, np.nan) for c in cols}, **_world_T(t)})
    lines += ["## Cells, policy and action values", "",
              "`*_spread`: RMS distance of pitcher cells from the league level (heterogeneity; a flatter world is "
              "smaller). `*_rms_diff`: RMS difference from the true world's matched cells. `pi0_tv`: total variation "
              "from the true pi0. `unmatched_pitches_a` / `_b`: real pitches on slots of this world missing from the "
              "true world / of the true world missing from this one. `qv_sd`: pi0-weighted SD of Q - V on S1 "
              "(x 100 runs); `best_gap`: mean V - min Q. `g(+-100)`: scientific value at kappa = +-100. "
              "`kappa_per_sd`: calibrated scale at 0.05 with one common z, per unit SD. `T_D`, `T_A`: mean "
              "reference T at 0.05 in that world (context).", "",
              _md(pd.DataFrame(rows), "{:.4f}"), ""]

    rows = []
    for t in s3_worlds():
        common, catchers, qual = _world_inputs(t)
        kap, h, n = _h0_curve(t[1], t[2:])
        order = np.argsort(common.g)
        hs = lambda s: float(np.interp(np.interp(s, common.g[order], common.kappa_grid[order]), kap, h))  # noqa: E731
        z = _common_z(catchers)
        scale = S.calibrate(common, z, qual, TEST_SD)
        sig = float(np.std(np.interp(scale * z[qual], kap, h), ddof=1))
        rows.append({"world": _world_label(t), "h0_samples": n, "h0_at_s-0.10": hs(-0.10), "h0_at_s-0.05": hs(-0.05),
                     "h0_at_s+0.05": hs(0.05), "h0_at_s+0.10": hs(0.10), "slope_pm05": (hs(0.05) - hs(-0.05)) / 0.10,
                     "target_sd_at_0.05": sig, **_world_T(t)})
    C = pd.DataFrame(rows)
    for dgp in S.DGPS:
        ref = C.loc[C["world"] == f"true {dgp}", "target_sd_at_0.05"].iloc[0]
        sel = C["world"].str.contains(f" {dgp}") | ((dgp == "pilot") & C["world"].str.contains("debug"))
        C.loc[sel, "target_sd_vs_true"] = C.loc[sel, "target_sd_at_0.05"] / ref
    lines += ["## Compression curve h0(g^-1(s))", "",
              "The h0 target (D; D and A coincide after centering on common states) at the kappa giving scientific "
              "value s, centered at kappa = 0, from the mean of the world's h0 samples. `slope_pm05`: change in h0 per "
              "unit of scientific value between s = -0.05 and +0.05. `target_sd_at_0.05`: SD over qualifying catchers "
              "of h0 at their kappa, with the common z calibrated to scientific SD 0.05 (the noise-free signal on the "
              "estimator's scale); `target_sd_vs_true`: its ratio to the true world's. **One part of the explanation, "
              "read alongside T:** T also depends on ranking shrinkage, score noise, assignment and eligibility.", "",
              _md(C, "{:.4f}"), ""]
    return lines


def summarize_s4() -> list:
    lines = ["# §4: ablations for the second-shrinkage explanation", "",
             f"Base: outer dataset {S4_J} of each true world (§2). Each variant refits the world differently "
             f"(the twin and more-data variants also change the training data), then draws {N_VREF} references at "
             "0.05. `gap_left_D`: (variant mean T - true mean T) / (base mean T - true mean T); 1 = no movement, "
             "0 = the true world's T (`gap_left_A` the same for A). `alpha_fixed` fixes all five concentrations; "
             "`alpha_cell_fixed` (§4b, spec 05) fixes only the outcome and contact pitcher-cell alpha. Movement "
             "toward the true world supports a mechanism; it does not supply a "
             f"correction. The observed T and p are descriptive ({N_VREF} references; p = (1 + k) / {N_VREF + 1}). "
             "`sim_pitches`: simulated pitches the variant world was refit from.", ""]
    rows = []
    for dgp in S.DGPS:
        t_obs = _outer(dgp, S4_J)["observed_fit"]
        true_w = _load_task("world", "true", dgp)
        base_w = _load_task("world", "s2", dgp, S4_J)
        ctl, base = _control(dgp), _refs(dgp, S4_J)
        entries = [("true", true_w, true_w["alphas"], ctl, np.nan), ("base", base_w, base_w["alphas"], base, np.nan)]
        for name in _s4_names():
            v = _variant(name, dgp)
            refs = pd.DataFrame([_load_task("vref", name, dgp, r) for r in range(N_VREF)])
            entries.append((name, v["metrics"], v["alphas"], refs, v["n_sim_pitches"]))
        for name, m, a, refs, n_sim in entries:
            row = {"world": dgp, "variant": name, **_alpha_cols(a)}
            row.update({k: m.get(k, np.nan) for k in ("outcome_spread", "contact_spread", "outcome_rms_diff",
                                                      "pi0_tv", "unmatched_pitches_a", "qv_sd", "kappa_per_sd")})
            row.update(n_refs=len(refs), T_D=refs["D_sd"].mean(), T_A=refs["A_sd"].mean(), sim_pitches=n_sim)
            if name not in ("true", "base"):
                row["p_upper_D"] = mc_p(t_obs["D_sd"], refs["D_sd"].to_numpy())[0]
            rows.append(row)
        R = pd.DataFrame(rows)
        sel = R["world"] == dgp
        for s in ("D", "A"):
            t_true = R.loc[sel & (R["variant"] == "true"), f"T_{s}"].iloc[0]
            t_base = R.loc[sel & (R["variant"] == "base"), f"T_{s}"].iloc[0]
            R.loc[sel, f"gap_left_{s}"] = (R.loc[sel, f"T_{s}"] - t_true) / (t_base - t_true)
        rows = R.to_dict("records")
    R = pd.DataFrame(rows)
    lines += [_md(R, "{:.4g}"), "",
              f"Observed T (outer {S4_J}, D): " + ", ".join(
                  f"{d} {_outer(d, S4_J)['observed_fit']['D_sd']:.4f}" for d in S.DGPS) + ".",
              "", "Variant runs: " + ", ".join(f"{n} {d} {_variant(n, d)['seconds']:.0f} s" for n, d in variants(_s4_names()))
              + ".", ""]
    return lines


def _offsets(results: list, arm: str, score: str, catchers: np.ndarray, qual: np.ndarray) -> pd.DataFrame:
    """Per qualifying catcher: mean (offset) and SE across datasets of the effect centered on the qualifying mean."""
    E = []
    for res in results:
        e = res["arms"][arm]["effects"].set_index("catcher")[f"{score}_effect"].reindex(catchers)
        e = e[qual]
        E.append(e - e.mean())
    E = pd.concat(E, axis=1)
    return pd.DataFrame({"offset": E.mean(1), "se": E.std(1, ddof=1) / np.sqrt(E.shape[1])})


def summarize_s5() -> list:
    _, _, _, opps = S._load("pilot")
    catchers, qual = UD._catchers(opps, S._CACHE["pitches"])
    results = [_load_task(*t) for t in s5_tasks()]
    n = len(results)
    lines = ["# §5: D's per-catcher offsets under label arms", "",
             f"True pilot world, every catcher at pi0 (all targets 0), {n} datasets simulated and scored once; each "
             "arm only relabels catchers before the ranking fits (D, A and B = D - A). Offsets: the mean over "
             "datasets of each qualifying catcher's effect centered on the qualifying mean (real qualifying flags, "
             f"fixed). SE: SD across datasets / sqrt({n}). Shuffles are drawn once and fixed across datasets. "
             "B is a diagnostic, not an additive decomposition of D's fitted effects.", ""]
    off = {(arm, s): _offsets(results, arm, s, catchers, qual) for arm in ARMS for s in S5_SCORES}
    rows = []
    for arm in ARMS:
        for s in S5_SCORES:
            o = off[arm, s].dropna()
            z = o["offset"] / o["se"]
            real = off["real", s].reindex(o.index)
            rows.append({"arm": arm, "score": s, "catchers": len(o), "offset_sd": o["offset"].std(ddof=1),
                         "mean_se": o["se"].mean(), "sd_over_se": o["offset"].std(ddof=1) / o["se"].mean(),
                         "share_abs_z_gt_2": float((z.abs() > 2).mean()),
                         "corr_with_real": float(np.corrcoef(o["offset"], real["offset"])[0, 1]),
                         "mean_T": np.mean([r["arms"][arm]["fit"][f"{s}_sd"] for r in results])})
    lines += ["## Offsets by arm", "", _md(pd.DataFrame(rows), "{:.4f}"), ""]

    rows = []
    for arm in ARMS:
        d, a, b = (off[arm, s]["offset"] for s in S5_SCORES)
        ok = d.notna() & a.notna() & b.notna()
        d, a, b = d[ok], a[ok], b[ok]
        rows.append({"arm": arm, "corr_D_B": float(np.corrcoef(d, b)[0, 1]), "slope_D_on_B": float(np.polyfit(b, d, 1)[0]),
                     "corr_DminusA_B": float(np.corrcoef(d - a, b)[0, 1]),
                     "slope_DminusA_on_B": float(np.polyfit(b, d - a, 1)[0]), "corr_D_A": float(np.corrcoef(d, a)[0, 1])})
    lines += ["## D against B (diagnostic)", "",
              "`DminusA`: the fitted D offset minus the fitted A offset (separately shrunk), against the fitted B "
              "offset.", "", _md(pd.DataFrame(rows), "{:.3f}"), ""]

    rows = []
    for arm in ARMS:
        cov = pd.concat([r["arms"][arm]["cov"] for r in results]).groupby(level=0).mean()
        o = off[arm, "D"].dropna()
        c = cov.reindex(o.index)
        rows.append({"arm": arm, **{f"corr_D_{k}": float(np.corrcoef(o["offset"], c[k])[0, 1])
                                    for k in ("staff_v00", "n_eligible", "top_pitcher_share", "n")}})
    lines += ["## Associations of D offsets (qualifying catchers)", "",
              "`staff_v00`: pitch-weighted mean of the catcher's pitchers' true V at 0-0, bases empty, nobody out. "
              "`n_eligible`: mean eligible actions per pitch. `top_pitcher_share`: share of the catcher's pitches from "
              "the top pitcher. `n`: pitches (mean over datasets).", "", _md(pd.DataFrame(rows), "{:.3f}"), ""]

    rows = []
    base_n = pd.concat([r["arms"]["real"]["cov"]["n"] for r in results], axis=1).mean(1)
    qset = catchers[qual]
    for arm in ARMS:
        nn = pd.concat([r["arms"][arm]["cov"]["n"] for r in results], axis=1).mean(1).reindex(qset)
        rel = (nn / base_n.reindex(qset) - 1).abs()
        rows.append({"arm": arm, "median_abs_rel_change_n": float(rel.median()), "max_abs_rel_change_n": float(rel.max()),
                     "qualifying_missing": int(nn.isna().sum()),
                     "groups": float(np.mean([r["arms"][arm]["groups"] for r in results])),
                     "main_group_share": float(np.mean([r["arms"][arm]["main_share"] for r in results])),
                     "converged": float(np.mean([all(r["arms"][arm]["fit"][f"{s}_converged"] for s in S5_SCORES)
                                                 for r in results]))})
    lines += ["## Pitch counts and connectivity", "",
              f"Share of pitches caught by a secondary catcher of their appearance (relabeled in every non-real arm): "
              f"{np.mean([r['secondary_share'] for r in results]):.4f}. Pitch-count change: qualifying catchers' mean "
              "pitches against the real arm. Groups: connected groups of qualifying catchers (pairs with >= 100 "
              "pitches through a shared pitcher), mean over datasets.", "", _md(pd.DataFrame(rows), "{:.4f}"), ""]

    lines += _offset_alignment(off["real", "D"], catchers, qual)
    return lines


def _offset_alignment(d_off: pd.DataFrame, catchers, qual) -> list:
    """Control audit follow-up: does T at 0.05 rise when the drawn effects line up with the real-arm D offsets?"""
    _, _, common, opps = S._load("pilot")
    rows = []
    for kind, idx in (("control", range(N_CONTROL)), ("outer", range(J))):
        for r in idx:
            seed = [UD.SEED, 6, 0, r] if kind == "control" else [SEED, 1, 0, r]
            z = np.random.default_rng(seed).normal(size=len(catchers))
            g = pd.Series(S.scientific(common, S.calibrate(common, z, qual, TEST_SD) * z), index=catchers)[qual]
            o = d_off["offset"].reindex(g.index)
            T = UD._load_task("check", "pilot", r)["D_sd"] if kind == "control" else _outer("pilot", r)["observed_fit"]["D_sd"]
            rows.append({"kind": kind, "r": r, "T_D": T, "corr_g_offset": float(np.corrcoef(g, o)[0, 1])})
    A = pd.DataFrame(rows)
    rho = stats.spearmanr(A["T_D"], A["corr_g_offset"])
    return ["## Control audit follow-up: alignment of drawn effects with the D offsets (pilot world)", "",
            "Per dataset at 0.05: the correlation over qualifying catchers between the drawn scientific value g(kappa) "
            "(a cost: lower is better) and the catcher's real-arm D offset. A positive value means the drawn effects "
            "line up with the offsets, which would add to T. Exploratory, chosen after seeing the two high outer "
            f"datasets. Spearman correlation with T_D over all {len(A)} datasets: {rho.statistic:.2f} "
            f"(p = {rho.pvalue:.3f}).", "", _md(A, "{:.4f}"), ""]


def summarize() -> str:
    lines = summarize_s2()
    for done, part in ((_done(s3_tasks()), summarize_s3), (_done(s4_tasks()) and _done(s3_tasks()), summarize_s4),
                       (_done(s5_tasks()), summarize_s5)):
        if done:
            lines += part()
    return "\n".join(lines)


def _pool(fn, items, workers, label, t0):
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        for i, _ in enumerate(ex.map(fn, items), 1):
            if i % 10 == 0 or i == len(items):
                print(f"{label}: {i}/{len(items)} done ({time.perf_counter() - t0:.0f} s)", flush=True)


def main(stage: str = "s2", workers: int = 8):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    OUT.mkdir(parents=True, exist_ok=True)
    man = _manifest_path(stage)
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest(stage))):
        raise ValueError(f"{man} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(stage), indent=1))
    t0 = time.perf_counter()
    if stage == "s2":
        _pool(prepare_outer, outers(), workers, "outer datasets", t0)
        _pool(run_task, tasks(), workers, "references", t0)
    else:
        if not all(_outer_path(*k).exists() for k in outers()) or not _done(tasks()):
            raise RuntimeError("§2 must finish first")
        if stage == "s3":
            _pool(run_task, s3_tasks(), workers, "§3 tasks", t0)
        elif stage in ("s4", "s4b"):
            if not _done(s3_worlds()) or (stage == "s4b" and not _done(s4_tasks())):
                raise RuntimeError("§3 world tasks (and, for s4b, §4) must finish first")
            names = VARIANTS if stage == "s4" else VARIANTS_B
            _pool(prepare_variant, variants(names), workers, "variant worlds", t0)
            _pool(run_task, s4_tasks(names), workers, "variant references", t0)
        else:
            prepare_arms()
            _pool(run_task, s5_tasks(), workers, "§5 datasets", t0)
    REPORT.write_text(summarize())
    print(f"refit diagnostic {stage} done in {time.perf_counter() - t0:.0f} s; report in {REPORT}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "s2", int(args[1]) if len(args) > 1 else 8)
