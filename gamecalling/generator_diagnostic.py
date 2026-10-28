"""Paired diagnostic of a revised reference-world generator
(research_notes/uncertainty_design/06_generator_diagnostic.md, revision 2).

Simulation only, fixed payoffs (the real PA payoff estimates), season 2024. Nothing here touches real
catcher results, and nothing here adopts a generator.

The procedure's own sequence, one generation: true world -> outer dataset at scientific SD exactly 0.05 ->
observed T (REML catcher SD of D, and of A) -> the production fit of the reference world from the outer data
-> per reference, one generated season -> T*. Arms differ only in that season's outcome and contact cells:
- A: fixed posterior means (n + alpha m) / (N + alpha), the same world for every reference (current);
- C: posterior draws Dirichlet(n + alpha m), drawn once per reference season (leading candidate);
- B: population draws Dirichlet(alpha m), drawn once per reference season (comparison).
League means, every alpha and pi0 stay at the outer fit's values (conditional on fitted hyperparameters).

True worlds: v2 pilot (fixed probabilities), v2 batter (hidden batter types), and two hierarchical worlds
whose cells are drawn once from Dirichlet(alpha m) around the real fit's league means, at the real cell
alpha (hier) and at 4x it (hier_stress). The pilot and batter outers are spec 04's outers 0..J-1, re-drawn
exactly, with their arm-A references and controls reused.

Stages: s1 hier true worlds, outers (and reference fits), hier controls; s2 arm A in the hier worlds and
arm C everywhere; s3 arm B in B_WORLDS.
Run with: python3.12 -m gamecalling.generator_diagnostic [STAGE] [WORKERS]
Outputs: data/generator_diagnostic/v1/ and reports/generator_diagnostic.md.
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

from gamecalling import arsenal, config, models, refit_diagnostic as R, simulation as S, solver, \
    uncertainty_debug as UD
from gamecalling.phase1 import _md

VERSION = "v1"
OUT = config.DATA_DIR / "generator_diagnostic" / VERSION
REPORT = config.REPORTS_DIR / "generator_diagnostic.md"
SEED = 20261102
TEST_SD = 0.05
WORLDS = ("pilot", "batter", "hier", "hier_stress")
HIER_MULT = {"hier": 1.0, "hier_stress": 4.0}
ARMS = ("A", "C", "B")
B_WORLDS = ("pilot", "hier_stress")
J = 3            # outer datasets per world
N_REF = 19       # references per outer, arms A and C
N_B = 9          # references per outer, arm B
N_CONTROL = 20   # true-world references at 0.05 per hier world (pilot and batter reuse debugging's)
P_CUT = 0.10     # tail-count rule: p <= 0.10 ...
N_TRIGGER = 4    # ... in at least 4 of the 12 outers on one side (global)
SCALE_RANGE = (0.85, 1.15)  # scale rule, per world
STAGES = ("s1", "s2", "s3")

SMOKE_DIR = os.environ.get("GENERATOR_DIAGNOSTIC_SMOKE")
if SMOKE_DIR:
    OUT, REPORT = Path(SMOKE_DIR), Path(SMOKE_DIR) / "report.md"
    J, N_REF, N_B, N_CONTROL = 1, 2, 1, 2

# Streams under SEED: 0 hier world cells, 1 hier outer, 2 hier control, 3 reference (world, outer, arm, r).


def manifest(stage: str) -> dict:
    return {"version": VERSION, "simulation_version": S.VERSION, "seed": SEED, "test_sd": TEST_SD,
            "worlds": list(WORLDS), "hier_mult": HIER_MULT, "j": J, "n_ref": N_REF, "n_b": N_B,
            "b_worlds": list(B_WORLDS), "n_control": N_CONTROL, "stage": stage,
            "reused": f"refit_diagnostic {R.VERSION} outers and arm-A references; uncertainty_debug {UD.VERSION} checks",
            "payoffs": "fixed (real PA payoff estimates)",
            "design": "research_notes/uncertainty_design/06_generator_diagnostic.md revision 2"}


# --- Cell draws (tested in tests/test_generator_diagnostic.py) ---

def cell_inputs(om: models.PooledModel, cm: models.PooledModel, units: models.Units) -> dict:
    """Counts n, league means m and cell alpha for every outcome cell (unit slot x count) and contact cell."""
    q = units.slots.merge(pd.DataFrame({"cidx": np.arange(solver.N_COUNTS)}), how="cross")
    s = units.slots
    return {"q": q[["u", "cidx", "slot"]], "n_o": models._lookup(om.cell_n, q, om.levels[2]),
            "m_o": om.league_mean(q), "a_o": om.alpha_cell,
            "s": s[["u", "slot"]], "n_c": models._lookup(cm.cell_n, s, cm.levels[2]),
            "m_c": cm.league_mean(s), "a_c": cm.alpha_cell}


def draw_cells(rng, n: np.ndarray, m: np.ndarray, alpha: float, arm: str) -> np.ndarray:
    """One latent probability vector per row: C Dirichlet(n + alpha m), B Dirichlet(alpha m), A the mean."""
    if arm == "A":
        return (n + alpha * m) / (n.sum(-1, keepdims=True) + alpha)
    g = rng.gamma(alpha * m + (n if arm == "C" else 0.0))
    return g / g.sum(-1, keepdims=True)


def reference_cells(inp: dict, arm: str, rng, mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One season's outcome and contact cells. `mult` scales alpha (hier true worlds only)."""
    return (draw_cells(rng, inp["n_o"], inp["m_o"], mult * inp["a_o"], arm),
            draw_cells(rng, inp["n_c"], inp["m_c"], mult * inp["a_c"], arm))


def unit_arrays(units: models.Units, inp: dict, p_o: np.ndarray, p_c: np.ndarray):
    """(U, 12, A, 6) and (U, A, 8) arrays as models.unit_outcome_probs / unit_contact_probs lay them out."""
    U, A = units.eligible.shape
    P1 = np.zeros((U, solver.N_COUNTS, A, models.N_OUTCOMES))
    P1[..., 0] = 1.0
    q, s = inp["q"], inp["s"]
    P1[q["u"], q["cidx"], q["slot"]] = p_o
    PC = np.full((U, A, models.N_BINS), 1.0 / models.N_BINS)
    PC[s["u"], s["slot"]] = p_c
    return P1, PC


# --- Stage 1: true worlds, outers, reference fits ---

def _world_path(name: str) -> Path:
    return OUT / f"world_{name}.pkl"


def _outer_path(name: str, j: int) -> Path:
    return OUT / f"outer_{name}_{j}.pkl"


_MEMO = {}


def _truth(name: str):
    """(world, common, opps) of a true world."""
    if name in S.DGPS:
        return S._load(name)[1:]
    if name not in _MEMO:
        with open(_world_path(name), "rb") as f:
            _MEMO[name] = pickle.load(f)
    t = _MEMO[name]
    return t["world"], t["common"], t["opps"]


def prepare_world(name: str) -> str:
    path = _world_path(name)
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    pas = S._load("pilot")[0]
    pitches = S._CACHE["pitches"]
    support = pd.read_parquet(config.DATA_DIR / "arsenal_support.parquet")
    units, om, cm, pol = S.fit_world_models(pitches, support)
    inp = cell_inputs(om, cm, units)
    p_o, p_c = reference_cells(inp, "B", np.random.default_rng([SEED, 0, WORLDS.index(name)]), HIER_MULT[name])
    P1, PC = unit_arrays(units, inp, p_o, p_c)
    mult = HIER_MULT[name]
    alphas = {"outcome": (om.alpha_league, mult * om.alpha_cell), "contact": (cm.alpha_league, mult * cm.alpha_cell),
              "policy": pol.alpha}
    batters = np.sort(pitches.loc[pitches["season"] == S.SEASON, "batter"].unique())
    world = S.solve_world(units, P1, PC, pol.probs, pas, batters, "pilot", alphas=alphas)
    opps = S.opportunities(pitches, pas, units)
    common = S.build_common(world, opps, S.real_states(pitches, world))
    catchers, qual = UD._catchers(opps, pitches)
    out = {"world": world, "common": common, "opps": opps,
           "metrics": R._world_metrics(world, common, catchers, qual), "seconds": time.perf_counter() - t0}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"world {name}: prepared in {out['seconds']:.0f} s", flush=True)
    return str(path)


def _fit_reference(sim: pd.DataFrame, pas: pd.DataFrame) -> dict:
    """The production fit of the reference world from simulated pitches, kept with its cell inputs."""
    pitches = S._CACHE["pitches"]
    units, om, cm, pol = S.fit_world_models(sim, arsenal.support_table(sim))
    alphas = {"outcome": (om.alpha_league, om.alpha_cell), "contact": (cm.alpha_league, cm.alpha_cell),
              "policy": pol.alpha}
    batters = np.sort(sim.loc[sim["season"] == S.SEASON, "batter"].unique())
    P1, PC = models.unit_outcome_probs(om, units), models.unit_contact_probs(cm, units)
    w_hat = S.solve_world(units, P1, PC, pol.probs, pas, batters, "pilot", alphas=alphas)
    opps_hat = S.opportunities(pitches, pas, units)
    common_hat = S.build_common(w_hat, opps_hat, S.real_states(pitches, w_hat))
    cat_hat, qual_hat = UD._catchers(opps_hat, pitches)
    return {"units": units, "inputs": cell_inputs(om, cm, units), "pi0": pol.probs, "batters": batters,
            "alphas": alphas, "w_hat": w_hat, "common_hat": common_hat, "opps_hat": opps_hat,
            "catchers_hat": cat_hat, "qualifying_hat": qual_hat}


def prepare_outer(args: tuple) -> str:
    name, j = args
    path = _outer_path(name, j)
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    pas = S._load("pilot")[0]
    pitches = S._CACHE["pitches"]
    if name in S.DGPS:
        # spec 04's outer j, re-drawn exactly; its observed T is reused
        sim = R._draw_outer(name, j)[4]
        o4 = R._outer(name, j)
        out = {"observed": o4["observed_fit"], "scale_true": o4["scale_true"], **_fit_reference(sim, pas)}
        out["rebuild_max_diff"] = max(float(np.abs(out["w_hat"].P - o4["w_hat"].P).max()),
                                      float(np.abs(out["w_hat"].PC - o4["w_hat"].PC).max()),
                                      float(np.abs(out["w_hat"].pi0 - o4["w_hat"].pi0).max())) \
            if out["w_hat"].units.slots.equals(o4["w_hat"].units.slots) else np.inf
    else:
        world, common, opps = _truth(name)
        catchers, qual = UD._catchers(opps, pitches)
        rng = np.random.default_rng([SEED, 1, WORLDS.index(name), j])
        z = rng.normal(size=len(catchers))
        scale = S.calibrate(common, z, qual, TEST_SD)
        code = pd.Series(np.arange(len(catchers)), index=catchers)
        sim = S.simulate(world, opps, (scale * z)[opps["start_catcher"].map(code).to_numpy()], rng)
        _, fits = S.estimate(sim, pas)
        out = {"observed": UD._fit_record(fits), "scale_true": scale, **_fit_reference(sim, pas)}
    out.update(name=name, j=j, metrics_A=R._world_metrics(out["w_hat"], out["common_hat"], out["catchers_hat"],
                                                          out["qualifying_hat"]),
               seconds=time.perf_counter() - t0)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"outer {name} {j}: prepared in {out['seconds']:.0f} s", flush=True)
    return str(path)


def _outer(name: str, j: int) -> dict:
    if ("outer", name, j) not in _MEMO:
        for k in [k for k in _MEMO if k[0] == "outer"]:  # keep one outer per process (memory)
            del _MEMO[k]
        with open(_outer_path(name, j), "rb") as f:
            _MEMO["outer", name, j] = pickle.load(f)
    return _MEMO["outer", name, j]


# --- Tasks ---

def _task_path(task: tuple) -> Path:
    return OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl")


def _season(world, common, opps, catchers, qual, rng, pas) -> dict:
    """One dataset at exactly TEST_SD (calibrated in `world`, fresh z): T, scale, and the cell alpha refit
    from its own simulated pitches (a mechanism check)."""
    z = rng.normal(size=len(catchers))
    scale = S.calibrate(common, z, qual, TEST_SD)
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    sim = S.simulate(world, opps, (scale * z)[opps["start_catcher"].map(code).to_numpy()], rng)
    _, fits = S.estimate(sim, pas)
    om, cm = models.fit_outcome_model(sim), models.fit_contact_model(sim)
    return {"scale": scale, **UD._fit_record(fits), "refit_a_out_cell": om.alpha_cell,
            "refit_a_con_cell": cm.alpha_cell}


def run_task(task: tuple) -> str:
    path = _task_path(task)
    if path.exists():
        return task[0]
    t0 = time.perf_counter()
    pas = S._load("pilot")[0]
    pitches = S._CACHE["pitches"]
    if task[0] == "control":
        _, name, r = task
        world, common, opps = _truth(name)
        catchers, qual = UD._catchers(opps, pitches)
        res = _season(world, common, opps, catchers, qual,
                      np.random.default_rng([SEED, 2, WORLDS.index(name), r]), pas)
    elif task[0] == "ref":
        _, arm, name, j, r = task
        o = _outer(name, j)
        rng = np.random.default_rng([SEED, 3, WORLDS.index(name), j, ARMS.index(arm), r])
        res = {}
        if arm == "A":
            world, common = o["w_hat"], o["common_hat"]
        else:
            P1, PC = unit_arrays(o["units"], o["inputs"], *reference_cells(o["inputs"], arm, rng))
            world = S.solve_world(o["units"], P1, PC, o["pi0"], pas, o["batters"], "pilot", alphas=o["alphas"])
            common = S.build_common(world, o["opps_hat"], S.real_states(pitches, world))
            res["metrics"] = R._world_metrics(world, common, o["catchers_hat"], o["qualifying_hat"])
        res.update(_season(world, common, o["opps_hat"], o["catchers_hat"], o["qualifying_hat"], rng, pas))
    else:
        raise ValueError(task[0])
    res["seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print(f"{path.stem}: {res['seconds']:.0f} s", flush=True)
    return task[0]


def hier_worlds() -> list:
    return [w for w in WORLDS if w in HIER_MULT]


def outers() -> list:
    return [(w, j) for w in WORLDS for j in range(J)]


def s1_tasks() -> list:
    return [("control", w, r) for w in hier_worlds() for r in range(N_CONTROL)]


def s2_tasks() -> list:
    return ([("ref", "A", w, j, r) for w in hier_worlds() for j in range(J) for r in range(N_REF)]
            + [("ref", "C", w, j, r) for w, j in outers() for r in range(N_REF)])


def s3_tasks() -> list:
    return [("ref", "B", w, j, r) for w in B_WORLDS for j in range(J) for r in range(N_B)]


def _done(task_list) -> bool:
    return all(_task_path(t).exists() for t in task_list)


def _load(task: tuple) -> dict:
    with open(_task_path(task), "rb") as f:
        return pickle.load(f)


# --- Summary ---

def _control(name: str) -> pd.DataFrame:
    if name in S.DGPS:
        return R._control(name)
    return pd.DataFrame([_load(("control", name, r)) for r in range(N_CONTROL)])


def _refs(arm: str, name: str, j: int) -> pd.DataFrame:
    if arm == "A" and name in S.DGPS:
        return R._refs(name, j)
    n = N_B if arm == "B" else N_REF
    return pd.DataFrame([_load(("ref", arm, name, j, r)) for r in range(n)])


def _arm_rows(arm: str, worlds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per outer and per world: tail p-values, per-outer and pooled scale ratios (D and A)."""
    per_outer, per_world = [], []
    for name in worlds:
        ctl = _control(name)
        allrefs = []
        for j in range(J):
            o, refs = _outer(name, j), _refs(arm, name, j)
            allrefs.append(refs)
            row = {"arm": arm, "world": name, "outer": j, "n_refs": len(refs)}
            for s in ("D", "A"):
                t_obs, ref = o["observed"][f"{s}_sd"], refs[f"{s}_sd"].to_numpy()
                pu, pl = R.mc_p(t_obs, ref)
                row.update({f"T_obs_{s}": t_obs, f"ref_mean_{s}": ref.mean(), f"p_upper_{s}": pu, f"p_lower_{s}": pl,
                            f"ratio_{s}": ref.mean() / ctl[f"{s}_sd"].mean()})
            row["converged"] = float(refs[["D_converged", "A_converged"]].all(axis=1).mean())
            row["scale_mean"] = refs["scale"].mean() if "scale" in refs else np.nan
            for k in ("refit_a_out_cell", "refit_a_con_cell"):
                row[k] = refs[k].median() if k in refs else np.nan
            row["outer_a_out_cell"], row["outer_a_con_cell"] = o["alphas"]["outcome"][1], o["alphas"]["contact"][1]
            if "metrics" in refs:
                m = pd.DataFrame(list(refs["metrics"]))
                row.update({f"{k}_mean": m[k].mean() for k in ("outcome_spread", "kappa_per_sd")})
            else:
                row.update(outcome_spread_mean=o["metrics_A"]["outcome_spread"],
                           kappa_per_sd_mean=o["metrics_A"]["kappa_per_sd"])
            per_outer.append(row)
        refs = pd.concat(allrefs, ignore_index=True)
        po = pd.DataFrame(per_outer)
        po = po[po["world"] == name]
        w = {"arm": arm, "world": name, "n_refs": len(refs), "control_T_D": ctl["D_sd"].mean(),
             "control_T_A": ctl["A_sd"].mean()}
        for s in ("D", "A"):
            w[f"pooled_ratio_{s}"] = refs[f"{s}_sd"].mean() / ctl[f"{s}_sd"].mean()
            w[f"upper_count_{s}"] = int((po[f"p_upper_{s}"] <= P_CUT).sum())
            w[f"lower_count_{s}"] = int((po[f"p_lower_{s}"] <= P_CUT).sum())
        per_world.append(w)
    return pd.DataFrame(per_outer), pd.DataFrame(per_world)


def summarize() -> str:
    lines = ["# Paired diagnostic of a revised reference-world generator", "",
             "Spec: `research_notes/uncertainty_design/06_generator_diagnostic.md` (revision 2). Simulation only, "
             "fixed payoffs, season 2024, scientific SD exactly 0.05. Arms: A fixed posterior means (current), "
             "C posterior draws Dirichlet(n + alpha m) once per reference season (leading candidate), B population "
             "draws Dirichlet(alpha m) (comparison). League means, every alpha and pi0 stay at the outer fit's "
             "values: **conditional on fitted hyperparameters.** This is a screen for gross failure, not "
             "validation, and nothing here adopts a generator or supplies a correction.", ""]
    rows = []
    for name in WORLDS:
        ctl = _control(name)
        rows.append({"world": name, "n_control": len(ctl), "control_T_D": ctl["D_sd"].mean(),
                     "control_q05_D": ctl["D_sd"].quantile(0.05), "control_q95_D": ctl["D_sd"].quantile(0.95),
                     "control_T_A": ctl["A_sd"].mean(),
                     "converged": float(ctl[["D_converged", "A_converged"]].all(axis=1).mean())})
    lines += ["## Stage 1: true worlds, controls and outer datasets", "",
              "Controls: true-world datasets at 0.05 (pilot and batter: the debugging `check` tasks, reused).", "",
              _md(pd.DataFrame(rows), "{:.4g}"), ""]
    hw = []
    for name in hier_worlds():
        with open(_world_path(name), "rb") as f:
            t = pickle.load(f)
        hw.append({"world": name, "a_out_cell": t["world"].alphas["outcome"][1],
                   "a_con_cell": t["world"].alphas["contact"][1],
                   **{k: t["metrics"][k] for k in ("outcome_spread", "contact_spread", "qv_sd", "kappa_per_sd")}})
    lines += ["Hierarchical true worlds (cells drawn once around the real fit's league means):", "",
              _md(pd.DataFrame(hw), "{:.4g}"), ""]
    orow = []
    for name, j in outers():
        o = _outer(name, j)
        orow.append({"world": name, "outer": j, "T_obs_D": o["observed"]["D_sd"], "T_obs_A": o["observed"]["A_sd"],
                     "a_out_cell": o["alphas"]["outcome"][1], "a_con_cell": o["alphas"]["contact"][1],
                     "a_policy": o["alphas"]["policy"], "rebuild_max_diff": o.get("rebuild_max_diff", np.nan),
                     "seconds": o["seconds"]})
    lines += ["Outer datasets and their reference fits (pilot and batter: spec 04's outers, re-drawn; "
              "`rebuild_max_diff` against spec 04's stored refit world):", "", _md(pd.DataFrame(orow), "{:.4g}"), ""]
    if not _done(s2_tasks()):
        return "\n".join(lines)

    a_o, a_w = _arm_rows("A", WORLDS)
    c_o, c_w = _arm_rows("C", WORLDS)
    arms_o, arms_w = [a_o, c_o], [a_w, c_w]
    if _done(s3_tasks()):
        b_o, b_w = _arm_rows("B", B_WORLDS)
        arms_o.append(b_o)
        arms_w.append(b_w)
    W = pd.concat(arms_w, ignore_index=True)
    for s in ("D", "A"):
        base = W[W["arm"] == "A"].set_index("world")[f"pooled_ratio_{s}"]
        W[f"gap_left_{s}"] = (W[f"pooled_ratio_{s}"] - 1) / (W["world"].map(base) - 1)
    lines += ["## Stage 2: references by arm", "",
              "Per outer: Monte Carlo p = (1 + k) / (N + 1); upper side rejects for small T (references too large), "
              "lower side for large T. `ratio`: mean reference T / mean control T. Mechanism checks (not criteria): "
              "`refit_a_*_cell`, the cell alpha refit from each reference's own simulated pitches (median), against "
              "`outer_a_*_cell`, the outer fit's; `outcome_spread_mean` and `kappa_per_sd_mean` of the reference "
              "worlds (spec 04 §3 definitions).", "", _md(pd.concat(arms_o, ignore_index=True), "{:.4g}"), "",
              "Per world: pooled ratio (all of the world's references / mean control T), tail counts of the three "
              "outers (p <= 0.10), and gap_left = (pooled ratio - 1) / (arm A's pooled ratio - 1).", "",
              _md(W, "{:.4g}"), ""]

    cw = W[W["arm"] == "C"]
    up, lo = int(cw["upper_count_D"].sum()), int(cw["lower_count_D"].sum())
    out_of = [n for n in cw["world"] if not SCALE_RANGE[0] <= float(cw.loc[cw["world"] == n, "pooled_ratio_D"].iloc[0])
              <= SCALE_RANGE[1]]
    tail = up >= N_TRIGGER or lo >= N_TRIGGER
    fails = tail or bool(out_of)
    lines += ["## Pre-stated screen for arm C (spec 06 §5)", "",
              f"- **Tail-count rule (global):** D outers with p <= {P_CUT} over {len(outers())}: upper side {up}, "
              f"lower side {lo}; triggers at {N_TRIGGER} on one side: **{'triggered' if tail else 'not triggered'}**. "
              f"A: upper {int(cw['upper_count_A'].sum())}, lower {int(cw['lower_count_A'].sum())}.",
              f"- **Scale rule (per world):** pooled D ratio in [{SCALE_RANGE[0]}, {SCALE_RANGE[1]}]: "
              + ", ".join(f"{n} {r:.3f}" for n, r in zip(cw["world"], cw["pooled_ratio_D"]))
              + (f"; outside in: {', '.join(out_of)}." if out_of else "; all inside."),
              f"- **Exact stop rule:** C fails the screen if the global tail-count rule triggers or any world's pooled "
              f"ratio is outside the range. **Result: C {'FAILS' if fails else 'passes'} the screen.**",
              "- **How to read it:** the nominal 2.6% (4 or more of 12 on one pre-specified side under independent "
              "calibrated tests) is context, not this experiment's significance level: six outers (pilot, batter) "
              "were already examined and helped motivate this generator, so passing is not independent "
              "confirmation, and any later validation needs fresh datasets. With three outers per world, a ratio "
              "outside the range fails the screen; it does not precisely establish the size or cause of a "
              "remaining discrepancy (per-outer ratios above). A pass allows drafting a coverage spec only, which "
              "must evaluate the full procedure including hyperparameter estimation.", ""]
    return "\n".join(lines)


def _pool(fn, items, workers, label, t0):
    if not items:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        for i, _ in enumerate(ex.map(fn, items), 1):
            if i % 10 == 0 or i == len(items):
                print(f"{label}: {i}/{len(items)} done ({time.perf_counter() - t0:.0f} s)", flush=True)


def main(stage: str = "s1", workers: int = 8):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / f"manifest_{stage}.json"
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest(stage))):
        raise ValueError(f"{man} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(stage), indent=1))
    t0 = time.perf_counter()
    if stage == "s1":
        _pool(prepare_world, hier_worlds(), workers, "hier worlds", t0)
        _pool(prepare_outer, outers(), workers, "outer datasets", t0)
        _pool(run_task, s1_tasks(), workers, "controls", t0)
    else:
        if not (all(_outer_path(*k).exists() for k in outers()) and _done(s1_tasks())):
            raise RuntimeError("stage s1 must finish first")
        if stage == "s3" and not _done(s2_tasks()):
            raise RuntimeError("stage s2 must finish first")
        _pool(run_task, s2_tasks() if stage == "s2" else s3_tasks(), workers, f"{stage} references", t0)
    REPORT.write_text(summarize())
    print(f"generator diagnostic {stage} done in {time.perf_counter() - t0:.0f} s; report in {REPORT}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "s1", int(args[1]) if len(args) > 1 else 8)
