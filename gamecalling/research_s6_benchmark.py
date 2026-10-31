"""Research program S6: pre-scoring checks and benchmarks for M3a and M4 (S6 solver addendum, revision 2).

Stages (none is a full scored run of M3a or M4; those wait until these results are reported):
- checks: the disabled extension through the default pipeline reproduces P0 exactly, P0's halves through the hook
  reproduce the stored ones, and the per-action decision-relevance routine reproduces S4's table.
- benchmark MODEL: 2024 scoring fold 0, at one math thread: both role fits and the scoring job, each in its own process
  for peak memory; the reuse check in both roles; the zero-coefficient comparison (M3a); one measured independent half;
  and the extrapolation by job schedule and memory limit.
- simulate MODEL: criterion 2, 96 checks per version, 4 million at-bats each (the original result).
- followup MODEL: the one-time rerun of addendum §12: the same checks at 16 million at-bats with fixed new draw seeds,
  to separate files. It refuses to run twice, and stops if the contexts or solver values differ from the original's.
- report: reports/research_s6_benchmark.md.

Importing research_s6 first limits every process to one math thread before numpy initializes.
"""

from gamecalling import research_s6  # noqa: I001 (first: sets the thread limit)

import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd

from gamecalling import arsenal, batter, batter_scoring as bs, config, models, research, research_s4, research_s7, scoring
from gamecalling import validation
from gamecalling import validation_pilot as vp
from gamecalling.phase1 import SCORES_PARQUET, _md
from gamecalling.research_checks import _load
from gamecalling.research_s4 import _same
from gamecalling.research_s6 import verify_threads

SPECS = {"M3a": replace(research_s6.SPECS["M3a"], pipeline=bs.batter_pipeline),
         "M4": replace(research_s7.SPECS["M4"], pipeline=bs.batter_pipeline)}
OUT = config.DATA_DIR / "research"
REPORT = config.REPORTS_DIR / "research_s6_benchmark.md"
SEASON, K = 2024, 0
MEMORY_LIMIT_GB = 32.0
BUDGET_MIN, HALVES_BUDGET_MIN = 60.0, 120.0
SIM_BASE_OUTS = (0, 11, 23)
SIM_SEED = 20261031
FOLLOWUP_N = 16_000_000
FOLLOWUP_WORKERS = 4       # from the measured memory per check (addendum §12)


def sim_files(model_id: str, followup: bool):
    tag = "s6_simulation_16M" if followup else "s6_simulation"
    return OUT / f"{tag}_{model_id}.csv", OUT / f"{tag}_{model_id}.json"


def draw_seed(season: int, role_i: int, ctx_i: int, b: int, followup: bool) -> list:
    """The original seeds, or the follow-up's fixed rule (selection = 0, valuation = 1)."""
    return [SIM_SEED, 2, season, role_i, ctx_i, b] if followup else [SIM_SEED, season, role_i, ctx_i, b]


def check_against_original(new: pd.DataFrame, original: pd.DataFrame) -> float:
    """Contexts and the solver's V(0-0) must be the original's: the refitted models are unchanged. Otherwise stop."""
    keys = ["season", "role", "context", "kind", "pitcher", "stand", "batter", "scored_pitches", "base_out"]
    if len(new) != len(original) or not new[keys].reset_index(drop=True).equals(original[keys].reset_index(drop=True)):
        raise RuntimeError("the follow-up's checks differ from the original's")
    return bs.compare_values(new["solver"].to_numpy(), original["solver"].to_numpy(), 1e-12, "solver V(0-0)")


def _isolated():
    """One fresh process per job, so each job's peak memory is its own."""
    return ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1, mp_context=multiprocessing.get_context("spawn"))


# --- Checks ---

def _p0_decision_rows(args):
    """One season of a version through run_pilot, with its per-action rows extracted (S4's reproduction check)."""
    spec_id, season, label = args
    spec = research.P0 if spec_id == "P0" else research_s4.SPECS[spec_id]
    verify_threads()
    pitches, pas, support = research.prepared(spec)
    run = scoring.run_pilot(pitches, pas, support, seasons=(season,), fit_role=research.role_fitter(spec),
                            fit_policy=spec.fit_policy, solve_role=spec.solve_role, keep_solutions=True)
    refs = []
    for k in range(config.N_FOLDS):
        sol = run.solutions[season, k]
        p = bs.scored_pitches(pitches, season, k, sol["units"])
        sel, val = bs.values_from_solution(sol["select"], p), bs.values_from_solution(sol["value"], p)
        rows = bs.action_value_rows(p, sel.Q_ref, val.Q_ref, sol["units"], k)
        refs.append(bs.write_action_values(rows, research.MODELS_DIR, "P0_repro" if spec_id == "P0" else spec_id, label,
                                           season, k))
    return spec_id, refs


def stage_checks(workers: int = 8):
    t0 = time.perf_counter()
    rows = []
    # Test 1: batter=False through the default pipeline hook and run_model.
    spec = research_s6.SPECS["M3_off"]
    research.run_model(spec, out_name="M3_off_pipeline")
    ok, detail = _same(research.load("M3_off_pipeline", "scores"), pd.read_parquet(SCORES_PARQUET))
    rows.append({"check": "test 1: batter=False via the pipeline hook: scores vs stored P0", "exact": ok, "detail": detail})
    for what in research_s6.PRED:
        ok, detail = _same(research.load("M3_off_pipeline", what), research.load("P0", what))
        rows.append({"check": f"test 1: {what} vs P0_repro", "exact": ok, "detail": detail})
    # Test 11 / halves: P0's independent halves through the hook.
    halves = research.independent_halves(research.P0, workers)
    stored = vp._load("half_runs")
    for key in sorted(halves):
        a, b = halves[key]["scores"], stored[key]["scores"]
        ok = a.shape == b.shape and all(a[c].equals(b[c]) for c in b.columns)
        rows.append({"check": f"test 10: P0 half {key} via the hook vs stored", "exact": ok,
                     "detail": "identical" if ok else "differs"})
    # S4's decision relevance through the per-action routine.
    label = "s4check"
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        results = list(ex.map(_p0_decision_rows, [(m, s, label) for m in ("P0", "M1c") for s in config.SEASONS]))
    refs = {m: [r for mm, rr in results if mm == m for r in rr] for m in ("P0", "M1c")}
    new = bs.decision_relevance_rows(refs["P0"], refs["M1c"]).set_index("season")
    s4 = _load("s4")["scored"]["relevance"].set_index("season")
    s4_rows = []
    for season in config.SEASONS:
        for mine, theirs in (("share_rec_changed", "share_rec_changed"), ("sd_pair_change", "sd_pair_change"),
                             ("sd_pair_base", "sd_pair_P0"), ("ratio", "ratio")):
            s4_rows.append({"season": season, "measure": mine, "new_routine": new.loc[season, mine],
                            "S4_table": s4.loc[season, theirs],
                            "abs_diff": abs(new.loc[season, mine] - s4.loc[season, theirs])})
    s4_t = pd.DataFrame(s4_rows)
    rows.append({"check": "S4 reproduction: decision relevance within 1e-12", "exact": bool((s4_t["abs_diff"] <= 1e-12).all()),
                 "detail": f"largest difference {s4_t['abs_diff'].max():.2e}"})
    out = {"checks": rows, "s4": s4_rows, "seconds": time.perf_counter() - t0}
    (OUT / "s6_benchmark_checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(pd.DataFrame(rows).to_string())
    if not all(r["exact"] for r in rows):
        raise RuntimeError("a pre-benchmark check failed")


# --- Benchmark ---

def _reuse_job(args):
    """The reuse check for a scoring fold, in both roles. A discrepancy above 1e-8 raises and stops the work."""
    rows, sel_role, val_role, sel_id, val_id, fit_policy, units = args
    verify_threads()
    t0 = time.perf_counter()
    p, ids, pid, sel, val, pols = bs.score_fold_values(rows, SEASON, K, sel_role, val_role, sel_id, val_id, fit_policy, units)
    roles = {"selection": (sel_role, sel, pols["select"]), "valuation": (val_role, val, pols["value"])}
    missing = (set(p["batter"]) - set(sel_role.outcome.features.batters)) | \
        (set(p["batter"]) - set(val_role.outcome.features.batters))
    sample = bs.reuse_sample(p, ids, pid, sel, SEASON, K, no_history=missing)
    out = []
    for role_name, (role, values, policy) in roles.items():
        for group, positions in sample.items():
            r = bs.reuse_check(role, units, policy, p, ids, pid, values, positions)
            out.append({"role": role_name, "group": group, **r})
    identity_ok = len(ids) == len(p[bs.IDENTITY_KEYS].drop_duplicates())
    return {"rows": out, "identities": len(ids), "identity_count_ok": identity_ok, "seconds": time.perf_counter() - t0,
            "peak_rss_gb": bs.peak_rss_gb()}


def _zero_job(args):
    """Real-context zero-coefficient comparison with P0's array solve: values, agreeing recommendations, tie flips, pi."""
    rows, pitches_sel, pas_sel, pitches_val, pas_val, sel_role, val_role, fit_policy, units = args
    verify_threads()
    t0 = time.perf_counter()
    p0_s = models.fit_role(pitches_sel, pas_sel, SEASON, sel_role.fold)
    p0_v = models.fit_role(pitches_val, pas_val, SEASON, val_role.fold)
    sol_s, sol_v = scoring.solve_role(p0_s, units, fit_policy), scoring.solve_role(p0_v, units, fit_policy)
    p = bs.scored_pitches(rows, SEASON, K, units)
    base = bs.values_from_solution(sol_s, p), bs.values_from_solution(sol_v, p)

    def zeroed(role):
        return replace(role, outcome=replace(role.outcome, theta=np.zeros_like(role.outcome.theta)),
                       contact=replace(role.contact, theta=np.zeros_like(role.contact.theta)))

    _, ids, pid, sel, val, pols = bs.score_fold_values(rows, SEASON, K, zeroed(sel_role), zeroed(val_role), "Z/z/2024/1/x",
                                                       "Z/z/2024/2/x", fit_policy, units)
    table = bs.compare_to_base(base[0], base[1], sel, val, p["slot"].to_numpy())
    policy_same = bool(np.array_equal(pols["select"].probs, sol_s.policy.probs)
                       and np.array_equal(pols["value"].probs, sol_v.policy.probs))
    return {"table": table.to_dict("records"), "policy_identical_to_P0": policy_same, "seconds": time.perf_counter() - t0}


def _half_measure(args):
    verify_threads()
    t0 = time.perf_counter()
    r = research._half_job(args)
    return {"seconds": time.perf_counter() - t0, "peak_rss_gb": bs.peak_rss_gb(), "scored_pitches": len(r["scores"])}


def list_schedule(durations, workers: int) -> float:
    """Wall-clock of jobs assigned in order to the first free worker."""
    free = [0.0] * workers
    for d in durations:
        i = int(np.argmin(free))
        free[i] += d
    return max(free)


def stored_forward_seconds(model_id: str) -> float:
    log = json.loads((OUT / ("s6_run.json" if model_id == "M3a" else "s7_run.json")).read_text())["jobs"]
    return next(e["seconds"] for e in log if e["model"] == model_id and e["fold"] == "forward")


def stage_benchmark(model_id: str, label: str = "benchmark"):
    spec = SPECS[model_id]
    t0 = time.perf_counter()
    parent = verify_threads()
    pitches, pas, support = research.prepared(spec)
    t_prepare = time.perf_counter() - t0
    fit_role = research.role_fitter(spec)
    s_fold, v_fold = arsenal.fold_roles(K)
    eligible = arsenal.eligible_sets(support)
    units = models.build_units(eligible[(eligible["season"] == SEASON) & (eligible["scoring_fold"] == K)])
    fold_rows = {f: (pitches[(pitches["season"] == SEASON) & (pitches["fold"] == f)],
                     pas[(pas["season"] == SEASON) & (pas["fold"] == f)]) for f in (s_fold, v_fold)}
    rows = pitches[(pitches["season"] == SEASON) & (pitches["fold"] == K) & pitches["scorable"]]
    res = {"model": model_id, "season": SEASON, "scoring_fold": K, "run_label": label, "parent_threads": parent,
           "seconds_prepare_tables": t_prepare}
    fits = {}
    for name, f in (("selection", s_fold), ("valuation", v_fold)):
        with _isolated() as ex:
            fits[name] = ex.submit(bs.fit_job, (fit_role, *fold_rows[f], SEASON, f, spec.id, label)).result()
        print(f"{model_id} fit {name}: {fits[name]['seconds']:.0f} s, {fits[name]['peak_rss_gb']:.1f} GB", flush=True)
    sel_role, val_role = fits["selection"]["role"], fits["valuation"]["role"]
    sel_id, val_id = fits["selection"]["fit_id"], fits["valuation"]["fit_id"]
    with _isolated() as ex:
        score = ex.submit(bs.score_job, (rows, SEASON, K, sel_role, val_role, sel_id, val_id, spec.fit_policy, units,
                                         research.MODELS_DIR, spec.id, label)).result()
    log = pd.DataFrame(score["solve_log"])
    print(f"{model_id} score job: {score['seconds']:.0f} s, {score['peak_rss_gb']:.1f} GB", flush=True)
    res["fits"] = {k: {x: v[x] for x in ("seconds", "peak_rss_gb", "fit_id", "libraries", "cpu_over_wall")}
                   for k, v in fits.items()}
    res["score_job"] = {x: score[x] for x in ("seconds", "seconds_score", "seconds_write", "peak_rss_gb", "libraries",
                                              "cpu_over_wall")}
    res["solve_log"] = log.to_dict("records")
    res["artifact"] = score["ref"]
    res["scored_pitches"] = len(score["scores"])
    res["fold_pitches_P0"] = int(((p0 := pd.read_parquet(SCORES_PARQUET, columns=["season", "fold"]))["season"].eq(SEASON)
                                  & p0["fold"].eq(K)).sum())
    res["abnormal"] = [e for r in (sel_role, val_role) for m in (r.outcome, r.contact) for e in getattr(m, "abnormal", [])]
    # Reuse in both roles.
    with _isolated() as ex:
        reuse = ex.submit(_reuse_job, (rows, sel_role, val_role, sel_id, val_id, spec.fit_policy, units)).result()
    res["reuse"] = reuse
    print(f"{model_id} reuse: {reuse['seconds']:.0f} s", flush=True)
    # Zero coefficients: the offset is the same in both versions, so M3a's comparison covers the path.
    if model_id == "M3a":
        with _isolated() as ex:
            zero = ex.submit(_zero_job, (rows, *fold_rows[s_fold], *fold_rows[v_fold], sel_role, val_role, spec.fit_policy,
                                         units)).result()
        res["zero"] = zero
        print(f"{model_id} zero coefficients: flips {[r['flips'] for r in zero['table']]}", flush=True)
    # One measured independent half.
    halves = vp._load("v0")["halves"]
    games = set(pitches.loc[pitches["season"] == SEASON, "game_pk"].unique())
    g = np.sort(halves.index[(halves == 0).to_numpy() & halves.index.isin(games)])
    with _isolated() as ex:
        res["half"] = ex.submit(_half_measure, (spec, SEASON, 0, g, f"{label}_half_{SEASON}_0")).result()
    print(f"{model_id} half: {res['half']['seconds']:.0f} s, {res['half']['peak_rss_gb']:.1f} GB", flush=True)
    res["peak_rss_parent_gb"] = bs.peak_rss_gb()
    res["estimate"] = estimate(res, model_id)
    (OUT / f"s6_benchmark_{model_id}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["estimate"], indent=1), flush=True)


def estimate(res: dict, model_id: str) -> dict:
    """Extrapolation: the plan's serial x 6, and the actual job schedule under the memory limit."""
    fit = [res["fits"]["selection"]["seconds"], res["fits"]["valuation"]["seconds"]]
    fit_mean, score_s = float(np.mean(fit)), res["score_job"]["seconds"]
    fit_rss = max(res["fits"]["selection"]["peak_rss_gb"], res["fits"]["valuation"]["peak_rss_gb"])
    score_rss, parent_rss = res["score_job"]["peak_rss_gb"], res["peak_rss_parent_gb"]
    forward_s = stored_forward_seconds(model_id)

    def workers_for(job_rss):
        return max(1, min(8, int((MEMORY_LIMIT_GB - parent_rss) // job_rss)))

    w_fit, w_score = workers_for(fit_rss), workers_for(score_rss)
    wall_fit = list_schedule([fit_mean] * 6, w_fit)
    wall_score = list_schedule([score_s] * 6, w_score)
    full_wall = wall_fit + wall_score + forward_s
    half_s, half_rss = res["half"]["seconds"], res["half"]["peak_rss_gb"]
    w_half = max(1, min(4, int((MEMORY_LIMIT_GB - parent_rss) // half_rss)))
    halves_wall = list_schedule([half_s] * 4, w_half)
    return {"serial_season_fold_seconds": sum(fit) + score_s, "serial_x6_minutes": 6 * (sum(fit) + score_s) / 60,
            "jobs": {"role_fits": 6, "scoring_folds": 6, "forward_fit_in_parent": 1},
            "fit_job_seconds": fit_mean, "score_job_seconds": score_s, "forward_seconds_from_stored_run": forward_s,
            "workers_fit": w_fit, "workers_score": w_score, "peak_sum_fit_gb": parent_rss + w_fit * fit_rss,
            "peak_sum_score_gb": parent_rss + w_score * score_rss, "full_run_wall_minutes": full_wall / 60,
            "full_run_budget_minutes": BUDGET_MIN, "halves_workers": w_half, "halves_peak_sum_gb": parent_rss + w_half * half_rss,
            "halves_wall_minutes": halves_wall / 60, "halves_budget_minutes": HALVES_BUDGET_MIN,
            "within_budget": full_wall / 60 <= BUDGET_MIN and halves_wall / 60 <= HALVES_BUDGET_MIN}


# --- Simulation (criterion 2) ---

def _sim_job(args):
    verify_threads()
    role, units, policy, ctx, ctx_i, season, role_i, role_name, kind, n_sims, followup = args
    rows = []
    for b in SIM_BASE_OUTS:
        r = bs.simulate_identity(role, units, policy, pd.Series(ctx), b, draw_seed(season, role_i, ctx_i, b, followup), n_sims)
        rows.append({"season": season, "role": role_name, "context": ctx_i, "kind": kind, "pitcher": ctx["pitcher"],
                     "stand": ctx["stand"], "batter": ctx["batter"], "scored_pitches": ctx["scored_pitches"], "base_out": b, **r})
    return rows


def stage_simulate(model_id: str, workers: int = 8, followup: bool = False):
    t0 = time.perf_counter()
    spec = SPECS[model_id]
    csv_path, json_path = sim_files(model_id, followup)
    if followup and (csv_path.exists() or json_path.exists()):
        raise FileExistsError(f"{csv_path.name} exists: the follow-up is one fixed rerun (addendum §12)")
    n_sims = FOLLOWUP_N if followup else 4_000_000
    pitches, pas, support = research.prepared(spec)
    fit_role = research.role_fitter(spec)
    eligible = arsenal.eligible_sets(support)
    s_fold, v_fold = arsenal.fold_roles(K)
    fit_args = [(fit_role, pitches[(pitches["season"] == s) & (pitches["fold"] == f)],
                 pas[(pas["season"] == s) & (pas["fold"] == f)], s, f, spec.id, "simulation")
                for s in config.SEASONS for f in (s_fold, v_fold)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fits = {(r["season"], r["fold"]): r["role"] for r in ex.map(bs.fit_job, fit_args)}
    jobs, contexts = [], []
    for season in config.SEASONS:
        units = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == K)])
        p = bs.scored_pitches(pitches, season, K, units)
        ctx = bs.simulation_contexts(p, np.random.default_rng([SIM_SEED, season, 1]))
        contexts.append(ctx.assign(season=season))
        for role_i, (name, f) in enumerate((("selection", s_fold), ("valuation", v_fold))):
            role = fits[season, f]
            policy = spec.fit_policy(role.policy_train, units)
            for i, row in ctx.iterrows():
                jobs.append((role, units, policy, row.to_dict(), int(i), season, role_i, name, row["kind"], n_sims, followup))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_sim_job, jobs) for r in part]
    sim, z_crit = validation.add_noise_flags(pd.DataFrame(rows))
    provenance = {}
    if followup:
        original = sim_files(model_id, False)[0]
        provenance = {"amendment": "S6_solver_addendum.md §12", "seed_rule": "[20261031, 2, season, role_index, context_index, "
                      "base_out]; selection = 0, valuation = 1", "context_seed_rule": "[20261031, season, 1], unchanged",
                      "solver_max_diff_vs_original": check_against_original(sim, pd.read_csv(original)),
                      "code": research.code_state(), "inputs": research.input_state(spec.extra_inputs),
                      "original_file_sha256": research._sha(original), "workers": workers}
    sim.to_csv(csv_path, index=False)
    out = {"model": model_id, "checks": len(sim), "within_tol": int(sim["within_tol"].sum()),
           "beyond_noise": int(sim["beyond_noise"].sum()), "max_abs_diff": float(sim["diff"].abs().max()),
           "max_se": float(sim["se"].max()), "z_crit": z_crit, "n_sims": n_sims, "seconds": time.perf_counter() - t0,
           **provenance}
    json_path.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("code", "inputs")}, flush=True)
    if followup and out["within_tol"] != out["checks"]:
        raise RuntimeError(f"{model_id}: {out['checks'] - out['within_tol']} follow-up check(s) failed: stop and diagnose; "
                           "no further reseeding or larger samples (addendum §12)")


# --- Report ---

def _benchmark_section(model_id: str) -> list:
    r = json.loads((OUT / f"s6_benchmark_{model_id}.json").read_text())
    e, log = r["estimate"], pd.DataFrame(r["solve_log"])
    job = pd.DataFrame([{"job": f"fit {k}", "seconds": v["seconds"], "peak_rss_gb": v["peak_rss_gb"],
                         "cpu_over_wall": v["cpu_over_wall"]} for k, v in r["fits"].items()]
                       + [{"job": "scoring fold (both solves, scoring, per-action file)", "seconds": r["score_job"]["seconds"],
                           "peak_rss_gb": r["score_job"]["peak_rss_gb"], "cpu_over_wall": r["score_job"]["cpu_over_wall"]},
                          {"job": f"one independent half ({SEASON}, half 0)", "seconds": r["half"]["seconds"],
                           "peak_rss_gb": r["half"]["peak_rss_gb"], "cpu_over_wall": np.nan}])
    solve = log[["role", "identities", "contexts", "batches", "iterations", "max_iterations", "seconds_probabilities",
                 "seconds_solve"]]
    reuse = pd.DataFrame(r["reuse"]["rows"])
    sim = json.loads((OUT / f"s6_simulation_{model_id}.json").read_text())
    sim_t = pd.read_csv(OUT / f"s6_simulation_{model_id}.csv")
    out = [f"## {model_id}", "",
           f"**Budget result: {'within' if e['within_budget'] else 'OVER'} budget.** Full run about "
           f"{e['full_run_wall_minutes']:.1f} min (budget {e['full_run_budget_minutes']:.0f}); independent halves about "
           f"{e['halves_wall_minutes']:.1f} min (budget {e['halves_budget_minutes']:.0f}).", "",
           f"**Reuse:** the largest separate-vs.-reused difference is "
           f"{reuse[[c for c in reuse if c.startswith('max_diff')]].to_numpy().max():.2e} (required at most 1e-8).", "",
           f"**Criterion 2: {'passed' if sim['within_tol'] == sim['checks'] else 'NOT PASSED as specified'}.** "
           f"{sim['within_tol']} of {sim['checks']} within 0.001 runs (every check is required); {sim['beyond_noise']} beyond "
           f"the Bonferroni bound (|z| > {sim['z_crit']:.2f}); largest |difference| {sim['max_abs_diff']:.5f}, largest SE "
           f"{sim['max_se']:.5f}.", "", *_sim_diagnostic(model_id),
           "### Jobs (2024 scoring fold 0; each in its own process; one math thread)", "", _md(job, "{:.2f}"), "",
           f"Scored pitches {r['scored_pitches']:,} (P0's fold: {r['fold_pitches_P0']:,}). Scoring within the job: "
           f"{r['score_job']['seconds_score']:.1f} s; writing the per-action file: {r['score_job']['seconds_write']:.1f} s "
           f"({r['artifact']['rows']:,} rows). Parent process peak: {r['peak_rss_parent_gb']:.1f} GB.", "",
           "### Solves", "", _md(solve, "{:.1f}"), "", "### Extrapolation", "",
           f"- **Serial, the plan's check:** season-fold total {e['serial_season_fold_seconds'] / 60:.1f} min × 6 = "
           f"{e['serial_x6_minutes']:.1f} min. This fits two roles per scoring fold; a full run fits each role once.",
           f"- **Job schedule:** {e['jobs']['role_fits']} role fits on {e['workers_fit']} workers (peak sum "
           f"{e['peak_sum_fit_gb']:.1f} GB with the parent), then {e['jobs']['scoring_folds']} scoring-fold jobs on "
           f"{e['workers_score']} workers (peak sum {e['peak_sum_score_gb']:.1f} GB), then the forward fit in the parent "
           f"({e['forward_seconds_from_stored_run']:.0f} s, from the stored prediction run): about "
           f"{e['full_run_wall_minutes']:.1f} min. Each job's duration is measured on one fold; serial job durations are "
           "not divided by eight, but concurrency slowdown is not measured.",
           f"- **Independent halves:** 4 half jobs on {e['halves_workers']} workers (peak sum {e['halves_peak_sum_gb']:.1f} GB), "
           f"using the one measured half: about {e['halves_wall_minutes']:.1f} min. Other halves may differ.", "",
           "### Reuse check (both roles; each identity solved alone; tolerance 1e-8)", "", _md(reuse.assign(**{
               c: reuse[c].map("{:.2e}".format) for c in reuse if c.startswith("max_diff")})), "",
           f"Seed `[20261031, {SEASON}, {K}]`; 500 pitches uniform from the fold's pitches sorted by key, then up to 100 each "
           f"from the three groups. Solve identities: {r['reuse']['identities']:,}.", ""]
    if r["abnormal"]:
        out += ["### Accepted abnormal terminations in these fits", "", _md(pd.DataFrame(r["abnormal"]).assign(
            max_grad=lambda d: d["max_grad"].map("{:.2e}".format), **{"lambda": lambda d: d["lambda"].map("{:g}".format)})), ""]
    if "zero" in r:
        z = pd.DataFrame(r["zero"]["table"])
        out += ["### Zero coefficients against P0's array solve (real contexts)", "",
                f"All value arrays agree within 1e-8 in both roles. π identical to P0's: "
                f"{'yes' if r['zero']['policy_identical_to_P0'] else 'NO'}. Tie flips, each checked against the valuation "
                "difference between the two recommendations:", "", _md(z.assign(
                    max_abs_diff_agreeing=z["max_abs_diff_agreeing"].map("{:.2e}".format),
                    max_flip_abs_valuation_gap=z["max_flip_abs_valuation_gap"].map("{:.2e}".format))), ""]
    wide = sim_t[["season", "role", "context", "kind", "pitcher", "stand", "batter", "scored_pitches", "base_out", "solver",
                  "simulated", "diff", "se", "z", "within_tol"]]
    worst = wide.reindex(wide["diff"].abs().sort_values(ascending=False).index).head(8)
    out += [f"### Simulation (criterion 2): the 8 largest of {sim['checks']} (all in `data/research/s6_simulation_{model_id}.csv`)",
            "", _md(worst, "{:.5f}"), ""]
    return out


def _sim_diagnostic(model_id: str) -> list:
    path = OUT / "s6_simulation_diagnostic.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    out = [f"Noise context (descriptive): about {d[model_id]['expected_misses_noise']:.2f} misses of 0.001 runs are expected "
           f"from simulation noise alone, given each check's SE; largest |z| {d[model_id]['max_abs_z']:.2f}.", ""]
    f = d["failing_check"]
    if model_id == "M4":
        out += [f"**The one miss:** {f['season']} {f['role']} role, base-out {f['base_out']}: simulated {f['simulated']:.5f}, solver "
                f"{f['solver']:.5f}, difference {f['diff']:.5f}, SE {f['se']:.5f}, z {f['z']:.2f}. An independent direct linear solve "
                f"of this context gives {f['direct_linear_solve']:.8f}, within {f['abs_solver_minus_direct']:.1e} of the solver. "
                "This points to simulation noise, not a solver error, but it does not change the result: the specified rule "
                "requires every check to pass, the check was not rerun, and M4 is not cleared for scoring.", ""]
    return out


def _followup_section() -> list:
    out = []
    for m in SPECS:
        csv_path, json_path = sim_files(m, True)
        if not json_path.exists():
            continue
        j = json.loads(json_path.read_text())
        d = pd.read_csv(csv_path)
        if not out:
            out = ["## One-time follow-up of criterion 2 (addendum §12)", "",
                   "All 96 checks per version at 16,000,000 at-bats. Only the simulation size and draw seeds changed "
                   f"(`{j['seed_rule']}`). The original 4-million results above stand as recorded.", ""]
        status = "passed" if j["within_tol"] == j["checks"] else "NOT PASSED: stop and diagnose"
        out += [f"### {m}: {status}", "",
                f"{j['within_tol']} of {j['checks']} within 0.001 runs; {j['beyond_noise']} beyond the Bonferroni bound "
                f"(|z| > {j['z_crit']:.2f}); largest |difference| {j['max_abs_diff']:.5f}; largest SE {j['max_se']:.5f}. Solver "
                f"V(0-0) against the original file: largest difference {j['solver_max_diff_vs_original']:.1e}. Code "
                f"`{j['code']['gamecalling_sha256'][:12]}`; {j['seconds'] / 60:.1f} min at {j['workers']} workers.", "",
                _md(d[["season", "role", "context", "kind", "pitcher", "stand", "batter", "scored_pitches", "base_out", "solver",
                       "simulated", "diff", "se", "z", "within_tol"]], "{:.5f}"), ""]
    return out


def stage_report():
    c = json.loads((OUT / "s6_benchmark_checks.json").read_text())
    ch = pd.DataFrame(c["checks"])
    out = ["# Research program S6: checks and benchmarks before scoring (M3a and M4)", "",
           "Generated by `python3.12 -m gamecalling.research_s6_benchmark report`. Specification: "
           "`research_notes/research_program/S6_solver_addendum.md` (revision 2). **No full scored run of M3a or M4 has "
           "been started.** The one scoring fold and one half below were scored only to measure them, and are written under "
           "their own run labels. Descriptive; seasons previously examined.", "",
           "**Limits of these measurements:** runtime estimates use one fold's job durations with no measured concurrency "
           "slowdown (the 8-worker S6 run took about 52 s per M3a role fit, against about 24 s alone here); swapping was not "
           "measured directly, only peak memory; and no numerical tie occurred in the sampled fold, so the tie group of the "
           "reuse check and the tie-flip check had nothing to test on real data here.", "",
           "## Pre-benchmark checks", "", _md(ch.assign(exact=ch["exact"].map({True: "yes", False: "NO"}))), "",
           "S4's decision relevance through the per-action routine:", "",
           _md(pd.DataFrame(c["s4"]).assign(abs_diff=lambda d: d["abs_diff"].map("{:.2e}".format)), "{:.8f}"), ""]
    for m in SPECS:
        out += _benchmark_section(m)
    out += _followup_section()
    REPORT.write_text("\n".join(out))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "checks":
        stage_checks()
    elif stage == "benchmark":
        stage_benchmark(sys.argv[2], *(sys.argv[3:4]))
    elif stage == "simulate":
        stage_simulate(sys.argv[2])
    elif stage == "followup":
        stage_simulate(sys.argv[2], FOLLOWUP_WORKERS, followup=True)
    else:
        stage_report()
