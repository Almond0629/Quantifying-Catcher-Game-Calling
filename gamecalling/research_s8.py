"""Research program S8: the combined model C1, prediction gate only (S8 specification, revision 1).

Specification: research_notes/research_program/S8_combined_model.md; model code in batter.py (runners=True).
- C1: M3a's outcome model, refitted unchanged (test 1); M4's contact regression with M1c's runner contact model as the
  offset, λ selected afresh.
- C1_off: M4's path through the same code (test 2: must reproduce M4's stored predictions exactly).
Checks before the gate: tests 1-2 (exact); test 3 (C1's contact with zero coefficients against M1c's stored
predictions, absolute error <= 1e-12, rtol 0); M3a's and M4's fingerprints for a refitted role against their stored
full-run manifests; stored outputs unchanged by hash.
Gate (contact; nats per ball in play; delta 0.0005; full precision): C1 must pass against both P0 and M4 (amendment,
2026-11-02). Otherwise C1 is M4's stored version; no other combination is tried. One run of the gate, then stop for
review: no criterion 2, scoring or validation here. Descriptive; 2024-2025 are previously examined seasons.

Run with: python3.12 -m gamecalling.research_s8 run [WORKERS]   (fits and predictions)
          python3.12 -m gamecalling.research_s8 gate             (checks, gate and report)
S8_SMOKE=<dir> runs both on one game in five, with references refitted on the same games, under that directory.

Importing research_s6 first limits every process to one math thread before numpy initializes.
"""

from gamecalling import research_s6  # noqa: I001 (first: sets the thread limit)

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from gamecalling import batter, batter_scoring as bs, config, research, research_s4
from gamecalling.phase1 import _md
from gamecalling.research import Spec
from gamecalling.research_checks import _save
from gamecalling.research_s4 import _same
from gamecalling.research_s6 import FIT_COLUMNS, verify_threads
from gamecalling.research_s6_full import changed_hashes, stored_hashes
from gamecalling.research_s7 import attach_raw

PRED = ("heldout_outcome", "heldout_contact", "forward_outcome", "forward_contact")
TOL = 1e-12
SMOKE = os.environ.get("S8_SMOKE")
STORED = config.DATA_DIR / "models"


def _subset(pitches, pas, inner=None):
    """Smoke test only: one game in five, after the version's own transform."""
    if inner is not None:
        pitches, pas = inner(pitches, pas)
    keep = lambda d: None if d is None else d[d["game_pk"] % 5 == 0].copy()
    return keep(pitches), keep(pas)


def _t(transform):
    return partial(_subset, inner=transform) if SMOKE else transform


SPECS = {
    "C1_off": Spec("C1_off", "M4's path through S8's code (runners=False): must reproduce M4's stored predictions",
                   fit_models=partial(batter.fit_models, ground_ball=True), training=batter.training,
                   transform=_t(attach_raw), extra_inputs=(config.RAW_PARQUET,), params={"runners": False}),
    "C1": Spec("C1", "M3a's outcome model; M4's contact regression with M1c's runner contact model as the offset",
               fit_models=partial(batter.fit_models, ground_ball=True, runners=True),
               training=partial(batter.training, runners=True), transform=_t(attach_raw),
               extra_inputs=(config.RAW_PARQUET,), params={"runners": True, "lambdas": list(batter.LAMBDAS)}),
}
if SMOKE:
    OUT = Path(SMOKE)
    research.MODELS_DIR = OUT / "models"
    REPORT = OUT / "research_s8_gate.md"
    SPECS["P0_smoke"] = replace(research.P0, id="P0_smoke", transform=_t(None))
    SPECS["M1c_smoke"] = replace(research_s4.SPECS["M1c"], id="M1c_smoke", transform=_t(None))
    REFS = {"P0": "P0_smoke", "M3a": "C1_off", "M4": "C1_off", "M1c": "M1c_smoke"}
else:
    OUT, REPORT = config.DATA_DIR / "research", config.REPORTS_DIR / "research_s8_gate.md"
    REFS = {"P0": "P0", "M3a": "M3a", "M4": "M4", "M1c": "M1c"}
RUN_JSON, GUARD_JSON = OUT / "s8_run.json", OUT / "s8_guard.json"


def adopted(vs_p0: bool, vs_m4: bool) -> str:
    """The amended C1 rule for contact: C1 only if it passes against both P0 and M4; otherwise M4's stored version."""
    return "C1" if vs_p0 and vs_m4 else "M4"


def guarded_files() -> list:
    files = [config.DATA_DIR / "phase1_scores.parquet"]
    for m in ("P0_repro", "M1", "M1c", "M3a", "M4", "M3a_full", "M4_full"):
        files += sorted(p for p in (STORED / m).iterdir() if p.is_file())
    return files


# --- Fits ---

def _fit_job(args):
    model_id, season, fold = args
    threads = verify_threads()
    t = time.perf_counter()
    p = pd.read_parquet(config.PITCHES_PARQUET, columns=FIT_COLUMNS + ["at_bat_number", "pitch_number", "runners_on"])
    rows = p[p["season"] == season]
    if fold != "forward":
        rows = rows[rows["fold"] == fold]
    if SPECS[model_id].transform is not None:
        rows, _ = SPECS[model_id].transform(rows, None)
    models_ = SPECS[model_id].fit_models(rows.copy())
    key = "forward" if fold == "forward" else (season, fold)
    return model_id, key, models_, {"model": model_id, "season": season, "fold": str(fold),
                                    "seconds": time.perf_counter() - t, "peak_rss_gb": bs.peak_rss_gb(), **threads}


def _fingerprint_job(model_id):
    """A refitted role's fit_id against the stored full run's: the extended fingerprint keeps earlier versions' ids."""
    from gamecalling.research_s6_benchmark import SPECS as BASE
    threads = verify_threads()
    t = time.perf_counter()
    season, fold = 2024, 0
    pitches, pas, _ = research.prepared(BASE[model_id])
    role = research.role_fitter(BASE[model_id])(pitches, pas, season, fold)
    new = bs.make_fit_id(model_id, "full", season, fold, role)
    stored = json.loads((STORED / f"{model_id}_full" / "manifest.json").read_text())["fit_ids"][f"{season}/{fold}"]
    return {"model": model_id, "role": f"{season}/{fold}", "recomputed": new, "stored": stored, "equal": new == stored,
            "seconds": time.perf_counter() - t, **threads}


def zero_coefficient_check(prefit: dict) -> list:
    """Test 3: C1's contact models with theta = 0 predict M1c's stored held-out and forward contact probabilities."""
    spec = SPECS["C1"]
    pitches = research.prepared(spec)[0]
    zero = {k: (o, replace(c, theta=np.zeros_like(c.theta))) for k, (o, c) in prefit.items()}
    roles = {k: research.PredictionRole(*v) for k, v in zero.items() if k != "forward"}
    new = {"heldout_contact": research.heldout(pitches, roles, spec, "contact"),
           "forward_contact": research.forward(pitches, spec, prefit=zero["forward"])["contact"]}
    rows = []
    for what, a in new.items():
        b = research.load(REFS["M1c"], what)
        same_rows = a.shape == b.shape and all(a[c].equals(b[c]) for c in ("season", "pa_id", "pitch_number", "y"))
        P = a.filter(regex=r"^p\d+$").to_numpy() - b.filter(regex=r"^p\d+$").to_numpy() if same_rows else np.array([np.inf])
        largest = float(np.abs(P).max())
        rows.append({"check": f"test 3: C1 contact with zero coefficients, {what} vs M1c stored", "rows": len(a),
                     "exact": bool(same_rows and largest <= TOL), "detail": f"largest |difference| {largest:.2e}"})
    return rows


def stage_run(workers: int = 8):
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (research.MODELS_DIR / "C1").exists() or RUN_JSON.exists():
        raise FileExistsError("C1's prediction run exists: the gate runs once")
    if not SMOKE:
        GUARD_JSON.write_text(json.dumps(stored_hashes(guarded_files()), indent=1))
    parent = verify_threads()
    jobs = [(m, s, f) for m in SPECS for s in config.SEASONS for f in range(config.N_FOLDS)]
    jobs += [(m, config.SEASONS[0], "forward") for m in SPECS]
    prefit, log, fingerprints = {m: {} for m in SPECS}, [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fp = [] if SMOKE else [ex.submit(_fingerprint_job, m) for m in ("M3a", "M4")]
        for model_id, key, models_, entry in ex.map(_fit_job, jobs):
            prefit[model_id][key] = models_
            log.append(entry)
            print(f"{model_id} {key}: {entry['seconds']:.0f} s", flush=True)
        fingerprints = [f.result() for f in fp]
    t_fit = time.perf_counter() - t0
    for m, spec in SPECS.items():
        research.run_predictions(spec, prefit=prefit[m], seconds_fit=sum(e["seconds"] for e in log if e["model"] == m))
    zero = zero_coefficient_check(prefit["C1"])
    RUN_JSON.write_text(json.dumps({"workers": workers, "parent_threads": parent, "jobs": log, "fingerprints": fingerprints,
                                    "zero_coefficients": zero, "wall_seconds_fits": t_fit,
                                    "wall_seconds_total": time.perf_counter() - t0,
                                    "parent_peak_rss_gb": bs.peak_rss_gb(), "budget_seconds": 3600}, indent=1))
    print(f"total {time.perf_counter() - t0:.0f} s", flush=True)


# --- Checks and gate ---

def checks(run: dict) -> pd.DataFrame:
    rows = []
    for what in PRED[0::2]:
        a = research.load("C1", what)
        ok, detail = _same(a, research.load(REFS["M3a"], what))
        rows.append({"check": f"test 1: C1 {what} vs M3a stored", "rows": len(a), "exact": ok, "detail": detail})
    for what in PRED:
        a = research.load("C1_off", what)
        ok, detail = _same(a, research.load(REFS["M4"], what))
        rows.append({"check": f"test 2: C1_off {what} vs M4 stored", "rows": len(a), "exact": ok, "detail": detail})
    rows += run["zero_coefficients"]
    for f in run["fingerprints"]:
        rows.append({"check": f"test 11: {f['model']} fit_id for refitted role {f['role']} vs stored full run",
                     "rows": 1, "exact": f["equal"], "detail": f["recomputed"].split("/")[-1]})
    if not SMOKE:
        changed = changed_hashes(json.loads(GUARD_JSON.read_text()), stored_hashes(guarded_files()))
        rows.append({"check": "test 10: stored outputs unchanged by hash", "rows": len(json.loads(GUARD_JSON.read_text())),
                     "exact": not changed, "detail": ", ".join(changed) or "none changed"})
    return pd.DataFrame(rows)


def stage_gate():
    t0 = time.perf_counter()
    run = json.loads(RUN_JSON.read_text())
    res = {"checks": checks(run), "run": run}
    print(res["checks"][["check", "exact", "detail"]].to_string(), flush=True)
    if not res["checks"]["exact"].all():
        raise RuntimeError("a check failed: stop and diagnose before applying the gate")
    res["cmp"] = {b: research.prediction_compare(REFS[b], "C1", "contact") for b in ("P0", "M4")}
    res["gate"] = {b: research.gate(c) for b, c in res["cmp"].items()}
    res["chosen"] = adopted(res["gate"]["P0"]["passes"], res["gate"]["M4"]["passes"])
    res["drop_one"] = {"M1c_contact": research.prediction_compare(REFS["M1c"], "C1", "contact"),
                       "P0_outcome": research.prediction_compare(REFS["P0"], "C1", "outcome")}
    man = research.load("C1", "manifest")
    res["alphas"], res["forward_alphas"] = pd.DataFrame(man["alphas"]), man["forward_alphas"]
    res["m4_alphas"] = pd.DataFrame(research.load(REFS["M4"], "manifest")["alphas"])
    res["code"] = man["code"]
    res["abnormal"] = pd.DataFrame([e for r in man["alphas"] + [man["forward_alphas"]]
                                    for kind in ("outcome", "contact") for e in (r.get(f"{kind}_abnormal_log") or [])],
                                   columns=["variant", "component", "season", "code", "inner", "lambda", "status",
                                            "message", "max_grad"])
    res["seconds"] = time.perf_counter() - t0
    if not SMOKE:
        _save("s8", res)
    REPORT.write_text(report(res))
    print(f"stage s8 gate done ({res['seconds']:.0f} s); contact component: {res['chosen']}", flush=True)
    return res


# --- Criterion 2 (S8 §6, stage 3) ---

SIM_SEED, SIM_N, SIM_WORKERS, SIM_BASE_OUTS, SIM_K = 20261102, 16_000_000, 4, (0, 11, 23), 0
SIM_CSV, SIM_JSON = OUT / "s8_simulation_16M_C1.csv", OUT / "s8_simulation_16M_C1.json"
ZERO_JSON = OUT / "s8_zero_solver.json"
CRIT_REPORT = REPORT.with_name("research_s8_criterion2.md")


def draw_seed(season: int, role_i: int, ctx_i: int, b: int) -> list:
    """The approved rule: [20261102, season, role_index, context_index, base_out]; selection = 0, valuation = 1."""
    return [SIM_SEED, season, role_i, ctx_i, b]


def _sim_job(args):
    role, units, policy, ctx, ctx_i, season, role_i, role_name, kind, n_sims = args
    verify_threads()
    return [{"season": season, "role": role_name, "context": ctx_i, "kind": kind, "pitcher": ctx["pitcher"],
             "stand": ctx["stand"], "batter": ctx["batter"], "scored_pitches": ctx["scored_pitches"], "base_out": b,
             **bs.simulate_identity(role, units, policy, pd.Series(ctx), b, draw_seed(season, role_i, ctx_i, b), n_sims)}
            for b in SIM_BASE_OUTS]


def _zero_job(args):
    """Test 3, solver part: C1's roles with the outcome AND contact batter coefficients zeroed (a test configuration,
    not a candidate) against M1c's array solve on the same rows, within 1e-8 (addendum's zero-coefficient comparison)."""
    from gamecalling import models, runners
    season, k, rows, sel_role, val_role, m1_sel, m1_val, units = args
    verify_threads()
    sol_s, sol_v = runners.solve_role(m1_sel, units), runners.solve_role(m1_val, units)
    p = bs.scored_pitches(rows, season, k, units)
    base = bs.values_from_solution(sol_s, p), bs.values_from_solution(sol_v, p)

    def zeroed(role):
        return replace(role, outcome=replace(role.outcome, theta=np.zeros_like(role.outcome.theta)),
                       contact=replace(role.contact, theta=np.zeros_like(role.contact.theta)))

    _, ids, pid, sel, val, pols = bs.score_fold_values(rows, season, k, zeroed(sel_role), zeroed(val_role),
                                                       f"Z/z/{season}/1/x", f"Z/z/{season}/2/x", models.fit_policy, units)
    table = bs.compare_to_base(base[0], base[1], sel, val, p["slot"].to_numpy())
    same_policy = bool(np.array_equal(pols["select"].probs, sol_s.policy.probs)
                       and np.array_equal(pols["value"].probs, sol_v.policy.probs))
    return {"season": season, "scoring_fold": k, "table": table.to_dict("records"), "identities": len(ids),
            "policy_identical_to_M1c": same_policy}


def stage_criterion2():
    from gamecalling import arsenal, models, validation
    t0 = time.perf_counter()
    if SIM_CSV.exists() or SIM_JSON.exists():
        raise FileExistsError(f"{SIM_CSV.name} exists: criterion 2 runs once (S8 §6)")
    spec = SPECS["C1"]
    m1c = SPECS["M1c_smoke"] if SMOKE else research_s4.SPECS["M1c"]
    pitches, pas, support = research.prepared(spec)
    eligible = arsenal.eligible_sets(support)
    s_fold, v_fold = arsenal.fold_roles(SIM_K)
    fit_args = [(research.role_fitter(spec), pitches[(pitches["season"] == s) & (pitches["fold"] == f)],
                 pas[(pas["season"] == s) & (pas["fold"] == f)], s, f, "C1", "criterion2")
                for s in config.SEASONS for f in (s_fold, v_fold)]
    with ProcessPoolExecutor(max_workers=SIM_WORKERS) as ex:
        fits = {(r["season"], r["fold"]): r for r in ex.map(bs.fit_job, fit_args)}
    m1 = {(s, f): research.role_fitter(m1c)(pitches, pas, s, f) for s in config.SEASONS for f in (s_fold, v_fold)}
    n_sims = 200_000 if SMOKE else SIM_N
    jobs, zero_args, contexts = [], [], []
    for season in config.SEASONS:
        units = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == SIM_K)])
        p = bs.scored_pitches(pitches, season, SIM_K, units)
        ctx = bs.simulation_contexts(p, np.random.default_rng([20261031, season, 1]))
        contexts.append(ctx.assign(season=season))
        for role_i, (name, f) in enumerate((("selection", s_fold), ("valuation", v_fold))):
            role = fits[season, f]["role"]
            policy = spec.fit_policy(role.policy_train, units)
            for i, row in ctx.iterrows():
                jobs.append((role, units, policy, row.to_dict(), int(i), season, role_i, name, row["kind"], n_sims))
        rows = pitches[(pitches["season"] == season) & (pitches["fold"] == SIM_K) & pitches["scorable"]]
        zero_args.append((season, SIM_K, rows, fits[season, s_fold]["role"], fits[season, v_fold]["role"],
                          m1[season, s_fold], m1[season, v_fold], units))
    contexts = pd.concat(contexts, ignore_index=True)
    context_check = "smoke: not compared"
    if not SMOKE:   # the same contexts as the M3a and M4 follow-up (context-selection seeds unchanged)
        cols = ["season", "context", "kind", "pitcher", "stand", "batter", "scored_pitches"]
        ref = pd.read_csv(OUT / "s6_simulation_16M_M4.csv")[cols].drop_duplicates().reset_index(drop=True)
        mine = contexts.rename_axis("context").reset_index()
        mine["context"] = mine.groupby("season").cumcount()
        context_check = bool(mine[cols].astype(ref.dtypes.to_dict()).reset_index(drop=True).equals(ref))
        if not context_check:
            raise RuntimeError("the simulation contexts differ from the M4 follow-up's: stop and diagnose")
    with ProcessPoolExecutor(max_workers=SIM_WORKERS) as ex:
        zero = list(ex.map(_zero_job, zero_args))
        rows_ = [r for part in ex.map(_sim_job, jobs) for r in part]
    ZERO_JSON.write_text(json.dumps(zero, indent=1, default=str))
    sim, z_crit = validation.add_noise_flags(pd.DataFrame(rows_))
    sim.to_csv(SIM_CSV, index=False)
    out = {"model": "C1", "checks": len(sim), "within_tol": int(sim["within_tol"].sum()),
           "beyond_noise": int(sim["beyond_noise"].sum()), "max_abs_diff": float(sim["diff"].abs().max()),
           "max_se": float(sim["se"].max()), "z_crit": z_crit, "n_sims": n_sims, "seconds": time.perf_counter() - t0,
           "workers": SIM_WORKERS, "seed_rule": "[20261102, season, role_index, context_index, base_out]; selection = 0, "
           "valuation = 1", "context_seed_rule": "[20261031, season, 1], unchanged", "contexts_equal_M4_followup":
           context_check, "fit_ids": {f"{s}/{f}": r["fit_id"] for (s, f), r in fits.items()},
           "zero_policy_identical": all(z["policy_identical_to_M1c"] for z in zero), "code": research.code_state(),
           "inputs": research.input_state(spec.extra_inputs), "parent_peak_rss_gb": bs.peak_rss_gb()}
    SIM_JSON.write_text(json.dumps(out, indent=1, default=str))
    CRIT_REPORT.write_text(criterion2_report(out, sim, zero))
    print({k: v for k, v in out.items() if k not in ("code", "inputs", "fit_ids")}, flush=True)
    failed = out["within_tol"] != out["checks"] or not out["zero_policy_identical"]
    if failed and not SMOKE:
        raise RuntimeError("criterion 2 or the zero-coefficient solver check failed: stop and diagnose; no reseeding "
                           "and no larger samples (S8 §6)")


def criterion2_report(out: dict, sim: pd.DataFrame, zero: list) -> str:
    zt = pd.concat([pd.DataFrame(z["table"]).assign(season=z["season"], identities=z["identities"],
                                                    policy_identical=z["policy_identical_to_M1c"]) for z in zero],
                   ignore_index=True)
    status = "passed" if out["within_tol"] == out["checks"] else "NOT PASSED: stop and diagnose"
    return "\n".join([
        "# Research program S8: criterion 2 for C1 (16 million at-bats)", "",
        "Generated by `python3.12 -m gamecalling.research_s8 criterion2`. Specification: S8 note §6 (stage 3) and §7 "
        "(test 3, solver part). The 96 checks of the addendum: the same contexts as the M3a and M4 follow-up, both roles, "
        "base-out states 0, 11 and 23 (both runner slices), 0.001-run limit; 96 of 96 required, no reseeding.", "",
        f"## Criterion 2: {status}", "",
        f"{out['within_tol']} of {out['checks']} within 0.001 runs; {out['beyond_noise']} beyond the Bonferroni bound "
        f"(|z| > {out['z_crit']:.2f}); largest |difference| {out['max_abs_diff']:.5f}; largest SE {out['max_se']:.5f}. "
        f"Contexts equal to the M4 follow-up's: {out['contexts_equal_M4_followup']}. Seeds `{out['seed_rule']}`. Code "
        f"`{out['code']['gamecalling_sha256'][:12]}`; {out['seconds'] / 60:.1f} min at {out['workers']} workers; parent "
        f"peak {out['parent_peak_rss_gb']:.1f} GB.", "",
        _md(sim[["season", "role", "context", "kind", "pitcher", "stand", "batter", "scored_pitches", "base_out", "solver",
                 "simulated", "diff", "se", "z", "within_tol"]], "{:.5f}"), "",
        "## Test 3, solver part: zero coefficients against M1c (1e-8)", "",
        "C1's selection and valuation roles of scoring fold 0, with the outcome and contact batter coefficients both "
        "zeroed (a test configuration, not a fitted candidate), against M1c's array solve. Every value array must agree "
        "within 1e-8 (enforced in the run); D, D_opt and D_1p are compared where recommendations agree, and tie flips "
        "are counted.", "",
        _md(zt[["season", "score", "pitches", "identities", "flips", "max_abs_diff_agreeing", "max_flip_abs_valuation_gap",
                "policy_identical"]], "{:.2e}"), ""])


# --- Report ---

def _gate_table(res) -> pd.DataFrame:
    rows = []
    for b, c in res["cmp"].items():
        g, s = res["gate"][b], c["season"].set_index("season")
        rows.append({"comparison": f"C1 vs {b}", "gain_2024": s.loc[2024, "gain"], "gain_2025": s.loc[2025, "gain"],
                     "both_seasons": g["both_seasons"], "max_cal": float(s["cal_cand"].max()),
                     "calibration": g["calibration"], "forward_gain": c["forward"]["gain"], "forward": g["forward"],
                     "passes": g["passes"]})
    return pd.DataFrame(rows)


def _loss_table(cmps: dict) -> pd.DataFrame:
    rows = []
    for b, c in cmps.items():
        for r in c["season"].itertuples():
            rows.append({"base": b, "sample": str(r.season), "n_matched": r.n, "base_loss": r.base, "C1_loss": r.cand,
                         "gain": r.gain, "max_cal_base": r.cal_base, "max_cal_C1": r.cal_cand})
        f = c["forward"]
        rows.append({"base": b, "sample": "forward 2024->2025", "n_matched": f["n"], "base_loss": f["base"],
                     "C1_loss": f["cand"], "gain": f["gain"], "max_cal_base": np.nan, "max_cal_C1": f["cal_cand"]})
    return pd.DataFrame(rows)


def _fold_table(res) -> pd.DataFrame:
    return pd.concat([c["fold"].rename(columns={"base": "base_loss", "cand": "C1_loss"}).assign(base=b)
                      for b, c in res["cmp"].items()], ignore_index=True)[
        ["base", "season", "fold", "n", "base_loss", "C1_loss", "gain"]]


def _penalty_table(res) -> pd.DataFrame:
    a = res["alphas"].to_dict("records") + [{"season": "forward", "fold": "", **res["forward_alphas"]}]
    m4 = res["m4_alphas"].set_index(["season", "fold"])["contact_lambda"]
    return pd.DataFrame([{"role": f"{r['season']} fold {r['fold']}".strip(), "lambda": f"{r['contact_lambda']:g}",
                          "M4_lambda": f"{m4[r['season'], r['fold']]:g}" if r["season"] != "forward" else "",
                          "alpha_league_r": r["contact_alpha_league_r"], "alpha_r": r["contact_alpha_r"],
                          **{f"cv_{lam:g}": r[f"contact_cv_{lam:g}"] for lam in batter.LAMBDAS}} for r in a])


def report(res) -> str:
    ch, chosen, ab, r = res["checks"], res["chosen"], res["abnormal"], res["run"]
    jobs = pd.DataFrame(r["jobs"])
    result = (f"**Gate result: {chosen}'s contact component.** " + (
        "C1 passes against both P0 and M4. The runner hierarchy is kept alongside batter information, subject to "
        "criterion 2 (next stage, after review)." if chosen == "C1" else
        "C1 does not pass against both bases, so C1 is M4's stored version: M3a's outcome model and M4's contact "
        "model, without the runner levels. No other combination is tried."))
    out = ["# Research program S8: combined model C1, prediction gate", "",
           "Generated by `python3.12 -m gamecalling.research_s8 gate` (fits by `... run`). Specification: "
           "`research_notes/research_program/S8_combined_model.md` (revision 1). Prediction-only; descriptive; 2024-2025 "
           "are previously examined seasons (exploratory reuse). One run of the gate. **The comparison with M4 is the "
           "amendment of 2026-11-02, made after inspecting the separate component results; it remains exploratory.** "
           "A passing gate shows better probabilities, not better comparisons between pitch choices.", "",
           f"Code: `{res['code']['gamecalling_sha256'][:12]}` on `{res['code']['git_head'][:7]}`, uncommitted: "
           f"{', '.join(res['code']['git_dirty_files']) or 'none'}.", "",
           "## Checks", "", _md(ch.assign(exact=ch["exact"].map({True: "yes", False: "NO"}))), "",
           "## Prediction gate (contact; nats per contact-fit-eligible ball in play; delta 0.0005; full precision)", "",
           "C1 must pass against both P0 and M4. Gain = base log loss minus C1 log loss on matched held-out observations.",
           "", _md(_gate_table(res), "{:.6f}"), "", result, "",
           "Matched held-out and forward log loss:", "", _md(_loss_table(res["cmp"]), "{:.6f}"), "",
           "Per fold (reported, not gated):", "", _md(_fold_table(res), "{:.6f}"), "",
           "## Drop-one predictions (descriptive)", "",
           "Dropping M4's batter features from contact (C1 vs. M1c), and M3a's outcome (C1 vs. P0, outcome; identical to "
           "S6's M3a result by test 1). Dropping the runner levels is the gate's comparison with M4 above.", "",
           _md(_loss_table(res["drop_one"]).replace({"base": {"M1c_contact": "M1c (contact)", "P0_outcome": "P0 (outcome)"}}),
               "{:.6f}"), "",
           "## Penalties and runner concentrations (reported, not gated)", "",
           "C1 selects its own lambda on M3a's inner folds.", "", _md(_penalty_table(res), "{:.6g}"), "",
           "## Convergence (S6 §4.3, revision 2)", "",
           *(["No fit ended with an accepted abnormal termination."] if ab.empty else
             [f"Accepted abnormal terminations: {len(ab)}.", "",
              _md(ab.assign(max_grad=ab["max_grad"].map("{:.2e}".format), **{"lambda": ab["lambda"].map("{:g}".format)}))]),
           "", "## Runtime, memory and math threads", "",
           f"{r['wall_seconds_total'] / 60:.1f} min wall-clock at {r['workers']} workers (budget 60). Parent peak "
           f"{r['parent_peak_rss_gb']:.1f} GB; largest job peak {jobs['peak_rss_gb'].max():.1f} GB. One math thread "
           f"verified in the parent and every job; largest CPU over wall time {jobs['cpu_over_wall'].max():.2f}.", "",
           _md(jobs[["model", "season", "fold", "seconds", "peak_rss_gb"]], "{:.1f}"), ""]
    return "\n".join(out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        {"gate": stage_gate, "criterion2": stage_criterion2}[stage]()
