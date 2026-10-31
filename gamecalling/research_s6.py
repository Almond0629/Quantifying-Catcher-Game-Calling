"""Research program S6, step 1: batter-index prediction gate (M3a, M3b), prediction-only.

Specification: research_notes/research_program/S6_batter_index.md (revision 2); model code in batter.py.
- M3_off: M3's code path with batter=False, as a full scored run; must reproduce P0's stored scores and
  predictions exactly (test 1).
- M3a, M3b: prediction-only runs (research.run_predictions). No policy, solver, scoring or clustering.
Gate, component by component (outcome per fit-eligible pitch, contact per ball in play; delta 0.0005; full
precision; both seasons; calibration <= 0.02; forward no worse), with the nested rule: M3b's component only if it
passes against both P0's and M3a's; otherwise M3a's if it passes against P0; otherwise P0's. Fold gains, the
penalties, the shrinkage hyperparameters, coefficient summaries and missing-history shares are reported, not
gated. Descriptive; 2024-2025 are previously examined seasons (exploratory reuse).

Each process is limited to one math thread: set below, before any numerical library initializes, and verified in
every fit (threadpoolctl, plus measured CPU time over wall time). A violation stops the run.

Run with: python3.12 -m gamecalling.research_s6 run [WORKERS]   (fit the versions in parallel; timed)
          python3.12 -m gamecalling.research_s6 gate             (checks, nested gate and report)
"""

import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")
os.environ.update({v: "1" for v in THREAD_VARS})

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from gamecalling import arsenal, batter, config, research
from gamecalling.phase1 import SCORES_PARQUET, _md
from gamecalling.research import Spec
from gamecalling.research_checks import _save
from gamecalling.research_s4 import _same

SPECS = {
    "M3_off": Spec("M3_off", "M3 with the batter extension explicitly disabled (batter=False): P0's own functions "
                   "only (full scored run)", fit_models=partial(batter.fit_models, batter=False),
                   training=batter.training, params={"batter": False}),
    "M3a": Spec("M3a", "Batter index, main effects: ridge multinomial with log P0 offset; outcome and contact",
                fit_models=partial(batter.fit_models, variant="M3a"), training=batter.training,
                params={"variant": "M3a", "lambdas": list(batter.LAMBDAS)}),
    "M3b": Spec("M3b", "Batter index, main effects plus four family contrasts; outcome and contact",
                fit_models=partial(batter.fit_models, variant="M3b"), training=batter.training,
                params={"variant": "M3b", "lambdas": list(batter.LAMBDAS)}),
}
SPECS["P0_predonly"] = Spec("P0_predonly", "P0 through the prediction-only path with prefit models (check of the path)",
                            params={"prediction_only": True})
FULL_RUNS = ("M3_off",)
PREDICTION_RUNS = ("P0_predonly", "M3a", "M3b")
FIT_COLUMNS = ["season", "fold", "game_pk", "batter", "pitcher", "action", "stand", "balls", "strikes", "description",
               "outcome", "zone", "contact_bin", "xwoba", "outcome_fit_eligible", "contact_fit_eligible"]
RUN_JSON = config.DATA_DIR / "research" / "s6_run.json"
COMPARISONS = (("P0", "M3a"), ("P0", "M3b"), ("M3a", "M3b"))
KINDS = ("outcome", "contact")


def check_one_thread(info: list, cpu_over_wall: float) -> None:
    """Stops the run unless every numerical library reports one thread and a matrix product used one core."""
    over = [(lib["internal_api"], lib["num_threads"]) for lib in info if lib["num_threads"] != 1]
    if not info or over:
        raise RuntimeError(f"math threads are not limited to one: {over or 'no library reported'}")
    if cpu_over_wall > 1.5:
        raise RuntimeError(f"a matrix product used {cpu_over_wall:.2f} cores")


def verify_threads() -> dict:
    """The limit that actually holds in this process."""
    a = np.random.default_rng(0).normal(size=(1500, 1500))
    wall, cpu = time.perf_counter(), time.process_time()
    a @ a
    ratio = (time.process_time() - cpu) / (time.perf_counter() - wall)
    info = threadpool_info()
    check_one_thread(info, ratio)
    return {"libraries": {lib["internal_api"]: lib["num_threads"] for lib in info}, "cpu_over_wall": ratio}


def _fit_job(args):
    """Fit one role, or the forward fit, in a worker. Reads only the columns the fit uses."""
    model_id, season, fold = args
    threads = verify_threads()
    t = time.perf_counter()
    p = pd.read_parquet(config.PITCHES_PARQUET, columns=FIT_COLUMNS)
    rows = p[p["season"] == season]
    if fold != "forward":
        rows = rows[rows["fold"] == fold]
    outcome, contact = SPECS[model_id].fit_models(rows.copy())
    key = "forward" if fold == "forward" else (season, fold)
    return model_id, key, (outcome, contact), {"model": model_id, "season": season, "fold": str(fold),
                                               "seconds": time.perf_counter() - t, **threads}


def _full_run_job(model_id):
    threads = verify_threads()
    t = time.perf_counter()
    research.run_model(SPECS[model_id])
    return model_id, {"model": model_id, "seconds": time.perf_counter() - t, **threads}


def stage_run(workers: int = 8):
    """All fits in parallel, one math thread each; then the predictions; all timed against the budget."""
    t0 = time.perf_counter()
    parent = verify_threads()
    jobs = [(m, s, f) for m in PREDICTION_RUNS for s in config.SEASONS for f in range(config.N_FOLDS)]
    jobs += [(m, config.SEASONS[0], "forward") for m in PREDICTION_RUNS]
    prefit, log = {m: {} for m in PREDICTION_RUNS}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        full = [ex.submit(_full_run_job, m) for m in FULL_RUNS]
        for model_id, key, models_, entry in ex.map(_fit_job, jobs):
            prefit[model_id][key] = models_
            log.append(entry)
            print(f"{model_id} {key}: {entry['seconds']:.0f} s", flush=True)
        for fut in full:
            model_id, entry = fut.result()
            log.append(entry)
            print(f"{model_id}: {entry['seconds']:.0f} s", flush=True)
    t_fit = time.perf_counter() - t0
    for m in PREDICTION_RUNS:
        t = time.perf_counter()
        research.run_predictions(SPECS[m], prefit=prefit[m],
                                 seconds_fit=sum(e["seconds"] for e in log if e["model"] == m))
        print(f"{m}: predictions {time.perf_counter() - t:.0f} s", flush=True)
    RUN_JSON.write_text(json.dumps({"workers": workers, "parent_threads": parent, "jobs": log, "wall_seconds_fits": t_fit,
                                    "wall_seconds_total": time.perf_counter() - t0, "budget_seconds": 3600}, indent=1))
    print(f"total {time.perf_counter() - t0:.0f} s", flush=True)


# --- Checks and reported quantities ---

PRED = ("heldout_outcome", "heldout_contact", "forward_outcome", "forward_contact")


def checks() -> pd.DataFrame:
    off = research.load("M3_off", "scores")
    ok, detail = _same(off, pd.read_parquet(SCORES_PARQUET))
    rows = [{"check": "test 1: M3_off scores vs data/phase1_scores.parquet", "rows": len(off), "exact": ok,
             "detail": detail}]
    for mid, label in (("M3_off", "test 1: M3_off"), ("P0_predonly", "prediction-only path with prefit models: P0")):
        for what in PRED:
            a = research.load(mid, what)
            ok, detail = _same(a, research.load("P0", what))
            rows.append({"check": f"{label} {what} vs P0_repro", "rows": len(a), "exact": ok, "detail": detail})
    return pd.DataFrame(rows)


def abnormal_log() -> pd.DataFrame:
    """Every accepted abnormal termination (S6 §4.3), from the manifests."""
    rows = []
    for m in ("M3a", "M3b"):
        man = research.load(m, "manifest")
        for r in man["alphas"] + [man["forward_alphas"]]:
            for kind in KINDS:
                rows += r.get(f"{kind}_abnormal_log") or []
    return pd.DataFrame(rows, columns=["variant", "component", "season", "code", "inner", "lambda", "status", "message",
                                       "max_grad"])


def role_sizes(model_id: str = "M3b") -> pd.DataFrame:
    """Each outer role's rows and games, from the fit's manifest, against the pitch table's single fold."""
    p = research.base_tables()[0]
    a = pd.DataFrame(research.load(model_id, "manifest")["alphas"])
    rows = []
    for r in a.itertuples():
        own = p[(p["season"] == r.season) & (p["fold"] == r.fold)]
        rows.append({"season": r.season, "fold": r.fold, "n_train": int(r.outcome_n_train),
                     "own_fold_fit_eligible": int(own["outcome_fit_eligible"].sum()),
                     "contact_n_train": int(r.contact_n_train),
                     "own_fold_contact": int(own["contact_fit_eligible"].sum()), "games": int(r.outcome_n_games),
                     "own_fold_games": own["game_pk"].nunique()})
    return pd.DataFrame(rows)


def missing_history() -> pd.DataFrame:
    """Share of held-out outcome rows whose batter has no denominator in the valuation role's fold, per feature;
    the forward rows against all of 2024."""
    p = research.base_tables()[0]
    denominators = {"any_pitch": "pitches", "whiff": "swings", "chase": "outside", "xwobacon": "con_n"}

    def shares(test: pd.DataFrame, train: pd.DataFrame) -> dict:
        totals = batter.feature_events(train).groupby("batter")[batter.STAT_COLUMNS].sum()
        return {name: 1 - test["batter"].map(totals[col]).fillna(0).gt(0).mean() for name, col in denominators.items()}

    rows = []
    for season in config.SEASONS:
        for k in range(config.N_FOLDS):
            role = arsenal.fold_roles(k)[1]
            test = p[(p["season"] == season) & (p["fold"] == k) & p["scorable"] & p["outcome_fit_eligible"]]
            rows.append({"sample": f"{season} scoring fold {k} (role fold {role})", "rows": len(test),
                         **shares(test, p[(p["season"] == season) & (p["fold"] == role)])})
    test = p[(p["season"] == config.SEASONS[1]) & p["outcome_fit_eligible"]]
    rows.append({"sample": "forward 2024->2025", "rows": len(test), **shares(test, p[p["season"] == config.SEASONS[0]])})
    return pd.DataFrame(rows)


# --- Gate ---

def stage_gate():
    t0 = time.perf_counter()
    res = {"checks": checks()}
    if not res["checks"]["exact"].all():
        raise RuntimeError("the disabled-extension check failed; fix before applying the gate")
    res["cmp"] = {(kind, *pair): research.prediction_compare(*pair, kind) for kind in KINDS for pair in COMPARISONS}
    res["gate"] = {key: research.gate(c) for key, c in res["cmp"].items()}
    res["chosen"] = {}
    for kind in KINDS:
        p = {pair: res["gate"][(kind, *pair)]["passes"] for pair in COMPARISONS}
        choice = research.nested_choice(p["P0", "M3a"], p["P0", "M3b"], p["M3a", "M3b"])
        res["chosen"][kind] = {"complex": "M3b", "simple": "M3a", "base": None}[choice]
    res["alphas"] = {m: pd.DataFrame(research.load(m, "manifest")["alphas"]) for m in ("M3a", "M3b")}
    res["forward_alphas"] = {m: research.load(m, "manifest")["forward_alphas"] for m in ("M3a", "M3b")}
    res["role_sizes"] = role_sizes()
    res["missing"] = missing_history()
    res["abnormal"] = abnormal_log()
    res["run"] = json.loads(RUN_JSON.read_text())
    _save("s6", res)
    S6_REPORT.write_text(s6_report(res))
    print(f"stage s6 gate done ({time.perf_counter() - t0:.0f} s); chosen: {res['chosen']}", flush=True)
    return res


# --- Report ---

S6_REPORT = config.REPORTS_DIR / "research_s6.md"
UNIT = {"outcome": "nats per fit-eligible pitch", "contact": "nats per contact-fit-eligible ball in play"}


def _gate_table(res) -> pd.DataFrame:
    rows = []
    for key, c in res["cmp"].items():
        kind, base, version = key
        g, s = res["gate"][key], c["season"].set_index("season")
        rows.append({"component": kind, "comparison": f"{version} vs {base}", "gain_2024": s.loc[2024, "gain"],
                     "gain_2025": s.loc[2025, "gain"], "both_seasons": g["both_seasons"],
                     "max_cal": float(s["cal_cand"].max()), "calibration": g["calibration"],
                     "forward_gain": c["forward"]["gain"], "forward": g["forward"], "passes": g["passes"]})
    return pd.DataFrame(rows)


def _loss_table(res) -> pd.DataFrame:
    rows = []
    for (kind, b, v), c in res["cmp"].items():
        for r in c["season"].itertuples():
            rows.append({"component": kind, "base": b, "version": v, "sample": str(r.season), "n_matched": r.n,
                         "base_loss": r.base, "version_loss": r.cand, "gain": r.gain, "max_cal_base": r.cal_base,
                         "max_cal_version": r.cal_cand})
        f = c["forward"]
        rows.append({"component": kind, "base": b, "version": v, "sample": "forward 2024->2025", "n_matched": f["n"],
                     "base_loss": f["base"], "version_loss": f["cand"], "gain": f["gain"],
                     "max_cal_version": f["cal_cand"]})
    out = pd.DataFrame(rows)
    out["max_cal_base"] = out["max_cal_base"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    return out


def _fold_table(res) -> pd.DataFrame:
    return pd.concat([c["fold"].rename(columns={"base": "base_loss", "cand": "version_loss"})
                      .assign(component=k, base=b, version=v) for (k, b, v), c in res["cmp"].items()],
                     ignore_index=True)[["component", "base", "version", "season", "fold", "n", "base_loss",
                                         "version_loss", "gain"]]


def _penalty_table(res) -> pd.DataFrame:
    rows = []
    for m, a in res["alphas"].items():
        for kind in KINDS:
            for r in a.to_dict("records"):
                rows.append({"variant": m, "component": kind, "role": f"{r['season']} fold {r['fold']}",
                             "lambda": f"{r[f'{kind}_lambda']:g}", **{f"cv_{lam:g}": r[f"{kind}_cv_{lam:g}"]
                                                                      for lam in batter.LAMBDAS}})
        for kind in KINDS:
            f = res["forward_alphas"][m]
            rows.append({"variant": m, "component": kind, "role": "forward (all of 2024)",
                         "lambda": f"{f[f'{kind}_lambda']:g}", **{f"cv_{lam:g}": f[f"{kind}_cv_{lam:g}"]
                                                                  for lam in batter.LAMBDAS}})
    return pd.DataFrame(rows)


def _edge_note(res) -> str:
    t = _penalty_table(res)
    edge = t[t["lambda"].isin([f"{max(batter.LAMBDAS):g}", f"{min(batter.LAMBDAS):g}"])]
    if edge.empty:
        return "No selected penalty is at either end of the grid."
    return ("Selected penalties at an end of the grid (reported; the grid is not extended): "
            + "; ".join(f"{r.variant} {r.component} {r.role} ({r._4})" for r in edge.itertuples(index=False)) + ".")


def _hyper_table(res) -> pd.DataFrame:
    a = res["alphas"]["M3b"]
    cols = ["kappa_whiff", "kappa_chase", "mu_whiff", "mu_chase", "mu_xwobacon", "sigma2_xwobacon", "tau2_xwobacon",
            "mean_whiff", "sd_whiff", "mean_chase", "sd_chase", "mean_xwobacon", "sd_xwobacon"]
    out = a[["season", "fold"] + [f"outcome_{c}" for c in cols]].rename(columns=lambda c: c.replace("outcome_", ""))
    f = res["forward_alphas"]["M3b"]
    return pd.concat([out, pd.DataFrame([{"season": "forward", "fold": "", **{c: f[f"outcome_{c}"] for c in cols}}])],
                     ignore_index=True)


def _coef_table(res) -> pd.DataFrame:
    rows = []
    for m, a in res["alphas"].items():
        for kind in KINDS:
            rows.append({"variant": m, "component": kind, "coef_norm_min": a[f"{kind}_coef_norm"].min(),
                         "coef_norm_max": a[f"{kind}_coef_norm"].max(),
                         **{f"shift_rms_{n}": a[f"{kind}_shift_rms_{n}"].mean() for n in batter.FEATURES}})
    return pd.DataFrame(rows)


def s6_report(res) -> str:
    ch = res["checks"]
    chosen = res["chosen"]
    lines = [f"- **{kind}:** " + (f"{chosen[kind]}'s component is chosen." if chosen[kind] else
                                  "neither variant is chosen; P0's component stays.") for kind in KINDS]
    if any(chosen.values()):
        next_step = ("The next steps are the runtime benchmark of per-PA solves and, after that, the solver and scoring "
                     "extension (S6 §7). Neither is started here: this report is prediction-only.")
    else:
        next_step = "No component is adopted, so S6 stops with a negative predictive result and S7 is dropped (S6 §8)."
    out = ["# Research program S6, step 1: batter-index prediction gate (M3a, M3b)", "",
           "Generated by `python3.12 -m gamecalling.research_s6 gate` (versions fitted by `... run`). Specification: "
           "`research_notes/research_program/S6_batter_index.md` (revision 2); model code `gamecalling/batter.py`. "
           "Built on P0 alone (not on M1c). Prediction-only: no policy, solver, scoring or clustering. Descriptive; "
           "2024-2025 are previously examined seasons (exploratory reuse). One run of the gate. A passing gate shows "
           "better probabilities, not better comparisons between pitch choices.", "",
           "## Checks", "", _md(ch.assign(exact=ch["exact"].map({True: "yes", False: "NO"}))), "",
           "Outer roles train on their own fold only (`batter.role_code` refuses any other set of folds); rows and "
           "games from the fits against the pitch table:", "", _md(res["role_sizes"]), "",
           "Unit tests (`tests/test_batter.py`): event definitions, shrinkage and each degenerate branch by hand, "
           "standardization, leave-one-game-out features and offsets by hand, orthonormal contrasts, the design's "
           "coefficient counts at full rank, the gradient against finite differences, the zero-feature identity to "
           "1e-14, seeds and role codes, unit weights only, inner-validation isolation at a fixed penalty, "
           "leave-one-game-out invariance with parents held fixed, and outer-fold isolation.", "",
           "## Prediction gate (component by component; delta 0.0005; full precision)", "",
           "Gain = base log loss minus version log loss on matched held-out observations (positive = the version "
           f"predicts better). Outcome: {UNIT['outcome']}. Contact: {UNIT['contact']}.", "",
           _md(_gate_table(res), "{:.6f}"), "", "**Nested rule, component by component:**", "", *lines, "", next_step, "",
           "Matched held-out and forward log loss:", "", _md(_loss_table(res), "{:.6f}"), "",
           "Per fold (reported, not gated):", "", _md(_fold_table(res), "{:.6f}"), "",
           "## Penalties (reported, not gated)", "",
           "Selected lambda and the pooled inner-validation loss for each lambda, by role. Each role selects "
           "separately, per variant and per component.", "", _edge_note(res), "",
           _md(_penalty_table(res), "{:.6f}"), "",
           "## Shrinkage hyperparameters and standardization (reported)", "",
           "From M3b's fits; the feature preprocessing does not depend on the variant.", "",
           _md(_hyper_table(res), "{:.4f}"), "",
           "## Coefficient summaries (reported)", "",
           "`coef_norm`: range of the coefficient norm over the six role folds. `shift_rms`: mean over roles of the "
           "root-mean-square centered log-probability shift per feature SD, for the average family.", "",
           _md(_coef_table(res), "{:.4f}"), "",
           "## Batters without history (reported)", "",
           "Share of held-out rows whose batter has no denominator for that feature in the role fold (they get the "
           "league value).", "", _md(res["missing"], "{:.4f}"), "",
           "## Convergence (S6 §4.3, revision 2)", "", *_abnormal_lines(res), "",
           "## Runtime and math threads", "", *_runtime_lines(res), ""]
    return "\n".join(out)


def _abnormal_lines(res) -> list:
    a = res["abnormal"]
    head = ("Every accepted fit has finite values and a recomputed max|gradient| at most 1e-7, with ordinary optimizer "
            "success or the specific L-BFGS-B ABNORMAL line-search termination. Any other outcome would have stopped "
            "the run.")
    if a.empty:
        return [head, "", "No fit ended with an accepted abnormal termination."]
    b = a.assign(max_grad=a["max_grad"].map("{:.2e}".format), **{"lambda": a["lambda"].map("{:g}".format)})
    return [head, "", f"Accepted abnormal terminations: {len(a)}, all at lambda = {a['lambda'].max():g}. The training-only "
            "diagnostic, run with multi-threaded math and the full path on each final fit, logged 3: which fits end this "
            "way can vary with the numerical environment.", "", _md(b)]


def _runtime_lines(res) -> list:
    r = res["run"]
    jobs = pd.DataFrame(r["jobs"])
    libs = sorted({f"{k}: {v}" for e in r["jobs"] for k, v in e["libraries"].items()})
    t = jobs[["model", "season", "fold", "seconds", "cpu_over_wall"]].fillna("")
    return [f"Full step-1 run: {r['wall_seconds_total'] / 60:.1f} min wall-clock at {r['workers']} workers "
            f"(fits {r['wall_seconds_fits'] / 60:.1f} min), against the 60-minute budget. Includes M3_off's full scored "
            "run and P0's check of the prediction-only path.", "",
            f"Math threads verified in the parent and in every job: {', '.join(libs)}; largest measured CPU time over "
            f"wall time on a matrix product: {jobs['cpu_over_wall'].max():.2f}.", "", _md(t, "{:.2f}")]


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        {"gate": stage_gate}[stage]()
