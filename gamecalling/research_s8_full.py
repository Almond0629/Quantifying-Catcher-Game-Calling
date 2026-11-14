"""Research program S8 stages 4-5: C1's runtime benchmark, full scored run and descriptive validation.

Specification: research_notes/research_program/S8_combined_model.md (revision 1), §6 stages 4-5 and §8. C1 was cleared
for full scoring by criterion 2 (§12) and the user's review of it (2026-11-08). Stages, in order:
- benchmark: the plan's one season-fold benchmark (2024, scoring fold 0) and its extrapolation against the budget
  (60 min per full run at 8 workers, 120 for the halves). Over budget stops the work (plan: one optimization pass, then
  the user decides).
- guard: hash the stored outputs this stage must not change.
- run: C1's full scored run into data/models/C1_full/ (per-action rows in data/models/C1/runs/full/).
- check: held-out and forward predictions against C1's stored gate outputs and C1's outcome against M3a's full run,
  exactly; the scored roles' fit hashes against those criterion 2 checked. Any difference stops.
- describe: per-action reconstruction; recommendations; decision relevance and D, A and B against P0 and M4; coverage.
- halves: the independent-halves refit (protocol §3b).
- s1runs: S1 (c), (d) and (e) as full reruns under C1.
- validate: the locked protocol's analyses for C1 with V0's frozen halves and groups (S8 §8), S1 (a)-(e),
  interpretation and the drop-one table.
- s1a_splits (added 2026-11-14, after the validation): S1 (a)'s random context split repeated with whole plate
  appearances and whole games as the unit, stored separately; the validation's stored results are not rewritten.
- report: reports/research_s8.md.
Descriptive; 2024-2025 are previously examined seasons (exploratory reuse). Reliability never chooses a model or score.
S8_FULL_SMOKE=<dir> runs every stage on one game in five under that directory (with S6_FULL_SMOKE=<dir> for the P0 and
M4 references); the smoke results mean nothing for decisions.

Importing research_s6 first limits every process to one math thread before numpy initializes.
"""

from gamecalling import research_s6  # noqa: I001 (first: sets the thread limit)

import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gamecalling import arsenal, batter_scoring as bs, config, models, payoff, research
from gamecalling import research_checks as rc
from gamecalling import research_s6_benchmark as rb
from gamecalling import research_s6_full as sf
from gamecalling import research_s8 as s8
from gamecalling import validation_pilot as vp
from gamecalling.phase1 import _md
from gamecalling.research_s6 import PRED, verify_threads
from gamecalling.research_s7 import attach_raw

SMOKE = os.environ.get("S8_FULL_SMOKE")
WORKERS = 8
LABEL = "full"
SCORES3 = (("D", "D"), ("A", "A"), ("B", "B"))
STORED = config.DATA_DIR / "models"
BUDGET_MIN, HALVES_BUDGET_MIN = 60.0, 120.0
KEYS = research.KEYS


def ref(model_id: str) -> str:
    """Stored prediction references; the smoke test has none on its subset, so M4's smoke run stands in."""
    return "M4_full" if SMOKE and model_id in ("P0", "M4", "M1c") else model_id


# --- Specifications (module level, so they pickle into worker processes) ---

def _subset(pitches, pas):
    keep = lambda d: None if d is None else d[d["game_pk"] % 5 == 0].copy()  # noqa: E731
    return keep(pitches), keep(pas)


def c1_transform(pitches, pas):
    """C1's own transform (bb_type from the raw table); the smoke test then keeps one game in five."""
    p, a = attach_raw(pitches, pas)
    return _subset(p, a) if SMOKE else (p, a)


def c1_dugout(pitches, pas):
    return rc.drop_dugout_games(*c1_transform(pitches, pas))


def c1_re24(pitches, pas):
    return rc.re24_all_complete(*c1_transform(pitches, pas))


C1 = replace(s8.SPECS["C1"], transform=c1_transform, pipeline=bs.batter_pipeline)
S1_SPECS = {
    "S1c_C1": replace(C1, id="S1c_C1", seasons=(2025,), transform=c1_dugout,
                      description="S1 (c) under C1: dugout-called games removed from probability training, payoffs and "
                                  "evaluation (2025)", params={**C1.params, **rc.SPECS["S1c"].params}),
    "S1d_C1": replace(C1, id="S1d_C1", transform=c1_re24,
                      description="S1 (d) under C1: RE24 from all complete half-innings",
                      params={**C1.params, **rc.SPECS["S1d"].params}),
    "S1e_C1": replace(C1, id="S1e_C1", train_mask=rc.bases_empty, eval_mask=rc.bases_empty,
                      description="S1 (e) under C1: outcome, contact and policy models and batter features trained on "
                                  "bases-empty pitches; payoffs and arsenal unchanged; runner hierarchy active",
                      params={**C1.params, **rc.SPECS["S1e"].params, "batter_features": "bases-empty"}),
}

if SMOKE:
    OUT, REPORT = Path(SMOKE), Path(SMOKE) / "research_s8.md"
    research.MODELS_DIR = OUT / "models"
else:
    OUT, REPORT = config.DATA_DIR / "research", config.REPORTS_DIR / "research_s8.md"

STAGES_JSON = OUT / "s8_full_stages.json"
BENCH_JSON = OUT / "s8_benchmark_C1.json"
GUARD_JSON = OUT / "s8_full_guard.json"
CHECK_JSON = OUT / "s8_full_check.json"
DESCRIBE_PKL = OUT / "s8_full_describe.pkl"
HALVES_PKL = OUT / "s8_full_halves_C1.pkl"
VALIDATE_PKL = OUT / "s8_validate.pkl"
S1A_SPLITS_PKL = OUT / "s8_s1a_splits.pkl"
S1A_UNITS = (("pa", "pa_id", 1), ("game", "game_pk", 2))  # name, column, seed tag
EFFECTS_CSV = OUT / "s8_full_catcher_effects.csv"
MEMORY_CSV = OUT / "s8_full_memory.csv"


def _record(stage: str, t_start: float, **extra) -> None:
    done = json.loads(STAGES_JSON.read_text()) if STAGES_JSON.exists() else {}
    done[stage] = {"started": t_start, "finished": time.time(), "seconds": time.time() - t_start,
                   "parent_peak_rss_gb": bs.peak_rss_gb(), **extra}
    STAGES_JSON.write_text(json.dumps(done, indent=1, default=str))
    print(f"stage {stage}: {done[stage]['seconds']:.0f} s, parent peak {done[stage]['parent_peak_rss_gb']:.1f} GB", flush=True)


def _refuse(path: Path):
    if path.exists():
        raise FileExistsError(f"{path} exists: this stage runs once")


# --- Benchmark (plan: one season-fold x 6 before any full run) ---

def stored_forward_seconds() -> float:
    run = json.loads((config.DATA_DIR / "research" / "s8_run.json").read_text())["jobs"]
    return next(e["seconds"] for e in run if e["model"] == "C1" and e["fold"] == "forward")


def estimate(res: dict) -> dict:
    """rb.estimate's extrapolation with C1's measured forward fit: serial x 6 and the job schedule under memory."""
    fit = [res["fits"]["selection"]["seconds"], res["fits"]["valuation"]["seconds"]]
    fit_mean, score_s = float(np.mean(fit)), res["score_job"]["seconds"]
    fit_rss = max(res["fits"]["selection"]["peak_rss_gb"], res["fits"]["valuation"]["peak_rss_gb"])
    score_rss, parent_rss = res["score_job"]["peak_rss_gb"], res["peak_rss_parent_gb"]
    forward_s = stored_forward_seconds()

    def workers_for(job_rss):
        return max(1, min(WORKERS, int((rb.MEMORY_LIMIT_GB - parent_rss) // job_rss)))

    w_fit, w_score = workers_for(fit_rss), workers_for(score_rss)
    full_wall = rb.list_schedule([fit_mean] * 6, w_fit) + rb.list_schedule([score_s] * 6, w_score) + forward_s
    half_s, half_rss = res["half"]["seconds"], res["half"]["peak_rss_gb"]
    w_half = max(1, min(4, int((rb.MEMORY_LIMIT_GB - parent_rss) // half_rss)))
    halves_wall = rb.list_schedule([half_s] * 4, w_half)
    return {"serial_season_fold_seconds": sum(fit) + score_s, "serial_x6_minutes": 6 * (sum(fit) + score_s) / 60,
            "fit_job_seconds": fit_mean, "score_job_seconds": score_s, "forward_seconds_from_stored_run": forward_s,
            "workers_fit": w_fit, "workers_score": w_score, "peak_sum_fit_gb": parent_rss + w_fit * fit_rss,
            "peak_sum_score_gb": parent_rss + w_score * score_rss, "full_run_wall_minutes": full_wall / 60,
            "full_run_budget_minutes": BUDGET_MIN, "halves_workers": w_half,
            "halves_peak_sum_gb": parent_rss + w_half * half_rss, "halves_wall_minutes": halves_wall / 60,
            "halves_budget_minutes": HALVES_BUDGET_MIN,
            "within_budget": full_wall / 60 <= BUDGET_MIN and halves_wall / 60 <= HALVES_BUDGET_MIN}


def stage_benchmark(label: str = "benchmark"):
    """rb.stage_benchmark's measurements for C1 (2024, scoring fold 0, one math thread, each job in its own process):
    both role fits, the scoring job, the reuse check in both roles and one measured independent half."""
    _refuse(BENCH_JSON)
    OUT.mkdir(parents=True, exist_ok=True)
    season, k = rb.SEASON, rb.K
    t0 = time.perf_counter()
    parent = verify_threads()
    pitches, pas, support = research.prepared(C1)
    fit_role = research.role_fitter(C1)
    s_fold, v_fold = arsenal.fold_roles(k)
    eligible = arsenal.eligible_sets(support)
    units = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == k)])
    fold_rows = {f: (pitches[(pitches["season"] == season) & (pitches["fold"] == f)],
                     pas[(pas["season"] == season) & (pas["fold"] == f)]) for f in (s_fold, v_fold)}
    rows = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
    res = {"model": "C1", "season": season, "scoring_fold": k, "run_label": label, "parent_threads": parent,
           "seconds_prepare_tables": time.perf_counter() - t0}
    fits = {}
    for name, f in (("selection", s_fold), ("valuation", v_fold)):
        with rb._isolated() as ex:
            fits[name] = ex.submit(bs.fit_job, (fit_role, *fold_rows[f], season, f, C1.id, label)).result()
        print(f"C1 fit {name}: {fits[name]['seconds']:.0f} s, {fits[name]['peak_rss_gb']:.1f} GB", flush=True)
    sel_role, val_role = fits["selection"]["role"], fits["valuation"]["role"]
    sel_id, val_id = fits["selection"]["fit_id"], fits["valuation"]["fit_id"]
    with rb._isolated() as ex:
        score = ex.submit(bs.score_job, (rows, season, k, sel_role, val_role, sel_id, val_id, C1.fit_policy, units,
                                         research.MODELS_DIR, C1.id, label)).result()
    print(f"C1 score job: {score['seconds']:.0f} s, {score['peak_rss_gb']:.1f} GB", flush=True)
    res["fits"] = {n: {x: v[x] for x in ("seconds", "peak_rss_gb", "fit_id", "libraries", "cpu_over_wall")}
                   for n, v in fits.items()}
    res["score_job"] = {x: score[x] for x in ("seconds", "seconds_score", "seconds_write", "peak_rss_gb", "libraries",
                                              "cpu_over_wall")}
    res["solve_log"] = score["solve_log"]
    res["scored_pitches"] = len(score["scores"])
    res["abnormal"] = [e for r in (sel_role, val_role) for m in (r.outcome, r.contact) for e in getattr(m, "abnormal", [])]
    with rb._isolated() as ex:
        res["reuse"] = ex.submit(rb._reuse_job, (rows, sel_role, val_role, sel_id, val_id, C1.fit_policy, units)).result()
    print(f"C1 reuse: {res['reuse']['seconds']:.0f} s", flush=True)
    halves = vp._load("v0")["halves"]
    games = set(pitches.loc[pitches["season"] == season, "game_pk"].unique())
    g = np.sort(halves.index[(halves == 0).to_numpy() & halves.index.isin(games)])
    with rb._isolated() as ex:
        res["half"] = ex.submit(rb._half_measure, (C1, season, 0, g, f"{label}_half_{season}_0")).result()
    print(f"C1 half: {res['half']['seconds']:.0f} s, {res['half']['peak_rss_gb']:.1f} GB", flush=True)
    res["peak_rss_parent_gb"] = bs.peak_rss_gb()
    res["estimate"] = estimate(res)
    res["code"] = research.code_state()
    BENCH_JSON.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["estimate"], indent=1), flush=True)
    if not res["estimate"]["within_budget"]:
        raise RuntimeError("the extrapolated full run or halves exceed the budget: stop and report (plan)")


# --- Guard ---

def guarded_files() -> list:
    files = [config.DATA_DIR / "phase1_scores.parquet"]
    for m in ("P0_repro", "M1c", "M3a", "M4", "C1", "C1_off", "M3a_full", "M4_full"):
        files += sorted(p for p in (STORED / m).iterdir() if p.is_file())
    files += [config.DATA_DIR / "research" / f for f in ("s8_run.json", "s8.pkl", "s8_simulation_16M_C1.csv",
                                                           "s8_simulation_16M_C1.json", "s6_full_p0_refs.json")]
    files += sorted((vp.OUT).glob("*.pkl")) + sorted(vp.OUT.glob("*.json"))
    return files


def stage_guard():
    OUT.mkdir(parents=True, exist_ok=True)
    _refuse(GUARD_JSON)
    GUARD_JSON.write_text(json.dumps(sf.stored_hashes(guarded_files()), indent=1))
    print(f"guarded {len(json.loads(GUARD_JSON.read_text()))} files", flush=True)


def guard_changes() -> list:
    return sf.changed_hashes(json.loads(GUARD_JSON.read_text()), sf.stored_hashes(guarded_files()))


# --- Full run and its exact checks ---

def stage_run():
    t = time.time()
    _refuse(research.MODELS_DIR / "C1_full")
    man = research.run_model(C1, out_name="C1_full", run_label=LABEL, workers=WORKERS)
    _record("run_C1", t, workers=WORKERS, scored=len(research.load("C1_full", "scores")), timings=man["timings"],
            abnormal=len(man["abnormal_log"]))


def fit_hash(fit_id: str) -> str:
    return fit_id.split("/")[-1]


def stage_check():
    t = time.time()
    rows = []
    if SMOKE:
        rows.append({"check": "smoke: references are fitted on all games, not compared", "identical": True,
                     "detail": "not compared"})
    else:
        for what in PRED:
            r = sf.exact_compare(research.load("C1_full", what), pd.read_parquet(STORED / "C1" / f"{what}.parquet"))
            rows.append({"check": f"C1 full run: {what} vs. C1's stored prediction gate", **r})
        for what in ("heldout_outcome", "forward_outcome"):
            r = sf.exact_compare(research.load("C1_full", what), research.load("M3a_full", what))
            rows.append({"check": f"test 1 (full-run form): C1's {what} vs. M3a's full run", **r})
        crit = json.loads((config.DATA_DIR / "research" / "s8_simulation_16M_C1.json").read_text())["fit_ids"]
        full = research.load("C1_full", "manifest")["fit_ids"]
        for key, fid in crit.items():
            same = fit_hash(full[key]) == fit_hash(fid)
            rows.append({"check": f"role {key}: fit hash vs. the role criterion 2 checked", "identical": same,
                         "rows": 1, "detail": f"{fit_hash(full[key])} vs. {fit_hash(fid)}"})
    CHECK_JSON.write_text(json.dumps(rows, indent=1))
    print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)
    _record("check", t)
    if not all(r["identical"] for r in rows):
        raise RuntimeError("a full-run check differs: stop and diagnose")


# --- Descriptive comparison with P0 and M4 ---

def versions():
    scores = {"P0": sf.p0_scores(), "M4": research.load("M4_full", "scores"), "C1": research.load("C1_full", "scores")}
    refs = {"P0": json.loads(sf.P0_REFS_JSON.read_text()), "M4": research.load("M4_full", "manifest")["action_values"],
            "C1": research.load("C1_full", "manifest")["action_values"]}
    return scores, refs


def stage_describe():
    t = time.time()
    _refuse(DESCRIBE_PKL)
    scores, refs = versions()
    d = {"reconstruct": [{"version": "C1", **sf.reconstruct_check(bs.read_action_values(refs["C1"]), scores["C1"])}]}
    print(pd.DataFrame(d["reconstruct"]).to_string(), flush=True)
    r = d["reconstruct"][0]
    if r["unmatched"] or r["rec_differs"] or max(r["max_Q_actual_diff"], r["max_D_diff"]) > 1e-12:
        raise RuntimeError(f"C1's per-action rows do not give back its scores: {r}")
    d["recommend"] = pd.concat([s.groupby("season").agg(pitches=("agree", "size"), agree=("agree", "mean"),
                                                        mean_D=("D", "mean"), mean_A=("A", "mean"), mean_B=("B", "mean"))
                                .reset_index().assign(version=m) for m, s in scores.items()], ignore_index=True)
    pairs = (("P0", "C1"), ("M4", "C1"))
    d["relevance"] = pd.concat([bs.decision_relevance_rows(refs[b], refs[c]).assign(base=b, version=c) for b, c in pairs],
                               ignore_index=True)
    cache, label = {}, "s8_full"
    d["rank"] = {(b, c): research.rank_compare(scores[b], scores[b], scores[c], SCORES3, label, cache, b, c)
                 for b, c in pairs}
    d["common_equal"] = d["rank"]["P0", "C1"]["common"] == d["rank"]["M4", "C1"]["common"]
    d["coverage"] = rc._coverage_diff("C1_full", config.SEASONS)
    key = lambda s: s[bs.PITCH_KEYS].sort_values(bs.PITCH_KEYS).reset_index(drop=True)  # noqa: E731
    d["same_scored"] = {m: bool(key(scores["C1"]).equals(key(scores[m]))) for m in ("P0", "M4")}
    common = d["rank"]["P0", "C1"]["common"]
    pd.concat([pd.DataFrame({"season": season, "version": v, "score": sc,
                             "saved": cache[label, season, v, sc]["saved"].reindex(cs)}).rename_axis("catcher").reset_index()
               for season, cs in common.items() for v in scores for sc, _ in SCORES3
               if (label, season, v, sc) in cache], ignore_index=True).to_csv(EFFECTS_CSV, index=False)
    with open(DESCRIBE_PKL, "wb") as f:
        pickle.dump(d, f)
    _record("describe", t)


# --- Independent halves and the S1 reruns ---

def stage_halves():
    t = time.time()
    _refuse(HALVES_PKL)
    runs = research.independent_halves(C1, WORKERS, job=sf._half_job)
    with open(HALVES_PKL, "wb") as f:
        pickle.dump({"runs": runs}, f)
    _record("halves_C1", t, jobs=[r["meta"] for r in runs.values()])


def stage_s1runs(ids=None):
    for model_id in ids or S1_SPECS:
        t = time.time()
        _refuse(research.MODELS_DIR / model_id)
        man = research.run_model(S1_SPECS[model_id], run_label=LABEL, workers=WORKERS)
        _record(f"run_{model_id}", t, workers=WORKERS, scored=len(research.load(model_id, "scores")),
                timings=man["timings"], abnormal=len(man["abnormal_log"]))


# --- Validation (S8 §8) ---

def c1_scores() -> pd.DataFrame:
    """C1's scores with at_bat_number, as vp.scores_table() builds P0's, injected so vp's analyses run on C1."""
    s = research.load("C1_full", "scores")
    ab = pd.read_parquet(config.PITCHES_PARQUET, columns=["pa_id", "pitch_number", "at_bat_number"])
    s = s.merge(ab, on=["pa_id", "pitch_number"], how="left", validate="one_to_one")
    vp._DATA["scores"] = s
    return s


def prediction_tables() -> dict:
    """§2: matched held-out log loss by season (both components, P0 alongside), and the forward check with seen and
    previously unseen pitcher cells separately."""
    out = {"heldout": {k: research.prediction_compare(ref("P0"), "C1_full", k) for k in ("outcome", "contact")}}
    rows = []
    for kind in ("outcome", "contact"):
        fb, fc = research.load(ref("P0"), f"forward_{kind}"), research.load("C1_full", f"forward_{kind}")
        if fb is None or fc is None:
            continue
        m = research.matched(fb, fc)
        seen = m.merge(fc[KEYS + ["seen_cell"]], on=KEYS, how="left", validate="one_to_one")["seen_cell"].to_numpy(bool)
        seen_p0 = m.merge(fb[KEYS + ["seen_cell"]], on=KEYS, how="left", validate="one_to_one")["seen_cell"].to_numpy(bool)
        for name, g in (("all", m), ("seen cells", m[seen]), ("unseen cells", m[~seen])):
            rows.append({"model": kind, "sample": name, "n": len(g), "P0": -g["lp_base"].mean(),
                         "C1": -g["lp_cand"].mean(), "gain": (g["lp_cand"] - g["lp_base"]).mean(),
                         "seen_flags_equal": bool(np.array_equal(seen, seen_p0))})
        rows[-3]["max_cal_P0"] = research.max_calibration(fb, m[KEYS])
        rows[-3]["max_cal_C1"] = research.max_calibration(fc, m[KEYS])
    out["forward"] = pd.DataFrame(rows)
    return out


def s1_checks(C: pd.DataFrame, split_random: dict, sens: pd.DataFrame) -> dict:
    """research_checks.stage_s1 with C1 as the reference instead of P0."""
    pitches, pas, _ = research.base_tables()
    res, cache = {}, {}
    empty = rc.bases_empty(C)
    res["a"] = research.rank_compare(C, C[empty], C[~empty], rc.SCORES2, "a", cache, "bases_empty", "runners_on")
    parts = []
    for season in sorted(set(C["season"])):
        s = C[C["season"] == season]
        rng = np.random.default_rng([rc.CONTEXT_SEED, season])
        parts.append(pd.Series(rng.random(len(s)) < rc.bases_empty(s).mean(), index=s.index))
    rnd = pd.concat(parts).reindex(C.index)
    res["a_context_random"] = research.rank_compare(C, C[rnd], C[~rnd], rc.SCORES2, "a_ctx", cache,
                                                    "random_share_empty", "random_rest")
    res["a_shares"] = C.assign(e=empty).groupby("season")["e"].mean()
    res["a_context_halves"] = pd.DataFrame([{"season": se, **r} for se, d in split_random.items()
                                            for r in d["compare"] if r["model"] == "random" and r["score"] in "DA"])
    res["r4"] = sens
    flags = rc.addendum_flags(pitches, pas)
    pa_flag = C["pa_id"].map(flags["any"]).fillna(False).astype(bool)
    res["b"] = research.rank_compare(C, C, C[~pa_flag], rc.SCORES2, "b", cache, "C1", "restricted")
    res["b_removed"] = C[pa_flag].groupby("season").size()

    pred = lambda cand: {k: research.prediction_compare("C1_full", cand, k) for k in ("outcome", "contact")}  # noqa: E731
    c = research.load("S1c_C1", "scores")
    C25 = C[C["season"] == 2025]
    res["c_pred"] = pred("S1c_C1")
    res["c"] = research.rank_compare(C25, C25, c, rc.SCORES2, "c", cache, "C1", "S1c_C1")
    res["c_vs_R3"] = research.rank_compare(C25, C25[~C25["dugout_called"].astype(bool)], c, rc.SCORES2, "c_r3", cache,
                                           "R3_sample", "S1c_C1")
    res["c_coverage"] = rc._coverage_diff("S1c_C1", (2025,))
    d = research.load("S1d_C1", "scores")
    res["d_pred"] = pred("S1d_C1")
    res["d"] = research.rank_compare(C, C, d, rc.SCORES2 + (("D_1p", "D_1p"),), "d", cache, "C1", "S1d_C1")
    re24 = []
    for season in sorted(set(C["season"])):
        for f in range(config.N_FOLDS):
            a = pas[(pas["season"] == season) & (pas["fold"] == f)]
            r0, r1 = payoff.re24_table(a), payoff.re24_table(a.assign(re24_eligible=a["half_inning_complete"]))
            re24.append({"season": season, "fold": f, "max_abs_change": float(np.abs(r1 - r0).max())})
    res["d_re24"] = pd.DataFrame(re24)
    e = research.load("S1e_C1", "scores")
    res["e_pred"] = pred("S1e_C1")
    res["e_vs_C1"] = research.rank_compare(C, C, e[rc.bases_empty(e)], rc.SCORES2, "e", cache, "C1", "S1e_C1")
    res["e_vs_R4"] = research.rank_compare(C, C[empty], e[rc.bases_empty(e)], rc.SCORES2, "e_r4", cache, "R4_sample",
                                           "S1e_C1")
    res["e_alphas"] = pd.DataFrame(research.load("S1e_C1", "manifest")["alphas"])
    return res


def unit_split(C: pd.DataFrame, column: str, tag: int) -> pd.Series:
    """S1 (a)'s random context split with whole units (plate appearances or games) on one side: each unit goes
    to the first part with the season's bases-empty pitch share as probability. rng [CONTEXT_SEED, season, tag]."""
    parts = []
    for season in sorted(set(C["season"])):
        s = C[C["season"] == season]
        units = np.sort(s[column].unique())
        rng = np.random.default_rng([rc.CONTEXT_SEED, season, tag])
        side = pd.Series(rng.random(len(units)) < rc.bases_empty(s).mean(), index=units)
        parts.append(pd.Series(s[column].map(side).to_numpy(bool), index=s.index))
    return pd.concat(parts).reindex(C.index)


def stage_s1a_splits():
    """Added after the validation (2026-11-14): the pitch-level random split shares plate appearances and games
    between its parts, while bases-empty vs. runners-on mostly separates whole plate appearances."""
    t = time.time()
    _refuse(S1A_SPLITS_PKL)
    C = c1_scores()
    res, cache = {"shares": {}}, {}
    for name, column, tag in S1A_UNITS:
        m = unit_split(C, column, tag)
        res[name] = research.rank_compare(C, C[m], C[~m], rc.SCORES2, f"a_ctx_{name}", cache,
                                          f"random_{name}_share_empty", f"random_{name}_rest")
        res["shares"][name] = C.assign(m=m).groupby("season")["m"].mean()
    res["seconds"] = time.time() - t
    with open(S1A_SPLITS_PKL, "wb") as f:
        pickle.dump(res, f)
    _record("s1a_splits", t)


def pitch_examples(sets: dict) -> tuple:
    """vp.pitch_examples' frozen selection (same rule and seeds) on C1's scores. Q for every eligible action comes from
    C1's per-action rows (valuation role); pi_ref is refitted from the valuation fold (C1's policy is P0's), and must
    give back the stored V_value; Q must give back the stored D."""
    s = vp.scores_table()
    season = vp.PITCH_EX_SEASON
    ids = sets[season]["qual"] & sets[season]["main"]
    d = s[(s["season"] == season) & s["catcher"].isin(ids)]
    qD, qA = d["D"].quantile([0.05, 0.5, 0.95]).to_numpy(), d["A"].quantile([0.05, 0.5, 0.95]).to_numpy()
    cats = {"agree-costly": (d["D"] > qD[2]) & (d["A"] > qA[2]),
            "agree-good": (d["D"] < qD[0]) & (d["A"] < qA[0]),
            "disagree": ((d["D"] > qD[2]) & (d["A"] < qA[1])) | ((d["A"] > qA[2]) & (d["D"] < qD[1]))}
    key = ["game_pk", "at_bat_number", "pitch_number"]
    picked = []
    for ci, (cat, m) in enumerate(cats.items()):
        cand = d[m].sort_values(key)
        rng = np.random.default_rng([vp.SEED, 8, ci])
        idx = np.arange(len(cand)) if len(cand) <= vp.N_PITCH_EX else np.sort(rng.choice(len(cand), vp.N_PITCH_EX, False))
        picked.append(cand.iloc[idx].assign(category=cat, candidates=len(cand)))
    ex = pd.concat(picked, ignore_index=True)
    refs = [r for r in research.load("C1_full", "manifest")["action_values"] if r["season"] == season]
    av = bs.read_action_values(refs)
    pitches, _, support = research.prepared(C1)
    eligible = arsenal.eligible_sets(support)
    rows, units_cache, policy_cache = [], {}, {}
    for _, r in ex.iterrows():
        k = int(r["fold"])
        if k not in units_cache:
            units_cache[k] = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == k)])
            v = arsenal.fold_roles(k)[1]
            train = pitches[(pitches["season"] == season) & (pitches["fold"] == v)]
            policy_cache[k] = C1.fit_policy(models.policy_training(train), units_cache[k])
        units, policy = units_cache[k], policy_cache[k]
        u = int(units.table.set_index(["pitcher", "stand"]).loc[(r["pitcher"], r["stand"]), "u"])
        c = int(r["balls"]) * 3 + int(r["strikes"])
        acts = [a for a in units.actions[u] if a is not None]
        q_by = av[(av["pa_id"] == r["pa_id"]) & (av["pitch_number"] == r["pitch_number"])].set_index("action")["Q_value"]
        q = np.array([q_by[a] for a in acts])
        pi = policy.probs[u, c, :len(acts)]
        rows.append({"category": r["category"], "candidates": r["candidates"], "game_pk": r["game_pk"],
                     "at_bat_number": r["at_bat_number"], "pitch_number": r["pitch_number"], "pitcher": r["pitcher"],
                     "catcher": r["catcher"], "batter": r["batter"], "stand": r["stand"],
                     "count": f"{int(r['balls'])}-{int(r['strikes'])}", "base_out": int(r["base_out"]),
                     "actual": r["action"], "recommended": r["rec_action"],
                     "eligible (pi_ref, Q x100)": "; ".join(f"{a} ({p:.2f}, {100 * v:.2f})" for a, p, v in zip(acts, pi, q)),
                     "V_x100": 100 * r["V_value"], "D_x100": 100 * r["D"], "A_x100": 100 * r["A"], "B_x100": 100 * r["B"],
                     "D_recomputed_diff": abs(q_by[r["action"]] - q_by[r["rec_action"]] - r["D"]),
                     "V_recomputed_diff": abs(float(pi @ q) - r["V_value"])})
    out = pd.DataFrame(rows)
    if not SMOKE and max(out["D_recomputed_diff"].max(), out["V_recomputed_diff"].max()) > 1e-12:
        raise RuntimeError("the pitch examples do not give back the stored D or V: stop and diagnose")
    return out, {"D_q": qD, "A_q": qA, "counts": {k: int(v.sum()) for k, v in cats.items()}}


def stage_validate():
    t = time.time()
    _refuse(VALIDATE_PKL)
    vp._check_manifest()
    v0 = vp._load("v0")
    C = c1_scores()
    res = {"prediction": prediction_tables()}
    res["split_random"] = vp.within_split(C, v0["halves"], "split of existing scores (random halves)")
    res["split_calendar"] = vp.within_split(C, v0["calendar"], "split of existing scores (calendar halves; secondary)")
    with open(HALVES_PKL, "rb") as f:
        runs = pickle.load(f)["runs"]
    res["refit"] = vp.within_refit_compare(runs)
    print(f"reliability done ({time.time() - t:.0f} s)", flush=True)
    fits, sets = vp.season_fits(), vp.season_sets()
    res["sets"] = {k: {"qual": len(v["qual"]), "main": len(v["main"])} for k, v in sets.items()}
    res["repeatability"] = vp.repeatability(fits, sets, v0["movers"])
    res["sensitivity"] = vp.sensitivity(fits, sets)
    res["team_staff"] = vp.team_staff(fits, sets, v0["primary"])
    res["season_sd"] = pd.DataFrame([{"season": se, "score": sc, "model": "fixed" if fx else "random", "sd": f["sd"],
                                      "zero": f["zero"], "converged": f["converged"]}
                                     for se, ff in fits.items() for (sc, fx), f in ff.items()])
    print(f"repeatability, sensitivity, team and staff done ({time.time() - t:.0f} s)", flush=True)
    res["s1"] = s1_checks(C, res["split_random"], res["sensitivity"]["table"])
    print(f"S1 checks done ({time.time() - t:.0f} s)", flush=True)
    res["catchers"] = vp.catcher_examples(fits, sets)
    res["pitches"], res["pitch_meta"] = pitch_examples(sets)
    res["drop_one"] = {
        "M1's runner levels (C1 vs. M4, contact)": research.prediction_compare(ref("M4"), "C1_full", "contact"),
        "M4's batter features in contact (C1 vs. M1c, contact)": research.prediction_compare(ref("M1c"), "C1_full",
                                                                                              "contact"),
        "M3a's outcome (C1 vs. P0, outcome)": research.prediction_compare(ref("P0"), "C1_full", "outcome")}
    res["seconds"] = time.time() - t
    with open(VALIDATE_PKL, "wb") as f:
        pickle.dump(res, f)
    _record("validate", t)


# --- Report ---

def _side(c1_rows, p0_rows, keys=("score", "model")) -> pd.DataFrame:
    """C1's comparison rows with P0's stored ones alongside (Pearson and Spearman)."""
    a, b = pd.DataFrame(c1_rows), pd.DataFrame(p0_rows)
    cols = [c for c in ("n", "pearson", "spearman", "median_abs_diff") if c in a]
    m = a[list(keys) + cols].merge(b[list(keys) + [c for c in ("n", "pearson", "spearman") if c in b]], on=list(keys),
                                   how="left", suffixes=("", "_P0"))
    for c in ("pearson_P0", "spearman_P0"):
        if c in m:
            m[c] = m[c].map(vp._fmt)
    return m


def _season_side(c1: dict, p0: dict, key="compare") -> pd.DataFrame:
    parts = []
    for season in c1:
        parts.append(_side(c1[season][key], p0[season][key]).assign(season=season))
    out = pd.concat(parts, ignore_index=True)
    return out[["season"] + [c for c in out if c != "season"]]


def _loss_rows(cmp: dict, base: str) -> pd.DataFrame:
    rows = []
    for r in cmp["season"].itertuples():
        rows.append({"sample": str(r.season), "n_matched": r.n, base: r.base, "C1": r.cand, "gain": r.gain,
                     f"max_cal_{base}": r.cal_base, "max_cal_C1": r.cal_cand})
    if "forward" in cmp:
        f = cmp["forward"]
        rows.append({"sample": "forward 2024->2025", "n_matched": f["n"], base: f["base"], "C1": f["cand"],
                     "gain": f["gain"], "max_cal_C1": f["cal_cand"]})
    return pd.DataFrame(rows)


def _halves_tables(v, p0v2) -> tuple:
    base = pd.concat([pd.DataFrame(p0v2["refit"][s]["compare"]).assign(season=s, version="P0") for s in p0v2["refit"]],
                     ignore_index=True)
    others = []
    for m in ("M3a", "M4"):
        path = sf.halves_path(m)
        if path.exists():
            others.append(sf._halves_table(m))
    c1 = pd.concat([pd.DataFrame(v["refit"][s]["compare"]).assign(season=s, version="C1") for s in v["refit"]],
                   ignore_index=True)
    t = pd.concat([base, *others, c1], ignore_index=True)
    cols = ["version", "season", "score", "n", "pearson", "spearman", "median_abs_diff", "sd_h0", "sd_h1", "converged"]
    rnd = t[t["model"] == "random"][cols].sort_values(["score", "season"], kind="stable")
    fx = t[t["model"] == "fixed"][["version", "season", "score", "n", "pearson", "spearman"]].sort_values(
        ["score", "season"], kind="stable")
    return rnd, fx


def _ev_value(d: dict, sc: str) -> str:
    return " / ".join(vp._fmt(d[se].loc[(sc, "random"), "pearson"]) for se in vp.SEASONS)


def evidence_table(v: dict, p0v2: dict) -> pd.DataFrame:
    """The validation protocol's §8 table for C1, with P0's stored values alongside."""
    def per(res):
        sp = {se: pd.DataFrame(res["split_random"][se]["compare"]).set_index(["score", "model"]) for se in vp.SEASONS}
        rf = {se: pd.DataFrame(res["refit"][se]["compare"]).set_index(["score", "model"]) for se in vp.SEASONS}
        rr = res["repeatability"]["table"].set_index(["score", "model"])
        st = res["sensitivity"]["table"].set_index(["season", "check"])
        share = res["team_staff"]["share"]
        a, b = vp.SEASONS
        out = {}
        for sc in ("D", "A"):
            out[sc] = {"split": _ev_value(sp, sc), "refit": _ev_value(rf, sc),
                       "repeat": f"{vp._fmt(rr.loc[(sc, 'random'), 'pearson'])} / {vp._fmt(rr.loc[(sc, 'fixed'), 'pearson'])}",
                       "r4": f"{vp._fmt(st.loc[(a, f'R4_{sc}'), 'pearson'])} / {vp._fmt(st.loc[(b, f'R4_{sc}'), 'pearson'])}",
                       "share": " / ".join(vp._fmt(x) for x in share[share["score"] == sc]["share"])}
        return out
    c, p = per(v), per(p0v2)
    rows = [("Construct", "Difference from an independently selected recommendation",
             "Difference from the pooled reference policy's value", "", "", "plan")]
    for key, label, src in (("split", "Within-season reliability, split of existing scores (Pearson 2024 / 2025)", "§3a"),
                            ("refit", "Within-season reliability, independent refit (Pearson 2024 / 2025)", "§3b"),
                            ("repeat", "2024-2025 repeatability (Pearson, random / fixed; exploratory reuse)", "§4"),
                            ("r4", "Robustness: bases-empty ranking sample, R4 (Pearson 2024 / 2025)", "§5b"),
                            ("share", "Team share (2024 / 2025)", "§6")):
        rows.append((label, c["D"][key], c["A"][key], p["D"][key], p["A"][key], src))
    rows += [("Recovery simulation", "P0 only (not rerun for C1)", "P0 only", "see validation_pilot.md §8", "", "phase23"),
             ("Uncertainty procedures", "none validated", "none validated", "none validated", "none validated", ""),
             ("Interpretation", "see §7", "see §7", "", "", "§7")]
    return pd.DataFrame(rows, columns=["evidence", "C1: D", "C1: A", "P0: D", "P0: A", "source"])


def memory_table(stages: dict) -> pd.DataFrame | None:
    if not MEMORY_CSV.exists():
        return None
    return sf.memory_by_stage(pd.read_csv(MEMORY_CSV), stages)


def stage_report():
    stages = json.loads(STAGES_JSON.read_text())
    bench = json.loads(BENCH_JSON.read_text())
    checks = pd.DataFrame(json.loads(CHECK_JSON.read_text()))
    with open(DESCRIBE_PKL, "rb") as f:
        d = pickle.load(f)
    with open(VALIDATE_PKL, "rb") as f:
        v = pickle.load(f)
    p0v2 = vp._load("v2")
    changed = guard_changes()
    man = research.load("C1_full", "manifest")
    code = man["code"]
    L = ["# Research program S8: C1's full scored run and descriptive validation", "",
         "Generated by `python3.12 -m gamecalling.research_s8_full report` (stages `benchmark`, `guard`, `run`, `check`, "
         "`describe`, `halves`, `s1runs`, `validate`). Specification: `research_notes/research_program/S8_combined_model.md` "
         "(revision 1), §6 stages 4-5 and §8. **Descriptive throughout. 2024 and 2025 are previously examined seasons; "
         "every result here is exploratory reuse.** D is primary, A the comparison, B a diagnostic. No intervals or "
         "significance; reliability is never used to choose a model or a score. C1 was adopted by the prediction gate "
         "alone (`reports/research_s8_gate.md`); nothing here revisits that.", "",
         f"Code at the full run: `{code['gamecalling_sha256'][:12]}` on `{code['git_head'][:7]}`, uncommitted: "
         f"{', '.join(code['git_dirty_files']) or 'none'}.", "",
         "## 1. Runtime benchmark (before the full run)", "",
         "The plan's check: one season-fold (2024, scoring fold 0) at one math thread, each job in its own process, "
         "then the extrapolation to the full run (budget 60 min at 8 workers) and the independent halves (budget 120).",
         "", _md(pd.DataFrame([bench["estimate"]]).T.reset_index().rename(columns={"index": "quantity", 0: "value"})), "",
         f"Measured: role fits {bench['fits']['selection']['seconds']:.0f} s and {bench['fits']['valuation']['seconds']:.0f} s "
         f"(peaks {bench['fits']['selection']['peak_rss_gb']:.1f} and {bench['fits']['valuation']['peak_rss_gb']:.1f} GB); "
         f"scoring job {bench['score_job']['seconds']:.0f} s ({bench['score_job']['peak_rss_gb']:.1f} GB); one half "
         f"{bench['half']['seconds']:.0f} s ({bench['half']['peak_rss_gb']:.1f} GB). Reuse check (limit 1e-8): largest "
         f"difference {max(v_ for r in bench['reuse']['rows'] for k, v_ in r.items() if k.startswith('max_diff')):.2e}; "
         f"identity count consistent: "
         f"{bench['reuse']['identity_count_ok']}.", "",
         "## 2. Runtime and memory of the stages (measured)", ""]
    mt = memory_table(stages)
    st = pd.DataFrame([{"stage": n, "minutes": s["seconds"] / 60, "parent_peak_gb": s["parent_peak_rss_gb"]}
                       for n, s in stages.items()])
    L += [_md(mt if mt is not None else st, "{:.1f}"), ""]
    if mt is not None:
        L += ["Python sums are 10-second samples and count shared pages more than once.", ""]
    jobs = [{"stage": n, "job_peak_gb": s["timings"]["peak_rss_gb"], "wall_fit_s": s["timings"]["wall_fit"],
             "wall_pipeline_s": s["timings"]["wall_total"], "slowest_fit_s": max(s["timings"]["fit_jobs"]),
             "slowest_score_s": max(s["timings"]["score_jobs"]), "abnormal": s["abnormal"]}
            for n, s in stages.items() if n.startswith("run_")]
    L += [_md(pd.DataFrame(jobs), "{:.1f}"), "",
          "## 3. Checks", "",
          f"Stored outputs: {len(json.loads(GUARD_JSON.read_text()))} files hashed before and after: "
          f"{'none changed' if not changed else 'CHANGED: ' + ', '.join(changed)}.", "",
          _md(checks[[c for c in ("check", "identical", "rows", "detail") if c in checks]]), "",
          "Per-action rows against C1's scores (recommendation, Q_actual and D rebuilt for every scored pitch):", "",
          _md(pd.DataFrame(d["reconstruct"]), "{:.2e}"), "",
          f"Same scored pitches as P0 and M4: {d['same_scored']}. Coverage against P0:", "",
          _md(d["coverage"][["season", "status", "P0", "version", "change"]]), "",
          "## 4. Recommendations and decision relevance", "",
          "`agree` is model agreement. Means are per pitch, internal cost in runs (lower is better).", "",
          _md(d["recommend"][["version", "season", "pitches", "agree", "mean_D", "mean_A", "mean_B"]], "{:.5f}"), "",
          "S4's conventions: the share of decisions whose recommended pitch changes, and the pitch-weighted SD of the "
          "change in Q(a) − Q(b) over eligible pairs (runs per decision), with the base's own SD as scale.", "",
          _md(d["relevance"][["base", "version", "season", "pitches", "share_rec_changed", "sd_pair_change",
                              "sd_pair_base", "ratio"]], "{:.5f}"), "",
          "## 5. Catcher effects against P0 and M4", "",
          "Runs saved per 100 pitches, random-effects model, §5b common set (same set in both comparisons: "
          f"{'yes' if d['common_equal'] else 'NO'}). Differences after centering; rank moves count catchers moving 10 or "
          "more places. Changes, not improvements; point differences only.", ""]
    for (b, c), r in d["rank"].items():
        L += [f"### C1 against {b}", "", _md(r["table"][[k for k in rc.RANK_COLS if k in r["table"]]]), ""]
    pr = v["prediction"]
    L += ["## 6. Validation §2: held-out prediction and the forward check", "",
          "Matched held-out log loss by season, C1 against P0 (outcome: nats per fit-eligible pitch; contact: nats per "
          "contact-fit-eligible ball in play). These are the gate's quantities, repeated from the full run.", ""]
    for kind, cmp in pr["heldout"].items():
        L += [f"**{kind}:**", "", _md(_loss_rows(cmp, "P0"), "{:.6f}"), ""]
    L += ["Forward check: models fitted on all of 2024, evaluated on 2025 fit-eligible pitches. Seen cells: the pitcher "
          "threw that pitch type in 2024; unseen: they did not. Strictly historical; batters unseen in 2024 are at the "
          "league value.", "", _md(pr["forward"], "{:.6f}"), "",
          "## 7. Validation §3: within-season reliability", "",
          "V0's frozen random and calendar halves and eligibility rules. (a) splits C1's existing cross-fitted scores; "
          "(b) refits every fitted part of C1 inside each half. P0's stored values alongside (`_P0`).", ""]
    for key, title in (("split_random", "(a) Split of existing scores, random halves"),
                       ("refit", "(b) Independent whole-pipeline refit"),
                       ("split_calendar", "(a') Split of existing scores, calendar halves (secondary)")):
        L += [f"### {title}", "", _md(vp._cmp_table(_season_side(v[key], p0v2[key]).to_dict("records")), "{:.4f}"), ""]
        L += ["Catchers compared: " + ", ".join(f"{se} {r['n_eligible']}" for se, r in v[key].items()) + ".", ""]
        gaps = [pd.DataFrame(r["gap"]).assign(season=se) for se, r in v[key].items() if r["gap"]]
        if gaps:
            L += ["Group gap (top and bottom thirds chosen on one half, mean on the other):", "",
                  _md(pd.concat(gaps, ignore_index=True), "{:.4f}"), ""]
    rnd, fx = _halves_tables(v, p0v2)
    L += ["Independent refit across versions (random effects), with M3a's and M4's S6 results:", "",
          _md(rnd, "{:.4f}"), "", "Pitcher fixed effects:", "", _md(fx, "{:.4f}"), ""]
    r = v["repeatability"]
    L += ["## 8. Validation §4: 2024-2025 repeatability (exploratory reuse of previously examined seasons)", "",
          f"Catchers qualifying in both seasons and in both main groups: {len(r['eligible'])}. Seasons fitted "
          "independently.", "", _md(vp._cmp_table(_side(r["table"], p0v2["repeatability"]["table"]).to_dict("records")),
                                    "{:.4f}"), "",
          "By mover group (V0's groups; random effects):", "",
          _md(vp._cmp_table(_side(r["by_group"], p0v2["repeatability"]["by_group"], ("group", "score")).to_dict("records")),
              "{:.4f}"), "",
          "## 9. Robustness", "",
          "**Bridge removal is not rerun** (S8 §8). The reused bridge-removal results are P0 evidence; they do not "
          "establish C1's bridge-removal robustness. The pitcher-fixed-effects check is in each fit.", "",
          "**R1-R4 (§5b):** R1 optimal continuation, R2 one-pitch RE288 benchmark, R3 without dugout-called games, R4 "
          "bases-empty pitches; R3 and R4 change only the ranking sample. P0's values alongside.", "",
          _md(vp._cmp_table(_side(v["sensitivity"]["table"], p0v2["sensitivity"]["table"], ("season", "check")
                                  ).to_dict("records")), "{:.4f}"), ""]
    s1 = v["s1"]
    sh = s1["a_shares"]
    L += ["### S1 checks under C1", "",
          "As `reports/research_s1.md`, with C1 as the reference. (a) and (b) are ranking refits on C1's scores; (c), (d) "
          "and (e) are full reruns under C1. Arsenal eligibility is unchanged in every check. These identify "
          "sensitivity; they cannot by themselves establish why rankings differ.", "",
          f"**(a) Bases-empty vs. runners-on ranking fits** (bases-empty share {sh.iloc[0]:.3f} / {sh.iloc[-1]:.3f}):", "",
          rc._rank_md(s1["a"]), "",
          "Rough context only: a seeded random split with the same proportions, **drawn pitch by pitch.** Its two "
          "parts share plate appearances and games, which the bases-empty vs. runners-on split mostly does not, so "
          "it is not a like-for-like baseline for (a):", "",
          rc._rank_md(s1["a_context_random"]), ""]
    if S1A_SPLITS_PKL.exists():
        with open(S1A_SPLITS_PKL, "rb") as f:
            sp = pickle.load(f)
        share = lambda k: " / ".join(f"{x:.3f}" for x in sp["shares"][k])  # noqa: E731
        L += ["**Added 2026-11-14, after the validation (stage `s1a_splits`; descriptive, one seed each):** the same "
              "random split with whole plate appearances, then whole games, assigned to one part (same probability; "
              f"realized pitch share {share('pa')} and {share('game')}). Whole plate appearances:", "",
              rc._rank_md(sp["pa"]), "", "Whole games:", "", rc._rank_md(sp["game"]), "",
              "Agreement falls as the split unit grows. The plate-appearance split is the closer baseline for (a); "
              "the bases-empty and runners-on parts also differ in context, pitch counts and spread, so this "
              "remains rough context and identifies no cause.", ""]
    L += [
          "**(b) Addendum sample restriction** (PAs with a base-out change within the PA, a censored endpoint or an "
          f"excluded decision removed; scored pitches removed: {', '.join(f'{k} {v_:,}' for k, v_ in s1['b_removed'].items())}):",
          "", rc._rank_md(s1["b"]), "",
          "**(c) Dugout-called games removed from training, payoffs and evaluation (2025).** Prediction against C1 on "
          "matched observations:", "", _md(rc._pred_table(s1["c_pred"]).rename(columns={"P0": "C1", "n_P0": "n_C1",
                                                                                         "max_cal_P0": "max_cal_C1"}),
                                          "{:.6f}"), "",
          "Against C1 on all of 2025:", "", rc._rank_md(s1["c"]), "",
          "Against C1's scores without those games (R3's sample), which isolates the training and payoff change:", "",
          rc._rank_md(s1["c_vs_R3"]), "",
          "**(d) RE24 from all complete half-innings.** Probability training is unchanged, so predictions should be "
          f"identical (largest RE24 change by fold {s1['d_re24']['max_abs_change'].max():.4f} runs):", "",
          _md(rc._pred_table(s1["d_pred"]).rename(columns={"P0": "C1", "n_P0": "n_C1", "max_cal_P0": "max_cal_C1"}),
              "{:.6f}"), "", rc._rank_md(s1["d"]), "",
          "**(e) Bases-empty probability training.** Outcome, contact and policy models and the batter features "
          "trained on bases-empty pitches; payoffs and arsenal unchanged. The runner hierarchy stays active (no "
          "constant-flag bypass). **This check cannot evaluate the runners-on contrast,** because no runners-on pitch "
          "is trained on or evaluated.", "",
          _md(rc._pred_table(s1["e_pred"]).rename(columns={"P0": "C1", "n_P0": "n_C1", "max_cal_P0": "max_cal_C1"}),
              "{:.6f}"), "", "Against C1 on all pitches:", "", rc._rank_md(s1["e_vs_C1"]), "",
          "Against C1's scores on bases-empty pitches (R4's sample), which isolates the training change:", "",
          rc._rank_md(s1["e_vs_R4"]), "",
          "Runner concentrations in S1 (e)'s roles (reported; with no runners-on rows the runner levels are fitted on "
          "the bases-empty slice only):", "",
          _md(s1["e_alphas"][[c for c in s1["e_alphas"] if c in ("season", "fold") or "alpha_league_r" in c
                              or c.endswith("alpha_r")]], "{:.6g}"), ""]
    ts = v["team_staff"]
    L += ["## 10. Validation §6: team and staff dependence", "",
          "Between-team share of fitted-score variation (descriptive share, not the fraction caused by teams):", "",
          _md(ts["share"].merge(p0v2["team_staff"]["share"][["season", "score", "share"]], on=["season", "score"],
                                suffixes=("", "_P0")), "{:.3f}"), "",
          "Staff diagnostic (catcher effect vs. pitch-weighted mean pitcher effect, same fit, cost scale):", "",
          _md(vp._cmp_table(_side(ts["staff"], p0v2["team_staff"]["staff"], ("season", "score")).to_dict("records")),
              "{:.4f}"), ""]
    ce, pe, pm = v["catchers"], v["pitches"], v["pitch_meta"]
    cols = [c for c in ("season", "group", "catcher", "rank_D", "rank_A", "D_saved", "A_saved", "B_saved", "pitches",
                        "top_pitchers") if c in ce]
    notes = ce[ce["catcher"].isna()] if "catcher" in ce else ce.iloc[:0]
    ce = ce[ce["catcher"].notna()] if "catcher" in ce else ce
    L += ["## 11. Validation §7: baseball interpretation", "",
          "Selected by the rules frozen in V0 (same seeds and tie-breaks) applied to C1's scores; the selected cases are "
          "new, and **every one is reported.** Ranks: 1 = most runs saved, among qualifying main-group catchers.", "",
          _md(ce[cols], "{:.4f}"), "",
          *([_md(notes[[c for c in ("season", "group", "note") if c in notes]]), ""] if len(notes) else []),
          f"**Pitch level ({vp.PITCH_EX_SEASON}).** C1's percentiles (runs per pitch): D 5th/50th/95th "
          f"{pm['D_q'][0]:.4f} / {pm['D_q'][1]:.4f} / {pm['D_q'][2]:.4f}; A {pm['A_q'][0]:.4f} / {pm['A_q'][1]:.4f} / "
          f"{pm['A_q'][2]:.4f}. Candidates: " + ", ".join(f"{k} {n:,}" for k, n in pm["counts"].items()) + ". Q for "
          "every eligible action comes from C1's per-action rows (valuation role, against this batter); pi_ref is "
          "refitted from the valuation fold. Both give back the stored values (largest D difference "
          f"{pe['D_recomputed_diff'].max():.1e}, V {pe['V_recomputed_diff'].max():.1e}). Costs x 100 (runs for the "
          "batting team). The recommended pitch comes from the independent selection role, so it need not have the "
          "lowest valuation Q.", "", _md(pe.drop(columns=["D_recomputed_diff", "V_recomputed_diff"]).astype(
              {c: str for c in ("game_pk", "pitcher", "catcher", "batter")}), "{:.3f}"), "",
          "## 12. Validation §8: D-vs.-A evidence table", "",
          "The addendum's rule: D stays primary unless the table as a whole supports a documented revision; no single "
          "row decides, and reliability or persistence alone never selects a score. **The outcome stays inconclusive "
          "unless the user decides otherwise;** this table is reported for that review. P0's stored values alongside.",
          "", _md(evidence_table(v, p0v2)), ""]
    drop = []
    for name, cmp in v["drop_one"].items():
        s = cmp["season"].set_index("season")
        drop.append({"dropped": name, "gain_2024": s["gain"].iloc[0], "gain_2025": s["gain"].iloc[-1],
                     "forward_gain": cmp.get("forward", {}).get("gain", np.nan)})
    rk = d["rank"]["M4", "C1"]["table"]
    L += ["## 13. Drop-one contributions (combined contact survived; descriptive)", "",
          "Prediction: the gain of C1 over the version without that component (nats; positive = C1 predicts better). "
          "Only the comparison with M4 entered a decision (the gate's amendment). Scores: only C1 vs. M4 exists; the "
          "other versions are not scored, and no version is fitted to fill the blanks.", "",
          _md(pd.DataFrame(drop), "{:.6f}"), "",
          "Scores, C1 against M4 (dropping M1's runner levels):", "", _md(rk[[k for k in rc.RANK_COLS if k in rk]]), "",
          "Dropping M4's batter features in contact, or M3a's outcome: blank (no such version is scored).", "",
          "## 14. What this does not show", "",
          "- A passing gate shows better probabilities, not better comparisons between pitch choices; these changes in "
          "recommendations and rankings are changes, not improvements.",
          "- Everything is descriptive on previously examined seasons. Only S9's 2026 data can be genuinely unseen.",
          "- No uncertainty procedure has been validated: no intervals, significance, upper bounds or absence claims.",
          "- Scores are catcher-associated: nobody knows who called each pitch.",
          "- Bridge-removal robustness is P0 evidence only.", ""]
    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT}; stored files changed: {changed or 'none'}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    fns = {"benchmark": stage_benchmark, "guard": stage_guard, "run": stage_run, "check": stage_check,
           "describe": stage_describe, "halves": stage_halves, "validate": stage_validate,
           "s1a_splits": stage_s1a_splits}
    if stage == "s1runs":
        stage_s1runs(sys.argv[2:] or None)
    elif stage == "report":
        stage_report()
    else:
        fns[stage]()
