"""Research program S6 steps 3-4: full scored runs of M3a and M4 and their descriptive comparison (S6 solver addendum §7).

Both versions were cleared by the user's review of the 16-million follow-up (addendum §12, §13). Stages, in order:
- guard: hash the stored prediction-stage outputs and P0's outputs, so the report can show they were not changed.
- run MODEL: the full scored run, into data/models/<MODEL>_full/ (per-action rows in data/models/<MODEL>/runs/full/).
- check: held-out and forward predictions against the stored prediction-stage files, exactly. Any difference stops.
- p0rows: P0's per-action rows, label `full` (addendum §5.3); its stored outputs are not touched.
- describe: scores against the per-action rows; recommendations and decision relevance; D, A and B against P0 and M4
  against M3a on the common catcher set; catcher effects against the mean batter index faced; coverage.
- halves MODEL: the independent-halves refit and its reliability (validation protocol §3b).
- report: reports/research_s6_full.md.

Everything here is descriptive and never used for adoption; 2024 and 2025 are previously examined seasons.
S6_FULL_SMOKE=<dir> runs every stage on one game in five, with all outputs under that directory.
"""

from gamecalling import research_s6  # noqa: I001 (first: sets the thread limit)

import hashlib
import json
import os
import pickle
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from gamecalling import arsenal, batter, batter_scoring as bs, config, research, scoring
from gamecalling import validation_pilot as vp
from gamecalling.phase1 import _md
from gamecalling.research_checks import RANK_COLS, _coverage_diff
from gamecalling.research_s6 import PRED, verify_threads
from gamecalling.research_s6_benchmark import SPECS as BASE_SPECS

SMOKE = os.environ.get("S6_FULL_SMOKE")
WORKERS = 8
LABEL = "full"
SCORES3 = (("D", "D"), ("A", "A"), ("B", "B"))
FEATURES = ("whiff", "chase", "xwobacon", "ground_ball")
PAIRS = (("P0", "M3a"), ("P0", "M4"), ("M3a", "M4"))
STORED_MODELS = config.DATA_DIR / "models"


def _smoke_subset(pitches, pas, inner=None):
    """Smoke test only: one game in five, after the version's own transform."""
    if inner is not None:
        pitches, pas = inner(pitches, pas)
    return pitches[pitches["game_pk"] % 5 == 0].copy(), pas[pas["game_pk"] % 5 == 0].copy()


if SMOKE:
    OUT, REPORT = Path(SMOKE), Path(SMOKE) / "research_s6_full.md"
    research.MODELS_DIR = OUT / "models"
    SPECS = {m: replace(s, transform=partial(_smoke_subset, inner=s.transform)) for m, s in BASE_SPECS.items()}
    P0_SPEC = replace(research.P0, transform=_smoke_subset)
else:
    OUT, REPORT = config.DATA_DIR / "research", config.REPORTS_DIR / "research_s6_full.md"
    SPECS, P0_SPEC = BASE_SPECS, research.P0

STAGES_JSON = OUT / "s6_full_stages.json"
GUARD_JSON = OUT / "s6_full_guard.json"
CHECK_JSON = OUT / "s6_full_check.json"
P0_REFS_JSON = OUT / "s6_full_p0_refs.json"
DESCRIBE_PKL = OUT / "s6_full_describe.pkl"
EFFECTS_CSV = OUT / "s6_full_catcher_effects.csv"
MEMORY_CSV = OUT / "s6_full_memory.csv"


def full_id(model_id: str) -> str:
    return f"{model_id}_full"


def _record(stage: str, t_start: float, **extra) -> None:
    """Runtime and the parent's peak memory of a stage; the 10-second system samples are joined on these times."""
    done = json.loads(STAGES_JSON.read_text()) if STAGES_JSON.exists() else {}
    done[stage] = {"started": t_start, "finished": time.time(), "seconds": time.time() - t_start,
                   "parent_peak_rss_gb": bs.peak_rss_gb(), **extra}
    STAGES_JSON.write_text(json.dumps(done, indent=1, default=str))
    print(f"stage {stage}: {done[stage]['seconds']:.0f} s, parent peak {done[stage]['parent_peak_rss_gb']:.1f} GB", flush=True)


# --- Guard on stored outputs ---

def stored_hashes(paths) -> dict:
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def guarded_files() -> list:
    """Top-level files only: runs/ under a stored version is where this stage adds its per-action rows."""
    files = [config.DATA_DIR / "phase1_scores.parquet"]
    for m in ("P0_repro", "M3a", "M4"):
        files += sorted(p for p in (STORED_MODELS / m).iterdir() if p.is_file())
    return files


def changed_hashes(before: dict, after: dict) -> list:
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


def stage_guard():
    OUT.mkdir(parents=True, exist_ok=True)
    if GUARD_JSON.exists():
        raise FileExistsError(f"{GUARD_JSON} exists")
    GUARD_JSON.write_text(json.dumps(stored_hashes(guarded_files()), indent=1))
    print(f"guarded {len(json.loads(GUARD_JSON.read_text()))} files", flush=True)


# --- Full runs ---

def stage_run(model_id: str):
    t = time.time()
    out = research.MODELS_DIR / full_id(model_id)
    if out.exists():
        raise FileExistsError(f"{out} exists: the full run is not repeated by accident")
    manifest = research.run_model(SPECS[model_id], out_name=full_id(model_id), run_label=LABEL, workers=WORKERS)
    _record(f"run_{model_id}", t, workers=WORKERS, scored=len(research.load(full_id(model_id), "scores")),
            timings=manifest["timings"], abnormal=len(manifest["abnormal_log"]))


# --- Exact prediction comparison ---

def exact_compare(new: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """Identical means the same columns, rows, order and values at full precision. The largest difference is only a
    diagnostic for a failure; no tolerance applies."""
    if new.shape != stored.shape or list(new.columns) != list(stored.columns):
        return {"identical": False, "detail": f"shape or columns differ: {new.shape} vs. {stored.shape}"}
    diff = [c for c in stored.columns if not new[c].equals(stored[c])]
    if not diff:
        return {"identical": True, "rows": len(new), "detail": "identical"}
    num = [c for c in diff if pd.api.types.is_numeric_dtype(stored[c])]
    largest = float(max((new[c] - stored[c]).abs().max() for c in num)) if num else None
    return {"identical": False, "rows": len(new), "detail": f"differ: {', '.join(diff)}", "largest_difference": largest}


def stage_check():
    t = time.time()
    rows = []
    for m in SPECS:
        ref = research.MODELS_DIR / full_id(m) if SMOKE else STORED_MODELS / m
        for what in PRED:
            r = exact_compare(research.load(full_id(m), what), pd.read_parquet(ref / f"{what}.parquet"))
            rows.append({"check": f"{m}: {what} vs. the stored prediction stage", **r})
    for what in ("heldout_outcome", "forward_outcome"):
        r = exact_compare(research.load(full_id("M4"), what), research.load(full_id("M3a"), what))
        rows.append({"check": f"M4's {what} vs. M3a's, both full runs", **r})
    CHECK_JSON.write_text(json.dumps(rows, indent=1))
    print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)
    _record("check", t)
    if not all(r["identical"] for r in rows):
        raise RuntimeError("a full-run prediction differs from the stored prediction stage: stop and diagnose")


# --- P0's per-action rows ---

def _p0_job(season: int) -> dict:
    verify_threads()
    pitches, pas, support = research.prepared(P0_SPEC)
    run = scoring.run_pilot(pitches, pas, support, seasons=(season,), fit_role=research.role_fitter(P0_SPEC),
                            fit_policy=P0_SPEC.fit_policy, solve_role=P0_SPEC.solve_role, keep_solutions=True)
    refs = []
    for k in range(config.N_FOLDS):
        sol = run.solutions[season, k]
        p = bs.scored_pitches(pitches, season, k, sol["units"])
        sel, val = bs.values_from_solution(sol["select"], p), bs.values_from_solution(sol["value"], p)
        rows = bs.action_value_rows(p, sel.Q_ref, val.Q_ref, sol["units"], k)
        refs.append(bs.write_action_values(rows, research.MODELS_DIR, "P0_repro", LABEL, season, k))
    return {"refs": refs, "peak_rss_gb": bs.peak_rss_gb()}


def stage_p0rows():
    t = time.time()
    with ProcessPoolExecutor(max_workers=len(config.SEASONS)) as ex:
        res = list(ex.map(_p0_job, config.SEASONS))
    P0_REFS_JSON.write_text(json.dumps([r for part in res for r in part["refs"]], indent=1))
    if SMOKE:
        pitches = research.prepared(P0_SPEC)[0]
        research.coverage(pitches, config.SEASONS).to_csv(research.MODELS_DIR / "P0_repro" / "coverage.csv", index=False)
    _record("p0rows", t, job_peak_rss_gb=[r["peak_rss_gb"] for r in res])


# --- Descriptive measures ---

def reconstruct_check(rows: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """The per-action rows must give back each pitch's recommendation, Q_actual and D (first minimum, in slot order)."""
    rec = rows.loc[rows.groupby(bs.PITCH_KEYS, sort=False)["Q_select"].idxmin(), bs.PITCH_KEYS + ["action", "Q_value"]]
    rec = rec.rename(columns={"action": "rec_rows", "Q_value": "Q_rec"})
    m = scores[bs.PITCH_KEYS + ["action", "rec_action", "Q_actual", "D"]].merge(rec, on=bs.PITCH_KEYS, how="outer",
                                                                               indicator=True, validate="one_to_one")
    unmatched = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"].merge(rows[bs.PITCH_KEYS + ["action", "Q_value"]], on=bs.PITCH_KEYS + ["action"], how="left",
                                       validate="one_to_one")
    return {"pitches": len(m), "unmatched": unmatched, "rec_differs": int((m["rec_action"] != m["rec_rows"]).sum()),
            "max_Q_actual_diff": float((m["Q_actual"] - m["Q_value"]).abs().max()),
            "max_D_diff": float((m["D"] - (m["Q_actual"] - m["Q_rec"])).abs().max())}


def p0_scores() -> pd.DataFrame:
    s = research.load("P0", "scores")
    return s[s["game_pk"] % 5 == 0].copy() if SMOKE else s


def index_faced(scores: pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    """Per catcher-season, the mean standardized batter feature over scored pitches. Each pitch uses the valuation role's
    features, the ones that value it. M3a's three features are M4's first three."""
    parts = []
    for (season, k), s in scores.groupby(["season", "fold"]):
        v = arsenal.fold_roles(int(k))[1]
        train = pitches[(pitches["season"] == season) & (pitches["fold"] == v)]
        z = batter.fit_features(train, ground_ball=True).z_full(s["batter"])
        three = batter.fit_features(train, ground_ball=False).z_full(s["batter"])
        if not np.array_equal(z[:, :3], three):
            raise ValueError("M3a's features are not M4's first three")
        parts.append(pd.DataFrame(z, columns=list(FEATURES)).assign(season=season, catcher=s["catcher"].to_numpy()))
    return catcher_means(pd.concat(parts, ignore_index=True))


def catcher_means(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["season", "catcher"])[list(FEATURES)].mean()


def index_correlations(cache: dict, label: str, faced: pd.DataFrame, common: dict) -> pd.DataFrame:
    rows = []
    for season, cs in common.items():
        f = faced.loc[season].reindex(cs)
        for version in ("P0", "M3a", "M4"):
            for score, _ in SCORES3:
                fit = cache[label, season, version, score]
                for feat in FEATURES:
                    r = vp.compare(fit["saved"].reindex(cs), f[feat], fit["zero"])
                    rows.append({"season": season, "version": version, "score": score, "feature": feat, "n": r["n"],
                                 "pearson": r["pearson"], "spearman": r["spearman"]})
    return pd.DataFrame(rows)


def stage_describe():
    t = time.time()
    scores = {"P0": p0_scores(), **{m: research.load(full_id(m), "scores") for m in SPECS}}
    refs = {"P0": json.loads(P0_REFS_JSON.read_text()),
            **{m: research.load(full_id(m), "manifest")["action_values"] for m in SPECS}}
    d = {"reconstruct": []}
    for m, r in refs.items():
        d["reconstruct"].append({"version": m, **reconstruct_check(bs.read_action_values(r), scores[m])})
    print(pd.DataFrame(d["reconstruct"]).to_string(), flush=True)
    bad = [r for r in d["reconstruct"] if r["unmatched"] or r["rec_differs"] or max(r["max_Q_actual_diff"], r["max_D_diff"]) > 1e-12]
    if bad and not SMOKE:   # the smoke test's P0 rows are fitted on the subset, its stored scores on the full season
        raise RuntimeError(f"per-action rows do not give back the scores: {bad}")

    d["recommend"] = pd.concat([s.groupby("season").agg(pitches=("agree", "size"), agree=("agree", "mean"), mean_D=("D", "mean"),
                                                        mean_A=("A", "mean"), mean_B=("B", "mean")).reset_index().assign(version=m)
                                for m, s in scores.items()], ignore_index=True)
    d["relevance"] = pd.concat([bs.decision_relevance_rows(refs[b], refs[c]).assign(base=b, version=c) for b, c in PAIRS],
                               ignore_index=True)
    cache, label = {}, "s6_full"
    d["rank"] = {(b, c): research.rank_compare(scores[b], scores[b], scores[c], SCORES3, label, cache, b, c) for b, c in PAIRS}
    pitches = research.prepared(SPECS["M4"])[0]
    faced = index_faced(scores["M4"], pitches)
    common = d["rank"]["P0", "M3a"]["common"]
    d["common_equal"] = all(d["rank"][pair]["common"] == common for pair in PAIRS)
    d["faced_sd"] = faced.groupby("season").std().reset_index()
    d["index_corr"] = index_correlations(cache, label, faced, common)
    d["coverage"] = {m: _coverage_diff(full_id(m), config.SEASONS) for m in SPECS}
    d["same_scored"] = {m: bool(scores[m][bs.PITCH_KEYS].sort_values(bs.PITCH_KEYS).reset_index(drop=True).equals(
        scores["P0"][bs.PITCH_KEYS].sort_values(bs.PITCH_KEYS).reset_index(drop=True))) for m in SPECS}
    effects = pd.concat([pd.DataFrame({"season": season, "version": version, "score": score,
                                       "saved": cache[label, season, version, score]["saved"].reindex(cs)}
                                      ).rename_axis("catcher").reset_index()
                         for season, cs in common.items() for version in scores for score, _ in SCORES3], ignore_index=True)
    effects.to_csv(EFFECTS_CSV, index=False)
    faced.reset_index().to_csv(OUT / "s6_full_index_faced.csv", index=False)
    with open(DESCRIBE_PKL, "wb") as f:
        pickle.dump(d, f)
    _record("describe", t)


# --- Independent halves ---

def _half_job(args) -> dict:
    threads = verify_threads()
    r = research._half_job(args)
    r["meta"] = {"season": r["season"], "half": r["half"], "scored": len(r["scores"]), "seconds": r["seconds"],
                 "peak_rss_gb": bs.peak_rss_gb(), **threads}
    return r


def halves_path(model_id: str) -> Path:
    return OUT / f"s6_full_halves_{model_id}.pkl"


def stage_halves(model_id: str):
    t = time.time()
    path = halves_path(model_id)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    runs = research.independent_halves(SPECS[model_id], WORKERS, job=_half_job)
    compare = vp.within_refit_compare(runs)
    with open(path, "wb") as f:
        pickle.dump({"runs": runs, "compare": compare}, f)
    _record(f"halves_{model_id}", t, jobs=[r["meta"] for r in runs.values()])


# --- Report ---

def memory_by_stage(samples: pd.DataFrame, stages: dict) -> pd.DataFrame:
    """Peak summed Python memory (shared pages counted more than once), swap and lowest free memory in each stage."""
    rows = []
    for name, s in stages.items():
        g = samples[(samples["t"] >= s["started"]) & (samples["t"] <= s["finished"])]
        rows.append({"stage": name, "minutes": s["seconds"] / 60, "parent_peak_gb": s["parent_peak_rss_gb"], "samples": len(g),
                     "peak_python_sum_gb": g["py_rss_gb"].max(), "swap_max_mb": g["swap_mb"].max(), "free_min_pct": g["free_pct"].min()})
    return pd.DataFrame(rows)


def _halves_table(model_id: str) -> pd.DataFrame:
    with open(halves_path(model_id), "rb") as f:
        compare = pickle.load(f)["compare"]
    return pd.concat([pd.DataFrame(compare[s]["compare"]).assign(season=s, version=model_id) for s in compare], ignore_index=True)


def stage_report():
    stages = json.loads(STAGES_JSON.read_text())
    checks = pd.DataFrame(json.loads(CHECK_JSON.read_text()))
    with open(DESCRIBE_PKL, "rb") as f:
        d = pickle.load(f)
    changed = changed_hashes(json.loads(GUARD_JSON.read_text()), stored_hashes(guarded_files()))
    code = research.load(full_id("M4"), "manifest")["code"]
    out = ["# S6 steps 3-4: full scored runs of M3a and M4", "",
           "Descriptive, never used for adoption. Seasons previously examined; exploratory reuse. The 16-million follow-up "
           "cleared both versions (addendum §13). D is primary, A the comparison, B a diagnostic.", "",
           f"Code at the M4 run: `{code['gamecalling_sha256'][:12]}`, commit `{code['git_head'][:7]}`, uncommitted: "
           f"{', '.join(code['git_dirty_files']) or 'none'}.", "",
           "## Runtime and memory", ""]
    if MEMORY_CSV.exists():
        out += [_md(memory_by_stage(pd.read_csv(MEMORY_CSV), stages), "{:.1f}"), "",
                "Measured. Python sums are 10-second samples and count shared pages more than once; job peaks are below.", ""]
    jobs = [{"stage": n, "job_peak_gb": s["timings"]["peak_rss_gb"], "wall_fit_s": s["timings"]["wall_fit"],
             "wall_pipeline_s": s["timings"]["wall_total"], "slowest_fit_s": max(s["timings"]["fit_jobs"]),
             "slowest_score_s": max(s["timings"]["score_jobs"]), "abnormal": s["abnormal"]}
            for n, s in stages.items() if n.startswith("run_")]
    halves = [{"stage": n, **{k: m[k] for k in ("season", "half", "scored", "seconds", "peak_rss_gb")}}
              for n, s in stages.items() if n.startswith("halves_") for m in s["jobs"]]
    out += [_md(pd.DataFrame(jobs), "{:.1f}"), "", _md(pd.DataFrame(halves), "{:.1f}"), "",
            "## Stored outputs", "",
            f"{len(json.loads(GUARD_JSON.read_text()))} stored files hashed before and after: "
            f"{'none changed' if not changed else 'CHANGED: ' + ', '.join(changed)}.", "",
            "## Exact prediction comparison", "", _md(checks[["check", "identical", "rows", "detail"]]), "",
            "## Per-action rows against the scores", "",
            "Recommendation, Q_actual and D rebuilt from each version's per-action rows, for all scored pitches.", "",
            _md(pd.DataFrame(d["reconstruct"]), "{:.2e}"), "",
            "## Recommendations", "",
            "`agree` is model agreement. Means are per pitch, internal cost in runs (lower is better), not the display sign.", "",
            _md(d["recommend"][["version", "season", "pitches", "agree", "mean_D", "mean_A", "mean_B"]], "{:.5f}"), "",
            "## Decision relevance", "",
            "S4's conventions: the share of decisions whose recommended pitch changes, and the pitch-weighted SD of the change "
            "in Q(a) − Q(b) over eligible pairs (runs per decision), with the base's own SD as scale.", "",
            _md(d["relevance"][["base", "version", "season", "pitches", "share_rec_changed", "sd_pair_change", "sd_pair_base",
                                "ratio"]], "{:.5f}"), "",
            "## Catcher effects", "",
            "Runs saved per 100 pitches, random-effects model, §5b common set. Differences are after centering; rank moves count "
            f"catchers moving 10 or more places. Same common set in all three comparisons: {'yes' if d['common_equal'] else 'NO'}. "
            "Point differences only: no intervals or significance.", ""]
    for (b, c), r in d["rank"].items():
        out += [f"### {c} against {b}", "", _md(r["table"][[k for k in RANK_COLS if k in r["table"]]]), ""]
        if len(r["excluded"]):
            out += [_md(r["excluded"]), ""]
    corr = d["index_corr"]
    out += ["## Catcher effects against the mean batter index faced", "",
            "Pearson correlation on the common set, by season (Spearman in the saved table). Each pitch uses the valuation role's "
            "standardized features. Catcher effects are displayed as runs saved. Descriptive; this does not separate staff, team "
            "or schedule from the catcher.", ""]
    for season, g in corr.groupby("season"):
        wide = g.pivot_table(index=["version", "score"], columns="feature", values="pearson", sort=False, aggfunc="first")
        out += [f"### {season}", "", _md(wide[list(FEATURES)].reset_index(), "{:.3f}"), ""]
    out += ["SD across catchers of the mean feature faced (standardized units):", "", _md(d["faced_sd"], "{:.4f}"), "",
            "## Coverage", "", f"Same scored pitches as P0: {d['same_scored']}.", ""]
    for m, c in d["coverage"].items():
        out += [f"### {m}", "", _md(c[["season", "status", "P0", "version", "change"]]), ""]
    p0 = vp._load("v2")["refit"]
    base = pd.concat([pd.DataFrame(p0[s]["compare"]).assign(season=s, version="P0") for s in p0], ignore_index=True)
    table = pd.concat([base, *[_halves_table(m) for m in SPECS]], ignore_index=True)
    table = table[table["model"] == "random"][["version", "season", "score", "n", "pearson", "spearman", "median_abs_diff", "sd_h0",
                                               "sd_h1", "converged"]]
    fixed = pd.concat([base, *[_halves_table(m) for m in SPECS]], ignore_index=True)
    fixed = fixed[fixed["model"] == "fixed"][["version", "season", "score", "n", "pearson", "spearman"]]
    out += ["## Independent-halves reliability", "",
            "The whole pipeline refitted inside each frozen random half (protocol §3b), with P0's stored results alongside. "
            "Reliability is never used to choose a model or a score.", "",
            "Random-effects model:", "", _md(table.sort_values(["score", "season"], kind="stable")), "",
            "Pitcher fixed effects:", "", _md(fixed.sort_values(["score", "season"], kind="stable")), "",
            "## What this does not show", "",
            "- A passing gate and these comparisons do not validate the catcher scores or establish calling skill.",
            "- Differences are point differences. No uncertainty procedure has been validated.",
            "- Scores are catcher-associated: nobody knows who called each pitch.", ""]
    REPORT.write_text("\n".join(out))
    print(f"wrote {REPORT}; stored files changed: {changed or 'none'}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "guard":
        stage_guard()
    elif stage == "run":
        stage_run(sys.argv[2])
    elif stage == "check":
        stage_check()
    elif stage == "p0rows":
        stage_p0rows()
    elif stage == "describe":
        stage_describe()
    elif stage == "halves":
        stage_halves(sys.argv[2])
    elif stage == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown stage {stage}")
