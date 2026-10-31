"""Thin comparison runner for the research program (project_plan.md, second amendment, S0).

Every model version has an ID and its own directory data/models/<ID>/ with a manifest (specification,
inputs, code state, timings). The runner calls the working pipeline (scoring.run_pilot, models, ranking)
through its default-preserving hooks; it does not reimplement it. P0's stored outputs
(data/phase1_scores.parquet, data/validation_pilot/v1/, the reports) are read, never rewritten.

A specification can change three things, which the comparisons keep apart (plan, S1):
- evaluated situations (`eval_mask`): which pitches are predicted and enter the ranking fit;
- probability training (`train_mask`, `fit_models`, `fit_policy`): which pitches train the outcome,
  contact and policy models, and how;
- payoff estimation and the data themselves (`transform`): for example PA flags or the action unit.
Arsenal eligibility stays P0's (data/arsenal_support.parquet) unless `recompute_support` is set.

Comparisons (all descriptive; 2024 and 2025 are previously examined seasons, so every result here is
exploratory reuse):
- matched held-out prediction: each scoring fold's scorable, fit-eligible pitches predicted by its
  valuation role, compared on the pitches both versions predict, by season and by fold; plus the forward
  2024 -> 2025 check (all 2025 fit-eligible pitches in the evaluated situations);
- the prediction gate (plan, "Prediction gate"), at full precision;
- coverage by exclusion reason, kept separate from prediction;
- ranking changes in runs saved per 100 pitches on a common catcher set (validation protocol §5b rules);
- independent-halves reliability (protocol §3b), for later stages.

Run with: python3.12 -m gamecalling.research s0 [WORKERS]   (P0 reproduction; report reports/research_s0.md)
"""

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, scoring, solver, validation
from gamecalling import validation_pilot as vp
from gamecalling.phase1 import SCORES_PARQUET, _md

MODELS_DIR = config.DATA_DIR / "models"
SUPPORT_PARQUET = config.DATA_DIR / "arsenal_support.parquet"
KEYS = ["season", "pa_id", "pitch_number"]
DELTA = {"outcome": 0.0005, "contact": 0.0005}   # nats per pitch / per ball in play (screening convention)
UNITS = {"outcome": "nats per fit-eligible pitch", "contact": "nats per contact-fit-eligible ball in play"}
CAL_MAX = validation.CALIBRATION_MAX


def default_fit_models(train: pd.DataFrame):
    return models.fit_outcome_model(train), models.fit_contact_model(train)


@dataclass
class Spec:
    id: str
    description: str
    base: str | None = "P0"
    seasons: tuple = tuple(config.SEASONS)
    transform: Callable | None = None      # (pitches, pas) -> (pitches, pas)
    recompute_support: bool = False
    train_mask: Callable | None = None     # pitch rows -> bool: probability-training restriction
    eval_mask: Callable | None = None      # pitch or score rows -> bool: evaluated situations
    fit_models: Callable = default_fit_models
    fit_policy: Callable = models.fit_policy
    solve_role: Callable = scoring.solve_role
    training: Callable | None = None       # (kind, pitch rows) -> observations for held-out / forward prediction
    extra_inputs: tuple = ()               # further input files, hashed into the manifest
    pipeline: Callable | None = None       # fit, solve and score; None is the P0 wrapper (pilot_pipeline)
    params: dict = field(default_factory=dict)

    @property
    def changes_training(self) -> bool:
        return self.train_mask is not None or self.fit_models is not default_fit_models


P0 = Spec("P0", "Count-based pilot (frozen baseline)", base=None)


def _mask(fn, df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index) if fn is None else pd.Series(np.asarray(fn(df), bool), index=df.index)


# --- Data ---

_DATA = {}


def base_tables():
    if "base" not in _DATA:
        _DATA["base"] = (pd.read_parquet(config.PITCHES_PARQUET), pd.read_parquet(config.PAS_PARQUET),
                         pd.read_parquet(SUPPORT_PARQUET))
    return _DATA["base"]


def prepared(spec: Spec):
    """Pitches, PAs and support for a specification (copies only when something changes)."""
    pitches, pas, support = base_tables()
    if spec.transform is not None:
        pitches, pas = spec.transform(pitches.copy(), pas.copy())
    if spec.recompute_support:
        support = arsenal.support_table(pitches)
        pitches = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(support))
    return pitches, pas, support


def _spec_fit_role(spec: Spec, pitches, pas, season, fold, with_re288=True):
    role = models.fit_role(pitches, pas, season, fold, with_re288)
    p = pitches[(pitches["season"] == season) & (pitches["fold"] == fold)]
    train = p[_mask(spec.train_mask, p)]
    outcome, contact = spec.fit_models(train)
    return replace(role, outcome=outcome, contact=contact, policy_train=models.policy_training(train))


def role_fitter(spec: Spec):
    """models.fit_role, with the probability models refit on the specification's training rows (picklable)."""
    return partial(_spec_fit_role, spec) if spec.changes_training else models.fit_role


# --- Held-out and forward predictions ---

def _training(kind, rows, spec: Spec | None = None):
    if spec is not None and spec.training is not None:
        return spec.training(kind, rows)
    return models.outcome_training(rows) if kind == "outcome" else models.contact_training(rows)


def _pred_frame(rows: pd.DataFrame, obs: pd.DataFrame, pred: np.ndarray, **extra) -> pd.DataFrame:
    out = rows.loc[obs.index, KEYS].reset_index(drop=True).assign(**extra, y=obs["y"].to_numpy())
    return pd.concat([out, pd.DataFrame(pred, columns=[f"p{j}" for j in range(pred.shape[1])])], axis=1)


def heldout(pitches, roles, spec: Spec, kind: str) -> pd.DataFrame:
    """Each scoring fold's scorable, fit-eligible pitches in the evaluated situations, by its valuation role."""
    parts = []
    for season in spec.seasons:
        for k in range(config.N_FOLDS):
            val = roles[season, arsenal.fold_roles(k)[1]]
            fold = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
            fold = fold[_mask(spec.eval_mask, fold)]
            obs = _training(kind, fold, spec)
            model = val.outcome if kind == "outcome" else val.contact
            parts.append(_pred_frame(fold, obs, model.predict(obs), fold=k))
    return pd.concat(parts, ignore_index=True)


def forward(pitches, spec: Spec, fitted: dict | None = None, prefit: tuple | None = None) -> dict:
    """Models fit on 2024 (training rows) predict all 2025 fit-eligible pitches in the evaluated situations.
    `fitted`, if given, receives the fitted forward models; `prefit` (outcome, contact) skips the fit."""
    if tuple(spec.seasons) != tuple(config.SEASONS):
        return {}
    train = pitches[pitches["season"] == config.SEASONS[0]]
    train = train[_mask(spec.train_mask, train)]
    test = pitches[pitches["season"] == config.SEASONS[1]]
    test = test[_mask(spec.eval_mask, test)]
    outcome, contact = prefit if prefit is not None else spec.fit_models(train)
    if fitted is not None:
        fitted.update(outcome=outcome, contact=contact)
    seen = set(map(tuple, models.outcome_training(train)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    out = {}
    for kind, model in (("outcome", outcome), ("contact", contact)):
        obs = _training(kind, test, spec)
        f = _pred_frame(test, obs, model.predict(obs), fold=-1)
        f["seen_cell"] = [(a, b) in seen for a, b in obs[["pitcher", "action"]].to_numpy()]
        out[kind] = f
    return out


# --- Running a version ---

def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_state() -> dict:
    files = sorted((config.ROOT / "gamecalling").glob("*.py"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    git = lambda *a: subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty_files": git("status", "--porcelain").splitlines(),
            "gamecalling_sha256": h.hexdigest()}


def input_state(extra=()) -> dict:
    return {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)}
            for p in (config.PITCHES_PARQUET, config.PAS_PARQUET, SUPPORT_PARQUET, *extra)}


def coverage(pitches: pd.DataFrame, seasons) -> pd.DataFrame:
    """Pitch rows by season and status: scorable, or the exclusion / arsenal reason."""
    p = pitches[pitches["season"].isin(seasons)]
    status = p["exclusion_reason"].fillna(p["arsenal_reason"]).fillna("scorable")
    return status.groupby([p["season"], status.rename("status")]).size().rename("pitches").reset_index()


def pilot_pipeline(pitches, pas, support, spec, seasons, fit_role, run_label="full", root=None, workers=1,
                   keep_solutions=True) -> scoring.PipelineRun:
    """The default pipeline: a thin wrapper of scoring.run_pilot (P0 and the array-solve versions)."""
    run = scoring.run_pilot(pitches, pas, support, seasons=seasons, fit_role=fit_role, fit_policy=spec.fit_policy,
                            solve_role=spec.solve_role, keep_solutions=keep_solutions)
    policy = [{"season": s, "scoring_fold": k, "role": nm, "alpha": sol[nm].policy.alpha}
              for (s, k), sol in sorted(run.solutions.items()) for nm in ("select", "value")]
    return scoring.PipelineRun(run.scores, run.roles, policy, [], [], [], {}, run.timings, run.solutions)


def run_model(spec: Spec, out_name: str | None = None, run_label: str = "full", workers: int = 1) -> dict:
    """Fit, solve and score a version; save scores, predictions, coverage and a manifest."""
    out = MODELS_DIR / (out_name or spec.id)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pitches, pas, support = prepared(spec)
    t = time.perf_counter()
    run = (spec.pipeline or pilot_pipeline)(pitches, pas, support, spec, spec.seasons, role_fitter(spec), run_label,
                                            MODELS_DIR, workers)
    t_pipeline = time.perf_counter() - t
    run.scores.to_parquet(out / "scores.parquet", index=False)
    for kind in ("outcome", "contact"):
        heldout(pitches, run.roles, spec, kind).to_parquet(out / f"heldout_{kind}.parquet", index=False)
    for kind, f in forward(pitches, spec).items():
        f.to_parquet(out / f"forward_{kind}.parquet", index=False)
    coverage(pitches, spec.seasons).to_csv(out / "coverage.csv", index=False)
    alphas = [{"season": s, "fold": f, **{f"{kind}_{k}": v for kind in ("outcome", "contact")
                                          for k, v in model_alphas(getattr(r, kind)).items()}}
              for (s, f), r in sorted(run.roles.items())]
    manifest = {"id": spec.id, "directory": out.name, "description": spec.description, "base": spec.base,
                "seasons": list(spec.seasons), "params": spec.params,
                "changes": {"evaluated_situations": spec.eval_mask is not None,
                            "probability_training": spec.changes_training or spec.fit_policy is not models.fit_policy,
                            "solve": spec.solve_role is not scoring.solve_role,
                            "data_transform": spec.transform is not None,
                            "support_recomputed": spec.recompute_support},
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code_state(),
                "inputs": input_state(spec.extra_inputs), "run_label": run_label,
                "seconds": {"pipeline": t_pipeline, "total": time.perf_counter() - t0},
                "alphas": alphas, "policy_alphas": run.policy_alphas, "action_values": run.action_values,
                "solve_log": run.solve_log, "abnormal_log": run.abnormal_log, "fit_ids": run.fit_ids}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def model_alphas(model) -> dict:
    """Fitted concentrations of an outcome or contact model, by level."""
    if hasattr(model, "alphas"):
        return model.alphas()
    out = {"league": model.alpha_league, "cell": model.alpha_cell}
    out.update({a: getattr(model, a) for a in ("alpha_league_r", "alpha_r") if hasattr(model, a)})
    return out


@dataclass
class PredictionRole:
    """The probability models of one role, for prediction-only runs (no payoffs, policy or solve)."""
    outcome: object
    contact: object


def run_predictions(spec: Spec, out_name: str | None = None, prefit: dict | None = None,
                    seconds_fit: float | None = None) -> dict:
    """Prediction-only version run (S5 gate): fit each role's outcome and contact models on the specification's
    training rows and write held-out and forward predictions, coverage and a manifest. No payoffs, solve or
    scores (`run_model` always solves and scores). `prefit` holds models already fitted elsewhere, for parallel
    fitting: {(season, fold): (outcome, contact), "forward": (outcome, contact)}."""
    out = MODELS_DIR / (out_name or spec.id)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pitches, _, _ = prepared(spec)
    roles = {}
    for season in spec.seasons:
        for f in range(config.N_FOLDS):
            if prefit is not None:
                roles[season, f] = PredictionRole(*prefit[season, f])
                continue
            p = pitches[(pitches["season"] == season) & (pitches["fold"] == f)]
            roles[season, f] = PredictionRole(*spec.fit_models(p[_mask(spec.train_mask, p)]))
    t_fit = seconds_fit if seconds_fit is not None else time.perf_counter() - t0
    for kind in ("outcome", "contact"):
        heldout(pitches, roles, spec, kind).to_parquet(out / f"heldout_{kind}.parquet", index=False)
    fwd_models = {}
    for kind, f in forward(pitches, spec, fwd_models, prefit["forward"] if prefit is not None else None).items():
        f.to_parquet(out / f"forward_{kind}.parquet", index=False)
    coverage(pitches, spec.seasons).to_csv(out / "coverage.csv", index=False)
    alphas = [{"season": s, "fold": f, **{f"{kind}_{k}": v for kind in ("outcome", "contact")
                                         for k, v in model_alphas(getattr(r, kind)).items()}}
              for (s, f), r in sorted(roles.items())]
    manifest = {"id": spec.id, "directory": out.name, "description": spec.description, "base": spec.base,
                "seasons": list(spec.seasons), "params": spec.params, "prediction_only": True,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code_state(),
                "inputs": input_state(spec.extra_inputs),
                "seconds": {"fit": t_fit, "total": time.perf_counter() - t0}, "alphas": alphas,
                "forward_alphas": {f"{kind}_{k}": v for kind, m in fwd_models.items()
                                   for k, v in model_alphas(m).items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def load(model_id: str, what: str):
    """Stored outputs of a version. P0's scores are the stored phase1 scores; its predictions come from the
    verified reproduction (data/models/P0_repro/)."""
    if model_id == "P0" and what == "scores":
        return pd.read_parquet(SCORES_PARQUET)
    d = MODELS_DIR / ("P0_repro" if model_id == "P0" else model_id)
    if what == "manifest":
        return json.loads((d / "manifest.json").read_text())
    if what == "coverage":
        return pd.read_csv(d / "coverage.csv")
    path = d / f"{what}.parquet"
    return pd.read_parquet(path) if path.exists() else None


# --- Prediction comparisons ---

def _logp(f: pd.DataFrame) -> np.ndarray:
    P = f.filter(regex=r"^p\d+$").to_numpy()
    return np.log(P[np.arange(len(f)), f["y"].to_numpy()])


def matched(base: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Observations predicted by both versions, with each version's log probability of the outcome.

    The plan requires the same rows in the same role and fold: a key held out in different scoring folds
    (and so predicted by different valuation roles) is refused, as is a disagreeing outcome.
    """
    b = base[KEYS + ["fold", "y"]].assign(lp_base=_logp(base))
    c = cand[KEYS + ["fold", "y"]].assign(lp_cand=_logp(cand))
    m = b.merge(c, on=KEYS, how="inner", suffixes=("", "_c"), validate="one_to_one")
    if (m["fold"] != m["fold_c"]).any():
        raise ValueError("matched observations were held out in different folds")
    if (m["y"] != m["y_c"]).any():
        raise ValueError("matched observations disagree on the outcome")
    return m.drop(columns=["y_c", "fold_c"])


def max_calibration(f: pd.DataFrame, keys: pd.DataFrame | None = None) -> float:
    if keys is not None:
        f = f.merge(keys, on=KEYS, how="inner")
    P, y = f.filter(regex=r"^p\d+$").to_numpy(), f["y"].to_numpy()
    return max(validation.calibration_error(P[:, j], y == j) for j in range(P.shape[1]))


def prediction_compare(base_id: str, cand_id: str, kind: str) -> dict:
    """Log loss of both versions on matched observations: by season, by fold and forward; full precision."""
    hb, hc = load(base_id, f"heldout_{kind}"), load(cand_id, f"heldout_{kind}")
    m = matched(hb, hc)
    keys = m[KEYS]
    rows, folds = [], []
    for season, g in m.groupby("season"):
        rows.append({"season": season, "n": len(g), "n_base": int((hb["season"] == season).sum()),
                     "n_cand": int((hc["season"] == season).sum()), "base": -g["lp_base"].mean(),
                     "cand": -g["lp_cand"].mean(), "gain": (g["lp_cand"] - g["lp_base"]).mean(),
                     "cal_base": max_calibration(hb[hb["season"] == season], keys),
                     "cal_cand": max_calibration(hc[hc["season"] == season], keys)})
        for k, gf in g.groupby("fold"):
            folds.append({"season": season, "fold": k, "n": len(gf), "base": -gf["lp_base"].mean(),
                          "cand": -gf["lp_cand"].mean(), "gain": (gf["lp_cand"] - gf["lp_base"]).mean()})
    out = {"kind": kind, "units": UNITS[kind], "season": pd.DataFrame(rows), "fold": pd.DataFrame(folds)}
    fb, fc = load(base_id, f"forward_{kind}"), load(cand_id, f"forward_{kind}")
    if fb is not None and fc is not None:
        fm = matched(fb, fc)
        out["forward"] = {"n": len(fm), "base": -fm["lp_base"].mean(), "cand": -fm["lp_cand"].mean(),
                          "gain": (fm["lp_cand"] - fm["lp_base"]).mean(),
                          "cal_cand": max_calibration(fc, fm[KEYS])}
    return out


def gate(cmp: dict) -> dict:
    """The prediction gate for one component (plan, revision 2): full precision, both seasons."""
    d = DELTA[cmp["kind"]]
    s = cmp["season"]
    both = bool(len(s) == len(config.SEASONS) and (s["gain"] >= d).all())
    cal = bool((s["cal_cand"] <= CAL_MAX).all())
    fwd = cmp.get("forward")
    fwd_ok = bool(fwd is not None and fwd["cand"] <= fwd["base"])
    return {"kind": cmp["kind"], "delta": d, "both_seasons": both, "calibration": cal, "forward": fwd_ok,
            "passes": both and cal and fwd_ok}


def nested_choice(simple_vs_base: bool, complex_vs_base: bool, complex_vs_simple: bool) -> str:
    """Revision 2's single rule: complex only against both base and simple; else simple if it passes; else base."""
    if complex_vs_base and complex_vs_simple:
        return "complex"
    return "simple" if simple_vs_base else "base"


# --- Ranking comparisons ---

def _fit(cache: dict, key, df: pd.DataFrame, score: str, fixed: bool = False):
    if key not in cache:
        cache[key] = vp.season_fit(df, score, fixed)
    return cache[key]


def rank_compare(ref: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, pairs, label: str, cache=None,
                 x_name="x", y_name="y") -> dict:
    """Catcher effects fitted on samples x and y (each season separately), on one common set (protocol §5b).

    Common set: qualifying in `ref` (>= 1,000 scorable pitches), in ref's main connected group, and for each
    of x and y that is not ref itself, >= 500 pitches and membership in that sample's main group. Ranks are
    recomputed within the common set; differences are in runs saved per 100 pitches after centering.
    """
    cache = {} if cache is None else cache
    rows, excluded, commons = [], [], {}
    for season in sorted(set(ref["season"])):
        r = ref[ref["season"] == season]
        n_ref = r.groupby("catcher").size()
        qual = set(n_ref[n_ref >= vp.QUALIFY].index)
        main_ref = vp.main_group(r, qual)
        common, samples = qual & main_ref, {}
        for name, df in ((x_name, x), (y_name, y)):
            d = r if df is ref else df[df["season"] == season]
            samples[name] = d
            if df is ref:
                continue
            n = d.groupby("catcher").size()
            cand = qual & set(n[n >= vp.HALF_MIN].index)
            main = vp.main_group(d, cand)
            for c in sorted(qual & main_ref - main):
                excluded.append({"season": season, "comparison": label, "sample": name, "catcher": c,
                                 "pitches": int(n.get(c, 0)),
                                 "reason": "pitches" if n.get(c, 0) < vp.HALF_MIN else "not in main group"})
            common &= main
        common = sorted(common)
        commons[season] = common
        for sx, sy in pairs:
            fx = _fit(cache, (label, season, x_name, sx), samples[x_name], sx)
            fy = _fit(cache, (label, season, y_name, sy), samples[y_name], sy)
            a, b = fx["saved"].reindex(common), fy["saved"].reindex(common)
            zero = fx["zero"] or fy["zero"]
            row = {"comparison": label, "season": season, "x": f"{x_name}:{sx}", "y": f"{y_name}:{sy}",
                   **vp.compare(a, b, zero), "sd_x": fx["sd"], "sd_y": fy["sd"],
                   "converged": fx["converged"] and fy["converged"]}
            if zero:
                row.update(max_abs_diff=np.nan, rank_moves=vp.UNAVAILABLE)
            elif len(common) >= vp.MIN_SET:
                ac, bc = a - a.mean(), b - b.mean()
                row["max_abs_diff"] = float((ac - bc).abs().max())
                row["rank_moves"] = int(((vp.ordinal_ranks(a) - vp.ordinal_ranks(b)).abs() >= vp.RANK_MOVE).sum())
            rows.append(row)
    return {"table": pd.DataFrame(rows), "excluded": pd.DataFrame(excluded), "common": commons}


# --- Independent halves (protocol §3b), for any version ---

def _half_job(args) -> dict:
    spec, season, half, game_ids, *label = args
    run_label = label[0] if label else f"half_{season}_{half}"
    t0 = time.perf_counter()
    p, a, _ = prepared(spec)
    hp = p[(p["season"] == season) & p["game_pk"].isin(game_ids)].copy()
    folds = arsenal.assign_folds(hp, seed=vp.SEED + 10 + half)
    hp["fold"] = hp["game_pk"].map(folds).astype("int8")
    ha = a[(a["season"] == season) & a["game_pk"].isin(game_ids)].copy()
    ha["fold"] = ha["game_pk"].map(folds).astype("int8")
    support = arsenal.support_table(hp)
    marked = arsenal.mark_arsenal(hp, arsenal.eligible_sets(support))
    run = (spec.pipeline or pilot_pipeline)(marked, ha, support, spec, (season,), role_fitter(spec), run_label, MODELS_DIR,
                                            keep_solutions=False)
    return {"season": season, "half": half, "scores": run.scores, "seconds": time.perf_counter() - t0}


def independent_halves(spec: Spec, workers: int = 8) -> dict:
    """The whole pipeline inside each frozen random half (validation v0 halves); {(season, half): scores}."""
    halves = vp._load("v0")["halves"]
    p = base_tables()[0]
    jobs = []
    for season in spec.seasons:
        games = set(p.loc[p["season"] == season, "game_pk"].unique())
        for half in (0, 1):
            g = np.sort(halves.index[(halves == half).to_numpy() & halves.index.isin(games)])
            jobs.append((spec, season, half, g))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        return {(r["season"], r["half"]): r for r in ex.map(_half_job, jobs)}


# --- S0: P0 reproduction ---

S0_REPORT = config.REPORTS_DIR / "research_s0.md"


def stage_s0(workers: int = 8):
    t0 = time.perf_counter()
    man = run_model(P0, out_name="P0_repro")
    print(f"P0 reproduction run done ({man['seconds']['total']:.0f} s)", flush=True)
    checks = []
    stored, repro = pd.read_parquet(SCORES_PARQUET), load("P0_repro", "scores")
    same = stored.shape == repro.shape and list(stored.columns) == list(repro.columns)
    diff_cols = [] if not same else [c for c in stored if not stored[c].equals(repro[c])]
    checks.append({"check": "scores (all columns, all rows, exact)", "rows": len(repro),
                   "exact": same and not diff_cols, "detail": ", ".join(diff_cols) or "identical"})

    cal = pd.read_csv(config.REPORTS_DIR / "phase1_calibration.csv")
    pitches = base_tables()[0]
    for kind, col in (("outcome", "log_loss"), ("contact", None)):
        h = load("P0_repro", f"heldout_{kind}")
        for season, g in h.groupby("season"):
            ll = -_logp(g).mean()
            if col is not None:
                ref = float(cal.loc[cal["season"] == season, col].iloc[0])
                checks.append({"check": f"held-out {kind} log loss {season} vs reports/phase1_calibration.csv",
                               "rows": len(g), "exact": ll == ref, "detail": f"{ll!r} vs {ref!r}"})
    fwd = vp._load("v2")["forward"]
    for kind in ("outcome", "contact"):
        f = load("P0_repro", f"forward_{kind}")
        ll = -_logp(f).mean()
        ref = float(fwd.loc[(fwd["model"] == kind) & (fwd["subset"] == "all"), "log_loss"].iloc[0])
        checks.append({"check": f"forward 2024 -> 2025 {kind} log loss vs validation v2", "rows": len(f),
                       "exact": ll == ref, "detail": f"{ll!r} vs {ref!r}"})

    t = time.perf_counter()
    halves = independent_halves(P0, workers)
    t_halves = time.perf_counter() - t
    stored_h = vp._load("half_runs")
    for key in sorted(halves):
        a, b = halves[key]["scores"], stored_h[key]["scores"]
        ok = a.shape == b.shape and all(a[c].equals(b[c]) for c in b.columns)
        checks.append({"check": f"independent half {key[0]} / {key[1]} scores vs validation half_runs",
                       "rows": len(a), "exact": ok, "detail": "identical" if ok else "differs"})
    res = {"checks": pd.DataFrame(checks), "manifest": man, "halves_seconds": t_halves,
           "total_seconds": time.perf_counter() - t0}
    S0_REPORT.write_text(s0_report(res))
    print(f"stage s0 done; all exact: {bool(res['checks']['exact'].all())}", flush=True)
    return res


def s0_report(res) -> str:
    man = res["manifest"]
    ok = bool(res["checks"]["exact"].all())
    out = ["# Research program S0: thin comparison runner and P0 reproduction", "",
           "Generated by `python3.12 -m gamecalling.research s0`. The runner (`gamecalling/research.py`) calls the "
           "working pipeline through default-preserving hooks (`scoring.run_pilot(fit_role=..., fit_policy=...)`). "
           "P0's stored outputs were read, not rewritten; the reproduction was written to `data/models/P0_repro/`.",
           "", f"**Result: {'P0 reproduces exactly' if ok else 'P0 does NOT reproduce exactly; fix before S1'}.**",
           "", _md(res["checks"].assign(exact=res["checks"]["exact"].map({True: "yes", False: "NO"}))), "",
           "## Timing (runtime budget: 60 min per full run at 8 workers)", "",
           f"- Full P0 run (fit, solve, score, held-out and forward predictions): {man['seconds']['total']:.0f} s "
           f"(pipeline {man['seconds']['pipeline']:.0f} s).",
           f"- Independent halves (4 whole-pipeline runs, 8 workers): {res['halves_seconds']:.0f} s.", "",
           "## Manifest", "",
           f"- Code: git HEAD `{man['code']['git_head'][:7]}`, uncommitted files "
           f"{len(man['code']['git_dirty_files'])}, `gamecalling/*.py` sha256 `{man['code']['gamecalling_sha256'][:12]}`.",
           "- Inputs: " + "; ".join(f"`{k}` {v['sha256'][:12]}" for k, v in man["inputs"].items()) + ".", ""]
    return "\n".join(out)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "s0"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    {"s0": stage_s0}[stage](workers)
