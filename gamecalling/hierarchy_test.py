"""Isolated test of the pooling hierarchy (research_notes/uncertainty_design/05_hierarchy_test.md, revision 1).

Counts only: no catchers, solver or rankings. Template: the real 2024 pilot training cells of the outcome
and contact models, each with its real pitch count N_c (all real weights are 1).

- Fitter: the production fit (models.fit_pooled) on count tables, checked to reproduce it on the real data;
  plus an oracle cell-alpha fit around the generating league means.
- Step 1, recovery: latent cells p_c ~ Dirichlet(alpha0 m_c) around the real league means, counts
  n_c ~ Multinomial(N_c, p_c), fit. alpha0 = the real cell alpha and 4x that.
- Steps 2-3: from each fit, new latent cells by one of three constructions, new counts with the same N_c,
  refit, for N_GEN generations: A fixed posterior means; B Dirichlet(alpha_hat m_hat) draws (population);
  C Dirichlet(n + alpha_hat m_hat) draws (posterior given the previous counts).

Run with: python3.12 -m gamecalling.hierarchy_test [WORKERS]
Outputs: data/hierarchy_test/v1/ (template, one file per task, manifest) and reports/hierarchy_test.md.
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

from gamecalling import config, models, pooling, simulation as S
from gamecalling.phase1 import _md

VERSION = "v1"
OUT = config.DATA_DIR / "hierarchy_test" / VERSION
REPORT = config.REPORTS_DIR / "hierarchy_test.md"
SEED = 20261029
MODELS = ("outcome", "contact")
A0_MULT = (1.0, 4.0)        # alpha0 = the real fit's cell alpha times these
N_REP = 50
N_GEN = 3
CONSTRUCTIONS = ("A_fixed_pm", "B_population", "C_posterior")
RECOVERY = (0.9, 1.1)       # step 1 reading
STABLE = (0.85, 1.15)       # steps 2-3: B and C do not flatten
FLATTEN = 1.5               # steps 2-3: A flattens
ALPHA_MAX = float(np.exp(pooling.LOG_ALPHA_BOUNDS[1]))

SMOKE_DIR = os.environ.get("HIERARCHY_TEST_SMOKE")
if SMOKE_DIR:
    OUT, REPORT = Path(SMOKE_DIR), Path(SMOKE_DIR) / "report.md"
    N_REP, N_GEN = 2, 2


def manifest() -> dict:
    return {"version": VERSION, "seed": SEED, "models": list(MODELS), "a0_mult": list(A0_MULT), "n_rep": N_REP,
            "n_gen": N_GEN, "constructions": list(CONSTRUCTIONS), "season": S.SEASON,
            "design": "research_notes/uncertainty_design/05_hierarchy_test.md revision 1"}


# --- Template and fitter ---

def _template(model: str) -> dict:
    S._load("pilot")
    p = S._CACHE["pitches"]
    p = p[p["season"] == S.SEASON]
    tr, levels, K = {"outcome": (models.outcome_training(p), models.OUTCOME_LEVELS, models.N_OUTCOMES),
                     "contact": (models.contact_training(p), models.CONTACT_LEVELS, models.N_BINS)}[model]
    n = models._counts(tr, levels[2], K)
    keys = n.index.to_frame(index=False)
    league = keys.groupby(list(levels[1]), sort=True).ngroup().to_numpy()
    lk = keys[list(levels[1])].drop_duplicates().sort_values(list(levels[1])).reset_index(drop=True)
    base = lk.groupby(list(levels[0]), sort=True).ngroup().to_numpy()
    fit = models.fit_pooled(tr, levels, K)
    return {"model": model, "K": K, "n": n.to_numpy(), "league": league, "base": base,
            "m": fit.league_mean(keys), "pred": fit.predict(keys), "alphas": (fit.alpha_league, fit.alpha_cell)}


def fit_counts(n: np.ndarray, league: np.ndarray, base: np.ndarray, K: int, m_oracle=None) -> dict:
    """models.fit_pooled on a cell count table (cells x K): league cell of each cell, base cell of each league."""
    n_l = np.stack([np.bincount(league, n[:, k], minlength=len(base)) for k in range(K)], 1)
    n_b = np.stack([np.bincount(base, n_l[:, k]) for k in range(K)], 1)
    b = (n_b + models.BASE_PSEUDO) / (n_b.sum(1, keepdims=True) + K * models.BASE_PSEUDO)
    keep = n_l.sum(1) > 0
    a1 = pooling.fit_alpha(n_l[keep], b[base][keep])
    m = pooling.posterior_mean(n_l, b[base], a1)[league]
    a2 = pooling.fit_alpha(n, m)
    out = {"alpha_league": a1, "alpha_cell": a2, "m": m, "pm": pooling.posterior_mean(n, m, a2)}
    if m_oracle is not None:
        out["alpha_cell_oracle"] = pooling.fit_alpha(n, m_oracle)
    return out


def _rms(x: np.ndarray, m: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.average(((x - m) ** 2).sum(1), weights=w)))


def _implied(m: np.ndarray, alpha: float, w: np.ndarray) -> float:
    """RMS distance of Dirichlet(alpha m) draws from m: sqrt(sum_k m_k (1 - m_k) / (alpha + 1))."""
    return float(np.sqrt(np.average((m * (1 - m)).sum(1) / (alpha + 1), weights=w)))


def _dirichlet(rng, shape: np.ndarray) -> np.ndarray:
    g = rng.gamma(shape)
    return g / g.sum(1, keepdims=True)


def _counts(rng, N: np.ndarray, p: np.ndarray) -> np.ndarray:
    return rng.multinomial(N.astype(np.int64), p / p.sum(1, keepdims=True)).astype(float)


# --- Tasks ---

_TEMPLATE = {}


def _template_path(model: str) -> Path:
    return OUT / f"template_{model}.pkl"


def _load_template(model: str) -> dict:
    if model not in _TEMPLATE:
        with open(_template_path(model), "rb") as f:
            _TEMPLATE[model] = pickle.load(f)
    return _TEMPLATE[model]


def _task_path(task: tuple) -> Path:
    return OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl")


def run_task(task: tuple) -> str:
    model, ai, rep = task
    path = _task_path(task)
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    T = _load_template(model)
    N, league, base, K = T["n"].sum(1), T["league"], T["base"], T["K"]
    mi = MODELS.index(model)
    a0 = A0_MULT[ai] * T["alphas"][1]

    def record(construction, gen, p_lat, m_gen, f):
        return {"model": model, "a0_mult": A0_MULT[ai], "a0": a0, "rep": rep, "construction": construction,
                "gen": gen, "alpha_league": f["alpha_league"], "alpha_cell": f["alpha_cell"],
                "alpha_cell_oracle": f["alpha_cell_oracle"], "latent_spread": _rms(p_lat, m_gen, N),
                "implied_spread": _implied(f["m"], f["alpha_cell"], N), "pm_spread": _rms(f["pm"], f["m"], N)}

    rng = np.random.default_rng([SEED, mi, ai, rep])
    p0 = _dirichlet(rng, a0 * T["m"])
    n0 = _counts(rng, N, p0)
    f0 = fit_counts(n0, league, base, K, T["m"])
    rows = [record("truth", 0, p0, T["m"], f0)]
    for ci, c in enumerate(CONSTRUCTIONS):
        f, n = f0, n0
        for g in range(1, N_GEN + 1):
            rng = np.random.default_rng([SEED, mi, ai, rep, 1 + ci, g])
            if c == "A_fixed_pm":
                p = f["pm"]
            elif c == "B_population":
                p = _dirichlet(rng, f["alpha_cell"] * f["m"])
            else:
                p = _dirichlet(rng, n + f["alpha_cell"] * f["m"])
            m_gen = f["m"]
            n = _counts(rng, N, p)
            f = fit_counts(n, league, base, K, m_gen)
            rows.append(record(c, g, p, m_gen, f))
    res = {"rows": rows, "seconds": time.perf_counter() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return str(path)


def tasks() -> list:
    return [(m, ai, r) for m in MODELS for ai in range(len(A0_MULT)) for r in range(N_REP)]


def prepare() -> pd.DataFrame:
    """Templates, and the check that fit_counts reproduces models.fit_pooled on the real data."""
    rows = []
    for model in MODELS:
        if not _template_path(model).exists():
            T = _template(model)
            with open(_template_path(model), "wb") as f:
                pickle.dump(T, f)
        T = _load_template(model)
        f = fit_counts(T["n"], T["league"], T["base"], T["K"])
        rows.append({"model": model, "cells": len(T["n"]), "leagues": len(T["base"]), "K": T["K"],
                     "pitches": int(T["n"].sum()), "median_N": float(np.median(T["n"].sum(1))),
                     "alpha_league": T["alphas"][0], "alpha_cell": T["alphas"][1],
                     "alpha_league_diff": abs(f["alpha_league"] - T["alphas"][0]),
                     "alpha_cell_diff": abs(f["alpha_cell"] - T["alphas"][1]),
                     "pred_max_diff": float(np.abs(f["pm"] - T["pred"]).max()),
                     "real_pm_spread": _rms(T["pred"], T["m"], T["n"].sum(1)),
                     "real_implied_spread": _implied(T["m"], T["alphas"][1], T["n"].sum(1))})
    return pd.DataFrame(rows)


# --- Summary ---

def _rows() -> pd.DataFrame:
    out = []
    for t in tasks():
        with open(_task_path(t), "rb") as f:
            out += pickle.load(f)["rows"]
    R = pd.DataFrame(out)
    key = ["model", "a0_mult", "rep"]
    first = R[R["gen"] == 0][key + ["alpha_cell"]].rename(columns={"alpha_cell": "fit0_alpha"})
    prev = R[R["gen"] > 0][key + ["construction", "gen", "alpha_cell"]].assign(gen=lambda d: d["gen"] + 1)
    R = R.merge(first, on=key).merge(prev.rename(columns={"alpha_cell": "prev_alpha"}),
                                     on=key + ["construction", "gen"], how="left")
    R["prev_alpha"] = R["prev_alpha"].where(R["gen"] != 1, R["fit0_alpha"])
    R["ratio_prev"] = R["alpha_cell"] / R["prev_alpha"]
    R["ratio_a0"] = R["alpha_cell"] / R["a0"]
    return R


def _iqr(x) -> str:
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return f"{q[1]:.3g} [{q[0]:.3g}, {q[2]:.3g}]"


def summarize(check: pd.DataFrame) -> str:
    R = _rows()
    lines = ["# Isolated test of the pooling hierarchy", "",
             "Spec: `research_notes/uncertainty_design/05_hierarchy_test.md` (revision 1). Counts only, no catchers, "
             "solver or rankings; real 2024 pilot training cells with their real pitch counts N_c. Spreads are "
             "N_c-weighted RMS distances (Euclidean over categories) from the league means: **latent** (the cell "
             "probabilities that generated the data, around the generating league means), **implied** "
             "(sqrt(sum_k m(1-m)/(alpha+1)) at the fitted alpha and league means), **posterior-mean** (the fitted "
             "cells). Medians with [quartiles] over "
             f"{N_REP} replicates. Simulation only; nothing here is a correction.", "",
             "## Fitter check (count-level fit against models.fit_pooled on real 2024 data)", "",
             _md(check, "{:.4g}"), ""]
    g0 = R[R["gen"] == 0]
    s1 = []
    for (model, a0m), d in g0.groupby(["model", "a0_mult"]):
        s1.append({"model": model, "a0": d["a0"].iloc[0], "alpha_hat/a0": _iqr(d["ratio_a0"]),
                   "oracle/a0": _iqr(d["alpha_cell_oracle"] / d["a0"]),
                   "implied/latent": _iqr(d["implied_spread"] / d["latent_spread"]),
                   "pm/latent": _iqr(d["pm_spread"] / d["latent_spread"]),
                   "latent": float(d["latent_spread"].median()), "league_alpha": float(d["alpha_league"].median())})
    s1 = pd.DataFrame(s1)
    ok1 = all(RECOVERY[0] <= float(np.median(d[col])) <= RECOVERY[1]
              for _, d in g0.assign(r1=g0["ratio_a0"], r2=g0["implied_spread"] / g0["latent_spread"])
              .groupby(["model", "a0_mult"]) for col in ("r1", "r2"))
    lines += ["## Step 1: recovery from a known Dirichlet hierarchy", "", _md(s1, "{:.4g}"), "",
              f"Pre-stated reading: recovery holds if the median alpha_hat/a0 and implied/latent lie in "
              f"[{RECOVERY[0]}, {RECOVERY[1]}] for both models and both a0. **{'Holds' if ok1 else 'Does not hold'}.**",
              ""]
    s2 = []
    for (model, a0m, c, g), d in R[R["gen"] > 0].groupby(["model", "a0_mult", "construction", "gen"]):
        s2.append({"model": model, "a0_mult": a0m, "construction": c, "gen": g,
                   "alpha_cell": float(d["alpha_cell"].median()), "ratio_prev": _iqr(d["ratio_prev"]),
                   "cum_ratio": float((d["alpha_cell"] / d["fit0_alpha"]).median()),
                   "ratio_a0": float(d["ratio_a0"].median()), "oracle_ratio_a0": float(
                       (d["alpha_cell_oracle"] / d["a0"]).median()),
                   "latent": float(d["latent_spread"].median()), "pm": float(d["pm_spread"].median()),
                   "at_bound": float((d["alpha_cell"] > 0.99 * ALPHA_MAX).mean())})
    s2 = pd.DataFrame(s2)
    verdict = []
    for (model, a0m), d in R[R["gen"] > 0].groupby(["model", "a0_mult"]):
        med = d.groupby(["construction", "gen"])["ratio_prev"].median()
        last = d[d["gen"] == N_GEN]
        cum = (last["alpha_cell"] / last["fit0_alpha"]).groupby(last["construction"]).median()
        a_flat = med["A_fixed_pm", 1] >= FLATTEN and med["A_fixed_pm", N_GEN] > 1
        bc = {c: bool(all(STABLE[0] <= med[c, g] <= STABLE[1] for g in range(1, N_GEN + 1)))
              for c in CONSTRUCTIONS[1:]}
        verdict.append({"model": model, "a0_mult": a0m, "A_flattens_and_repeats": a_flat,
                        "A_ratio_gen1": med["A_fixed_pm", 1], f"A_ratio_gen{N_GEN}": med["A_fixed_pm", N_GEN],
                        "B_per_gen_rule": bc["B_population"], f"B_cum_gen{N_GEN}": cum["B_population"],
                        "C_per_gen_rule": bc["C_posterior"], f"C_cum_gen{N_GEN}": cum["C_posterior"]})
    lines += ["## Steps 2-3: second datasets by three constructions, repeated", "",
              "`gen` g refits data generated from the fit to generation g-1 (gen 0 = step 1). `ratio_prev`: fitted "
              "cell alpha over the previous generation's. `ratio_a0`: over the alpha that generated step 1. "
              "`cum_ratio`: over the step-1 fit's (cumulative drift). `oracle_ratio_a0`: cell alpha fitted around the generating league means. `at_bound`: share of fits "
              f"at the upper alpha bound ({ALPHA_MAX:.0e}), where the ratios are truncated.", "",
              _md(s2, "{:.4g}"), "",
              f"Pre-stated reading: A reproduces the mechanism if the median ratio_prev >= {FLATTEN} in gen 1 and "
              f"> 1 in gen {N_GEN}; B and C do not flatten if the median ratio_prev lies in [{STABLE[0]}, "
              f"{STABLE[1]}] in every generation. That rule is per generation: passing it is not stability over "
              f"repeated generations, so the cumulative ratio at gen {N_GEN} is reported beside it.", "",
              _md(pd.DataFrame(verdict), "{:.3g}"), ""]
    return "\n".join(lines)


def main(workers: int = 8):
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.json"
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest())):
        raise ValueError(f"{man} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(), indent=1))
    t0 = time.perf_counter()
    check = prepare()
    print(check.to_string(), flush=True)
    items = tasks()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        for i, _ in enumerate(ex.map(run_task, items), 1):
            if i % 20 == 0 or i == len(items):
                print(f"hierarchy tasks: {i}/{len(items)} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    REPORT.write_text(summarize(check))
    print(f"hierarchy test done in {time.perf_counter() - t0:.0f} s; report in {REPORT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
