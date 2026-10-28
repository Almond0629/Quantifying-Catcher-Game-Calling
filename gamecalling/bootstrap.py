"""Paired whole-pipeline game bootstrap (project_plan.md, Phase 2 uncertainty; addendum).

Each replicate resamples games with replacement within each season and reruns
the whole pipeline: arsenal support and eligible sets, role fits (probabilities,
pooling strengths, payoffs, policy), solves, scored differences and the ranking
models. D, A and B come from the same replicate, so their comparison is paired.
Fold membership is fixed by original game ID.

A game drawn m times is represented by its rows with integer weight w = m
(weighting.py); `resample_season` builds the equivalent m explicit copies, and
the two give identical fits (tests/test_phase23.py).

Methods (outputs are versioned by method so different procedures never mix):
- "plain" (reference): the unmodified pipeline on the resampled data.
- "firstcopy" (experimental): hyperparameters (pooling concentrations, ranking
  variance components) are estimated with weight 1 per drawn game (w_hyper = 1);
  everything else uses w. This changes the estimator inside the replicate.

Run with: python3.12 -m gamecalling.bootstrap METHOD N_REPLICATES [WORKERS] [FIRST_REPLICATE]
Replicates go to data/bootstrap/<METHOD>_w_v2/rep_NNNN.parquet with a manifest.json.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, ranking, scoring

BOOT_ROOT = config.DATA_DIR / "bootstrap"
BOOT_SEED = 20261028
SCORES = ("D", "A", "B")
MODELS = ("random", "fixed")
METHODS = ("plain", "firstcopy")
VERSION = "w_v2"


def boot_dir(method: str):
    return BOOT_ROOT / f"{method}_{VERSION}"


def _draw(p: pd.DataFrame, rng) -> tuple[np.ndarray, np.ndarray]:
    games = np.sort(p["game_pk"].unique())
    m = np.bincount(rng.choice(len(games), size=len(games), replace=True), minlength=len(games))
    return games, m


def resample_season(pitches: pd.DataFrame, pas: pd.DataFrame, season: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One season's games drawn with replacement, as explicit copies (new IDs game_pk * 100 + copy)."""
    p = pitches[pitches["season"] == season]
    a = pas[pas["season"] == season]
    games, m = _draw(p, rng)
    draw = np.repeat(games, m)
    copy = pd.Series(draw).groupby(draw).cumcount().to_numpy()
    mapping = pd.DataFrame({"game_pk": draw, "new_game": draw * 100 + copy})

    def remap(df):
        out = df.merge(mapping, on="game_pk")
        out["game_pk"] = out.pop("new_game")
        out["pa_id"] = out["game_pk"] * 1000 + out["at_bat_number"]
        return out

    pb = remap(p).sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    ab = remap(a).sort_values(["game_pk", "at_bat_number"], kind="stable").reset_index(drop=True)
    return pb, ab


def resample_season_weights(pitches: pd.DataFrame, pas: pd.DataFrame, season: int, rng,
                            method: str = "plain") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same draw as resample_season (same rng state), as weights on the original rows."""
    p = pitches[pitches["season"] == season]
    a = pas[pas["season"] == season]
    games, m = _draw(p, rng)
    wmap = pd.Series(m, index=games)
    keep = wmap[wmap > 0]

    def weigh(df):
        out = df[df["game_pk"].isin(keep.index)].copy()
        out["w"] = out["game_pk"].map(keep).astype(float)
        if method == "firstcopy":
            out["w_hyper"] = 1.0
        return out.reset_index(drop=True)

    return weigh(p), weigh(a)


def run_on(pitches: pd.DataFrame, pas: pd.DataFrame, seasons=config.SEASONS) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Whole pipeline from Phase 0 tables (with folds): eligibility, scoring, ranking fits.

    Returns the pitch scores, the fits and a long table of catcher effects.
    """
    support = arsenal.support_table(pitches)
    marked = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(support))
    run = scoring.run_pilot(marked, pas, support, seasons=seasons)
    fits = ranking.fit_scores(run.scores, SCORES, MODELS)
    return run.scores, fits, effects_table(fits)


def effects_table(fits: dict) -> pd.DataFrame:
    rows = []
    for (season, score, model), f in fits.items():
        c = f.catchers.assign(season=season, score=score, model=model, sd_catcher=np.sqrt(f.variances["catcher"]),
                              sd_pitcher=np.sqrt(f.variances.get("pitcher", np.nan)), sd_resid=np.sqrt(f.sigma2),
                              converged=f.converged)
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


_TABLES = {}


def _load():
    if not _TABLES:
        _TABLES["pitches"] = pd.read_parquet(config.PITCHES_PARQUET)
        _TABLES["pas"] = pd.read_parquet(config.PAS_PARQUET)
    return _TABLES["pitches"], _TABLES["pas"]


def replicate(rep: int, method: str = "plain", seed: int = BOOT_SEED) -> dict:
    """Run and save one replicate. Its draw depends only on (seed, rep), so methods share draws."""
    t0 = time.perf_counter()
    pitches, pas = _load()
    rng = np.random.default_rng([seed, rep])
    parts = [resample_season_weights(pitches, pas, s, rng, method) for s in config.SEASONS]
    pb = pd.concat([p for p, _ in parts], ignore_index=True)
    ab = pd.concat([a for _, a in parts], ignore_index=True)
    _, _, eff = run_on(pb, ab)
    eff.assign(replicate=rep).to_parquet(boot_dir(method) / f"rep_{rep:04d}.parquet", index=False)
    return {"replicate": rep, "seconds": time.perf_counter() - t0}


def _manifest(method: str, seed: int) -> dict:
    return {"method": method, "version": VERSION, "seed": seed, "representation": "integer game weights",
            "min_contact_cell_n": config.MIN_CONTACT_CELL_N, "solver_tol": config.SOLVER_TOL,
            "arsenal_rules": config.ARSENAL_RULES, "n_folds": config.N_FOLDS, "fold_seed": config.FOLD_SEED}


def run_replicates(method: str, reps, workers: int = 5, seed: int = BOOT_SEED) -> pd.DataFrame:
    d = boot_dir(method)
    d.mkdir(parents=True, exist_ok=True)
    man = d / "manifest.json"
    if man.exists():
        if json.loads(man.read_text()) != json.loads(json.dumps(_manifest(method, seed))):
            raise ValueError(f"{d} was made with a different configuration; use a new version")
    else:
        man.write_text(json.dumps(_manifest(method, seed), indent=1))
    t0 = time.perf_counter()
    todo = [r for r in reps if not (d / f"rep_{r:04d}.parquet").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        timings = list(ex.map(replicate, todo, [method] * len(todo), [seed] * len(todo)))
    out = pd.DataFrame(timings)
    out.attrs["wall_seconds"] = time.perf_counter() - t0
    return out


def load_replicates(method: str = "plain") -> pd.DataFrame:
    files = sorted(boot_dir(method).glob("rep_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def intervals(point: pd.DataFrame, boot: pd.DataFrame, level=0.95) -> pd.DataFrame:
    """Percentile intervals of each catcher effect, plus the paired D - A change.

    point: effects_table of the original data. Catchers missing from a replicate
    (none of their games drawn) are left out of that replicate.
    """
    key = ["season", "score", "model", "catcher"]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    q = boot.groupby(key)["effect"].quantile([lo, hi]).unstack()
    q.columns = ["lo", "hi"]
    q["n_reps"] = boot.groupby(key).size()
    q["boot_se"] = boot.groupby(key)["effect"].std()
    q["boot_mean"] = boot.groupby(key)["effect"].mean()
    out = point.merge(q.reset_index(), on=key, how="left")

    wide = boot[boot["score"].isin(["D", "A"])].pivot_table(index=["season", "model", "catcher", "replicate"],
                                                             columns="score", values="effect")
    delta = (wide["D"] - wide["A"]).rename("d_minus_a").reset_index()
    dq = delta.groupby(["season", "model", "catcher"])["d_minus_a"].quantile([lo, hi]).unstack()
    dq.columns = ["d_minus_a_lo", "d_minus_a_hi"]
    pw = point[point["score"].isin(["D", "A"])].pivot_table(index=["season", "model", "catcher"], columns="score",
                                                            values="effect")
    dq["d_minus_a"] = pw["D"] - pw["A"]
    return out.merge(dq.reset_index(), on=["season", "model", "catcher"], how="left")


if __name__ == "__main__":
    method = sys.argv[1]
    if method not in METHODS:
        raise SystemExit(f"method must be one of {METHODS}")
    n = int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    t = run_replicates(method, range(first, first + n), workers)
    if len(t):
        t.assign(wall_seconds=t.attrs["wall_seconds"], workers=workers).to_csv(
            boot_dir(method) / f"timings_{first:04d}_{first + n - 1:04d}.csv", index=False)
        print(f"{len(t)} {method} replicates in {t.attrs['wall_seconds']:.0f} s wall; "
              f"mean {t['seconds'].mean():.0f} s per replicate per worker ({workers} workers)")
    else:
        print("nothing to do")
