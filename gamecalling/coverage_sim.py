"""Nested coverage experiment for the bootstrap intervals (small first pass).

For each simulated dataset (simulation.py v2 world, same fixed catcher effects
as the recovery runs, independent random streams), fit the complete estimator,
then build its game-bootstrap percentile intervals with both bootstrap methods
("plain" reference and "firstcopy" experimental, same draws), and check whether
each qualifying catcher's interval covers the target.

- Settings: identical calling (every target is 0) and a small non-zero setting
  (scientific SD 0.01 runs per 100 pitches, near the observed 0.006-0.013), in
  both data-generating models.
- Effects, interval endpoints and targets are centered on the mean over
  qualifying catchers within each dataset or bootstrap replicate.
- Targets: scientific (exact) and the D/A population targets from the recovery
  runs (mean of conditional targets over replicates). For identical calling all
  are 0.
- Conditional on the real payoff estimates: the bootstrap resamples simulated
  games for the pitch data only; payoffs stay at the real-PA estimates in both
  the truth and the estimator. Payoff uncertainty is excluded on both sides.

Run with: python3.12 -m gamecalling.coverage_sim N_DATASETS N_BOOT [WORKERS]
Outputs: data/coverage/v1/.
"""

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from gamecalling import config, simulation as S

VERSION = "v1"
COV_DIR = config.DATA_DIR / "coverage" / VERSION
SETTINGS = (0.0, 0.01)
METHODS = ("plain", "firstcopy")
COV_SEED = 777


def _effects(fits, catchers, qualifying) -> dict:
    out = {}
    for col, f in fits.items():
        e = f.catchers.set_index("catcher")["effect"].reindex(catchers).to_numpy()
        out[col] = e - np.nanmean(e[qualifying])
    return out


def _weighted(sim: pd.DataFrame, rng, method: str) -> pd.DataFrame:
    games = np.sort(sim["game_pk"].unique())
    m = np.bincount(rng.choice(len(games), size=len(games), replace=True), minlength=len(games))
    w = pd.Series(m, index=games)
    w = w[w > 0]
    out = sim[sim["game_pk"].isin(w.index)].copy()
    out["w"] = out["game_pk"].map(w).astype(float)
    if method == "firstcopy":
        out["w_hyper"] = 1.0
    return out.reset_index(drop=True)


def run_dataset(dgp: str, sd: float, rep: int, n_boot: int) -> pd.DataFrame:
    t0 = time.perf_counter()
    pas, world, common, opps = S._load(dgp)
    catchers, qualifying, z, kappa, scale = S.catcher_effects(opps, S._CACHE["pitches"], dgp, sd)
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    rng = np.random.default_rng([S.SIM_SEED, COV_SEED, S.DGPS.index(dgp), int(round(sd * 1000)), rep])
    sim = S.simulate(world, opps, kappa[opps["start_catcher"].map(code).to_numpy()], rng)
    _, fits = S.estimate(sim, pas)
    point = _effects(fits, catchers, qualifying)
    boots = {(m, col): [] for m in METHODS for col in ("D", "A")}
    for b in range(n_boot):
        for m in METHODS:
            brng = np.random.default_rng([S.SIM_SEED, COV_SEED, S.DGPS.index(dgp), int(round(sd * 1000)), rep, b])
            _, bf = S.estimate(_weighted(sim, brng, m), pas)
            for col, e in _effects(bf, catchers, qualifying).items():
                boots[m, col].append(e)
    out = pd.DataFrame({"catcher": catchers, "qualifying": qualifying, "kappa": kappa,
                        "scientific": S.scientific(common, kappa)})
    out["scientific"] -= out.loc[qualifying, "scientific"].mean()
    for col in ("D", "A"):
        out[f"{col}_point"] = point[col]
        for m in METHODS:
            B = np.vstack(boots[m, col])
            out[f"{col}_{m}_lo"] = np.nanquantile(B, 0.025, axis=0)
            out[f"{col}_{m}_hi"] = np.nanquantile(B, 0.975, axis=0)
            out[f"{col}_{m}_boot_mean"] = np.nanmean(B, axis=0)
            out[f"{col}_{m}_boot_se"] = np.nanstd(B, axis=0, ddof=1)
    out = out.assign(dgp=dgp, target_sd_setting=sd, rep=rep, n_boot=n_boot, seconds=time.perf_counter() - t0)
    return out


def _task(args):
    dgp, sd, rep, n_boot = args
    path = COV_DIR / f"cov_{dgp}_{int(round(sd * 1000)):03d}_{rep:03d}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    res = run_dataset(dgp, sd, rep, n_boot)
    res.to_pickle(path)
    return res


def manifest(n_boot) -> dict:
    return {"version": VERSION, "simulation_version": S.VERSION, "seed": [S.SIM_SEED, COV_SEED], "settings": list(SETTINGS),
            "methods": list(METHODS), "n_boot": n_boot, "intervals": "percentile 2.5-97.5, centered on qualifying mean",
            "payoffs": "conditional on real PA payoff estimates (not resampled)"}


def run_all(n_datasets: int, n_boot: int, workers: int = 8):
    COV_DIR.mkdir(parents=True, exist_ok=True)
    man = COV_DIR / "manifest.json"
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest(n_boot))):
        raise ValueError(f"{COV_DIR} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(n_boot), indent=1))
    tasks = [(d, sd, r, n_boot) for r in range(n_datasets) for d in S.DGPS for sd in SETTINGS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_task, tasks)), ignore_index=True)


def load_results() -> pd.DataFrame:
    files = sorted(COV_DIR.glob("cov_*.pkl"))
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def summarize(res: pd.DataFrame, pop: pd.DataFrame | None) -> pd.DataFrame:
    """Coverage by (DGP, setting, score, method, target), with Monte Carlo uncertainty.

    pop: simulation.summarize population targets (D_pop, A_pop) for the non-zero setting.
    The MC SE treats datasets as the independent unit (catchers within a dataset share it).
    """
    q = res[res["qualifying"]].copy()
    if pop is not None and len(pop):
        q = q.merge(pop[["dgp", "target_sd_setting", "catcher", "D_pop", "A_pop"]],
                    on=["dgp", "target_sd_setting", "catcher"], how="left")
    rows = []
    for (dgp, sd), g in q.groupby(["dgp", "target_sd_setting"]):
        for col in ("D", "A"):
            targets = {"zero": np.zeros(len(g))} if sd == 0 else {}
            if sd != 0:
                targets["scientific"] = g["scientific"].to_numpy()
                if f"{col}_pop" in g and g[f"{col}_pop"].notna().all():
                    targets["own population"] = (g[f"{col}_pop"] - g.groupby("rep")[f"{col}_pop"].transform("mean")).to_numpy()
            for m in METHODS:
                lo, hi = g[f"{col}_{m}_lo"].to_numpy(), g[f"{col}_{m}_hi"].to_numpy()
                for tname, tgt in targets.items():
                    cov = (lo <= tgt) & (tgt <= hi)
                    per = pd.Series(cov).groupby(g["rep"].to_numpy()).mean()
                    rows.append(dict(dgp=dgp, target_sd_setting=sd, score=col, method=m, target=tname,
                                     datasets=len(per), intervals=len(cov), coverage=cov.mean(),
                                     coverage_mcse=per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan,
                                     excludes_zero=((lo > 0) | (hi < 0)).mean(),
                                     median_width=np.median(hi - lo),
                                     boot_se_over_sd=np.nan))
            # Calibration of the bootstrap SE: mean bootstrap SE vs. SD of point estimates across datasets.
            sd_point = g.groupby("catcher")[f"{col}_point"].std(ddof=1).mean()
            for m in METHODS:
                se = g[f"{col}_{m}_boot_se"].mean()
                for r in rows[-len(targets) * len(METHODS):]:
                    if r["method"] == m and r["score"] == col:
                        r["boot_se_over_sd"] = se / sd_point if sd_point > 0 else np.nan
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n, nb = int(sys.argv[1]), int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    t0 = time.perf_counter()
    r = run_all(n, nb, workers)
    print(f"{r[['dgp', 'target_sd_setting', 'rep']].drop_duplicates().shape[0]} datasets in {time.perf_counter() - t0:.0f} s")
