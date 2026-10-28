"""Debugging stage for the uncertainty design notes (research_notes/uncertainty_design/, revision 4).

Simulation only, fixed payoffs (the real PA payoff estimates), season 2024. Nothing here
touches real catcher results. Truth is the v2 pilot world (simulation.py).

Outer datasets: one at scientific SD 0 and one at 0.01, from the true world with the v2
catcher effects (z fixed). For each, the world W_hat is refit from the outer dataset
(simulation.build_world on the simulated pitches), as the real analysis would do.

Tasks (each one whole-pipeline fit, about 48 s):
- h0 (R_H per outer): data from W_hat with every catcher at pi0; the conditional targets of a
  kappa grid (plus midpoints) under the fitted models give one draw of h0(kappa).
- h0true (R_H): the same in the true world, so the fitted-world h0 can be compared with it.
- hC (R_H per configuration, outer 0.01 only): training data from W_hat with the v2 z
  calibrated to scientific SD 0.01 and 0.02, giving h_C(kappa) for comparison with h0.
- design (M_DESIGN per outer): sigma ~ Uniform(0, SIGMA_MAX), fresh normal z, data from W_hat;
  records every catcher's effect, its kappa, its conditional target and the fitted catcher SDs.
- check (N_CHECK per DGP): the true pilot and batter worlds at scientific SD 0.05, fresh z;
  records the fitted catcher SDs (spec 3: configuration effect on T).

Run with: python3.12 -m gamecalling.uncertainty_debug [WORKERS]
Outputs: data/uncertainty_debug/v1/ (outer_*.pkl, tasks/*.pkl, manifest.json) and
reports/uncertainty_debug.md. Finished tasks are skipped on rerun.
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

from gamecalling import arsenal, config, simulation as S
from gamecalling.phase1 import _md

VERSION = "v1"
OUT = config.DATA_DIR / "uncertainty_debug" / VERSION
REPORT = config.REPORTS_DIR / "uncertainty_debug.md"
SEED = 20261030
OUTER_SDS = (0.0, 0.01)
R_H = 20
HC_SDS = (0.01, 0.02)
M_DESIGN = 100
SIGMA_MAX = 0.05
CHECK_SD = 0.05
N_CHECK = 20
N_BELT_BOOT = 200
SUPPORT_H, SUPPORT_HTAU, SUPPORT_MIN = 0.0025, 0.003, 30   # note 1 §4 joint-support rule (debug values)

# A directory here runs a tiny version for testing the code (read by worker processes too).
SMOKE_DIR = os.environ.get("UNCERTAINTY_DEBUG_SMOKE")
if SMOKE_DIR:
    OUT, REPORT = Path(SMOKE_DIR), Path(SMOKE_DIR) / "report.md"
    R_H, M_DESIGN, N_CHECK, HC_SDS, N_BELT_BOOT = 2, 4, 1, (0.02,), 10


def manifest() -> dict:
    return {"version": VERSION, "simulation_version": S.VERSION, "seed": SEED, "outer_sds": list(OUTER_SDS),
            "r_h": R_H, "hc_sds": list(HC_SDS), "m_design": M_DESIGN, "sigma_max": SIGMA_MAX,
            "check_sd": CHECK_SD, "n_check": N_CHECK, "payoffs": "fixed (real PA payoff estimates)",
            "design": "research_notes/uncertainty_design revision 4, note 1 §8 and spec 3 §10"}


def _key(sd: float) -> str:
    return f"{int(round(sd * 1000)):03d}"


def kappa_eval(grid: np.ndarray) -> np.ndarray:
    """The target kappa grid followed by its midpoints (for the interpolation check)."""
    return np.concatenate([grid, (grid[:-1] + grid[1:]) / 2])


def _catchers(opps, pitches):
    catchers = np.sort(opps["start_catcher"].unique())
    real = pitches[(pitches["season"] == S.SEASON) & pitches["scorable"]]
    n_real = real.groupby("catcher").size().reindex(catchers, fill_value=0).to_numpy()
    return catchers, n_real >= S.QUALIFY


def _fit_record(fits) -> dict:
    out = {}
    for col, f in fits.items():
        out[f"{col}_sd"] = float(np.sqrt(f.variances["catcher"]))
        out[f"{col}_converged"] = bool(f.converged)
    return out


def _effects(fits, catchers) -> pd.DataFrame:
    out = pd.DataFrame({"catcher": catchers})
    for col, f in fits.items():
        c = f.catchers.set_index("catcher")
        out[f"{col}_effect"] = c["effect"].reindex(catchers).to_numpy()
        out[f"{col}_pev_se"] = c["pev_se"].reindex(catchers).to_numpy()
        out[f"{col}_n"] = c["n_pitches"].reindex(catchers).to_numpy()
    return out


# --- Stage A: outer datasets and their refit worlds ---

def prepare_outer(sd: float) -> str:
    path = OUT / f"outer_{_key(sd)}.pkl"
    if path.exists():
        return str(path)
    t0 = time.perf_counter()
    pas, world, common, opps = S._load("pilot")
    pitches = S._CACHE["pitches"]
    catchers, qualifying, z, kappa, scale = S.catcher_effects(opps, pitches, "pilot", sd)
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    rng = np.random.default_rng([SEED, 1, int(round(sd * 1000))])
    sim = S.simulate(world, opps, kappa[opps["start_catcher"].map(code).to_numpy()], rng)
    _, fits = S.estimate(sim, pas)
    w_hat = S.build_world(sim, pas, arsenal.support_table(sim), "pilot")
    opps_hat = S.opportunities(pitches, pas, w_hat.units)
    common_hat = S.build_common(w_hat, opps_hat, S.real_states(pitches, w_hat))
    cat_hat, qual_hat = _catchers(opps_hat, pitches)
    obs = _effects(fits, catchers).assign(qualifying=qualifying, kappa=kappa, z=z)
    out = {"sd": sd, "w_hat": w_hat, "common_hat": common_hat, "opps_hat": opps_hat, "catchers_hat": cat_hat,
           "qualifying_hat": qual_hat, "z_true": pd.Series(z, index=catchers), "observed": obs,
           "observed_fit": _fit_record(fits), "scale_true": scale, "seconds": time.perf_counter() - t0}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"outer {sd}: prepared in {out['seconds']:.0f} s", flush=True)
    return str(path)


_OUTER = {}


def _outer(sd: float) -> dict:
    if sd not in _OUTER:
        with open(OUT / f"outer_{_key(sd)}.pkl", "rb") as f:
            _OUTER[sd] = pickle.load(f)
    return _OUTER[sd]


# --- Stage B: tasks ---

def _simulate_estimate(world, opps, catchers, kappa, rng, pas, keep_solutions=True):
    code = pd.Series(np.arange(len(catchers)), index=catchers)
    k_opp = kappa[opps["start_catcher"].map(code).to_numpy()]
    sim = S.simulate(world, opps, k_opp, rng)
    run, fits = S.estimate(sim, pas, keep_solutions=keep_solutions)
    sols = {k: v for (_, k), v in run.solutions.items()} if keep_solutions else None
    return fits, sols


def _curve(world, common, sols) -> pd.DataFrame:
    kap = kappa_eval(common.kappa_grid)
    tg = S.conditional_targets(world, common, sols, np.arange(len(kap)), kap)
    return pd.DataFrame({"kappa": kap, "D": tg["D_target"].to_numpy(), "A": tg["A_target"].to_numpy()})


def run_task(task: tuple):
    kind = task[0]
    path = OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl")
    if path.exists():
        return kind
    t0 = time.perf_counter()
    pas = S._load("pilot")[0]
    if kind == "h0":
        _, sd, r = task
        o = _outer(sd)
        rng = np.random.default_rng([SEED, 2, int(round(sd * 1000)), r])
        kap = np.zeros(len(o["catchers_hat"]))
        fits, sols = _simulate_estimate(o["w_hat"], o["opps_hat"], o["catchers_hat"], kap, rng, pas)
        res = {"curve": _curve(o["w_hat"], o["common_hat"], sols), **_fit_record(fits)}
    elif kind == "h0true":
        _, r = task
        _, world, common, opps = S._load("pilot")
        catchers, _ = _catchers(opps, S._CACHE["pitches"])
        rng = np.random.default_rng([SEED, 3, r])
        fits, sols = _simulate_estimate(world, opps, catchers, np.zeros(len(catchers)), rng, pas)
        res = {"curve": _curve(world, common, sols), **_fit_record(fits)}
    elif kind == "hC":
        _, sd, cfg, r = task
        o = _outer(sd)
        z = o["z_true"].reindex(o["catchers_hat"]).fillna(0.0).to_numpy()
        scale = S.calibrate(o["common_hat"], z, o["qualifying_hat"], cfg)
        rng = np.random.default_rng([SEED, 4, int(round(cfg * 1000)), r])
        fits, sols = _simulate_estimate(o["w_hat"], o["opps_hat"], o["catchers_hat"], scale * z, rng, pas)
        res = {"curve": _curve(o["w_hat"], o["common_hat"], sols), "scale": scale, **_fit_record(fits)}
    elif kind == "design":
        _, sd, m = task
        o = _outer(sd)
        rng = np.random.default_rng([SEED, 5, int(round(sd * 1000)), m])
        sigma = float(rng.uniform(0.0, SIGMA_MAX))
        z = rng.normal(size=len(o["catchers_hat"]))
        scale = S.calibrate(o["common_hat"], z, o["qualifying_hat"], sigma)
        kappa = scale * z
        fits, sols = _simulate_estimate(o["w_hat"], o["opps_hat"], o["catchers_hat"], kappa, rng, pas)
        rec = _effects(fits, o["catchers_hat"]).assign(qualifying=o["qualifying_hat"], kappa=kappa)
        tg = S.conditional_targets(o["w_hat"], o["common_hat"], sols, o["catchers_hat"], kappa)
        rec = rec.merge(tg[["catcher", "D_target", "A_target"]], on="catcher")
        res = {"records": rec, "sigma": sigma, "scale": scale, **_fit_record(fits)}
    elif kind == "check":
        _, dgp, r = task
        _, world, common, opps = S._load(dgp)
        catchers, qualifying = _catchers(opps, S._CACHE["pitches"])
        rng = np.random.default_rng([SEED, 6, S.DGPS.index(dgp), r])
        z = rng.normal(size=len(catchers))
        scale = S.calibrate(common, z, qualifying, CHECK_SD)
        fits, _ = _simulate_estimate(world, opps, catchers, scale * z, rng, pas, keep_solutions=False)
        res = _fit_record(fits)
    else:
        raise ValueError(kind)
    res["seconds"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print(f"{path.stem}: {res['seconds']:.0f} s", flush=True)
    return kind


def tasks() -> list:
    out = [("h0true", r) for r in range(R_H)]
    for sd in OUTER_SDS:
        out += [("h0", sd, r) for r in range(R_H)]
        out += [("design", sd, m) for m in range(M_DESIGN)]
    out += [("hC", 0.01, cfg, r) for cfg in HC_SDS for r in range(R_H)]
    out += [("check", dgp, r) for dgp in S.DGPS for r in range(N_CHECK)]
    return out


def _load_task(*task):
    with open(OUT / "tasks" / ("_".join(str(x) for x in task) + ".pkl"), "rb") as f:
        return pickle.load(f)


# --- Stage C: summary ---

def _curves(kind_args: list) -> tuple[np.ndarray, dict]:
    """Stack curves; returns kappa and {score: (R, n_kappa) array}."""
    cs = [_load_task(*a)["curve"] for a in kind_args]
    kap = cs[0]["kappa"].to_numpy()
    return kap, {s: np.vstack([c[s].to_numpy() for c in cs]) for s in ("D", "A")}


def _range_mask(kap, kmax):
    return np.abs(kap) <= kmax


def summarize() -> str:
    o = {sd: _outer(sd) for sd in OUTER_SDS}
    lines = ["# Uncertainty debugging stage (simulation only)", "",
             "Generated by `python3.12 -m gamecalling.uncertainty_debug`. Design: `research_notes/uncertainty_design/` "
             "revision 4 (note 1 §8, spec 3 §10). Season 2024, truth = v2 pilot world, **fixed payoffs**. No real "
             "catcher results are used. Targets are on the model-score scale, runs per 100 pitches. "
             "**h0 is a provisional, changed target** (reference-trained models), not the v2 h_C. This is debugging, "
             "not validation: nothing here establishes coverage or calibration.", ""]

    # Kappa range in use: |kappa| up to the 97.5th percentile over design datasets of max |kappa| among qualifying.
    design = {sd: [_load_task("design", sd, m) for m in range(M_DESIGN)] for sd in OUTER_SDS}
    kmax = {sd: float(np.quantile([np.abs(d["records"].loc[d["records"]["qualifying"], "kappa"]).max()
                                   for d in design[sd]], 0.975)) for sd in OUTER_SDS}
    lines += ["## 1. h0 stability and interpolation", "",
              f"Curves are compared after subtracting their value at kappa = 0 (targets are centered across catchers, "
              f"so a common shift does not matter). Range in use: |kappa| <= the 97.5th percentile of the largest "
              f"qualifying |kappa| per design dataset.", ""]
    rows = []
    kap_t, h_true = _curves([("h0true", r) for r in range(R_H)])
    n_grid = (len(kap_t) + 1) // 2
    for sd in OUTER_SDS:
        kap, h = _curves([("h0", sd, r) for r in range(R_H)])
        msk = _range_mask(kap[:n_grid], kmax[sd])
        i0 = int(np.argmin(np.abs(kap[:n_grid])))
        for s in ("D", "A"):
            c = h[s][:, :n_grid] - h[s][:, [i0]]
            ct = h_true[s][:, :n_grid] - h_true[s][:, [i0]]
            mean, sdv = c.mean(0), c.std(0, ddof=1)
            mid = h[s][:, n_grid:].mean(0) - h[s][:, i0].mean()
            interp = (mean[:-1] + mean[1:]) / 2
            mmid = msk[:-1] & msk[1:]
            span = float(mean[msk].max() - mean[msk].min())
            rows.append({"outer_sd": sd, "score": s, "kappa_range": kmax[sd], "h0_span": span,
                         "max_sd_across_R": float(sdv[msk].max()), "median_sd_across_R": float(np.median(sdv[msk])),
                         "max_mc_se_of_mean": float(sdv[msk].max() / np.sqrt(R_H)),
                         "max_interp_error": float(np.abs(mid[mmid] - interp[mmid]).max()),
                         "max_fitted_minus_true_world": float(np.abs(mean - ct.mean(0))[msk].max())})
    lines += [_md(pd.DataFrame(rows), "{:.5f}"), "",
              "`h0_span`: range of the mean h0 over the kappa range in use. `max_fitted_minus_true_world`: h0 from the "
              "refit world vs. h0 from the true world (world-fit error in the target). D and A rows coincide after "
              "centering: on common states the recommendation and baseline terms depend only on the state "
              "(handoff, v2), so this is expected, not a bug.", ""]

    # h_C - h0
    lines += ["## 2. h_C - h0 (the target change)", ""]
    kap, h0 = _curves([("h0", 0.01, r) for r in range(R_H)])
    msk = _range_mask(kap[:n_grid], kmax[0.01])
    i0 = int(np.argmin(np.abs(kap[:n_grid])))
    rows = []
    for cfg in HC_SDS:
        _, hc = _curves([("hC", 0.01, cfg, r) for r in range(R_H)])
        for s in ("D", "A"):
            a = (hc[s][:, :n_grid] - hc[s][:, [i0]])
            b = (h0[s][:, :n_grid] - h0[s][:, [i0]])
            diff = a.mean(0) - b.mean(0)
            se = np.sqrt(a.var(0, ddof=1) / R_H + b.var(0, ddof=1) / R_H)
            span = float(b.mean(0)[msk].max() - b.mean(0)[msk].min())
            j = np.argmax(np.abs(diff) * msk)
            rows.append({"config_sd": cfg, "score": s, "max_abs_diff": float(np.abs(diff[msk]).max()),
                         "se_at_max": float(se[j]), "h0_span": span,
                         "max_abs_diff_over_span": float(np.abs(diff[msk]).max() / span)})
    lines += [_md(pd.DataFrame(rows), "{:.5f}"), ""]
    # From the design datasets: conditional target under each configuration vs h0 at the same kappa.
    rows = []
    for sd in OUTER_SDS:
        kap, h = _curves([("h0", sd, r) for r in range(R_H)])
        hm = {s: h[s][:, :n_grid].mean(0) for s in ("D", "A")}
        for d in design[sd]:
            r = d["records"][d["records"]["qualifying"]]
            for s in ("D", "A"):
                h0c = np.interp(r["kappa"], kap[:n_grid], hm[s])
                cond = r[f"{s}_target"].to_numpy()
                e = (cond - cond.mean()) - (h0c - h0c.mean())
                rows.append({"outer_sd": sd, "score": s, "sigma": d["sigma"], "theta_sd": float(np.std(h0c, ddof=1)),
                             "rms_cond_minus_h0": float(np.sqrt(np.mean(e ** 2)))})
    dd = pd.DataFrame(rows)
    dd["sigma_bin"] = pd.cut(dd["sigma"], [0, 0.0125, 0.025, 0.0375, 0.05])
    tab = dd.groupby(["outer_sd", "score", "sigma_bin"], observed=True)[["theta_sd", "rms_cond_minus_h0"]].mean().reset_index()
    tab["sigma_bin"] = tab["sigma_bin"].astype(str)
    lines += ["Per design dataset, the centered conditional target (models trained under that dataset's configuration) "
              "minus the centered h0 at the same kappa, RMS over qualifying catchers, averaged within sigma bins. "
              "`theta_sd` is the SD of the centered h0 targets in the same datasets, for scale.", "",
              _md(tab, "{:.5f}"), ""]

    # Joint support and shape
    lines += ["## 3. Design simulations: support and the shape of estimate vs. target", ""]
    shape_rows, sup_rows, t_rows, intercept_rows = [], [], [], []
    belt_rows = []
    for sd in OUTER_SDS:
        kap, h = _curves([("h0", sd, r) for r in range(R_H)])
        hm = {s: h[s][:, :n_grid].mean(0) for s in ("D", "A")}
        recs = []
        for m, d in enumerate(design[sd]):
            r = d["records"].copy()
            q = r["qualifying"].to_numpy()
            for s in ("D", "A"):
                th = np.interp(r["kappa"], kap[:n_grid], hm[s])
                r[f"{s}_theta"] = th - th[q].mean()
                r[f"{s}_est"] = r[f"{s}_effect"] - np.nanmean(r.loc[q, f"{s}_effect"])
                r[f"{s}_tau"] = d[f"{s}_sd"]
            r["m"], r["sigma"] = m, d["sigma"]
            recs.append(r)
        R = pd.concat(recs, ignore_index=True)
        R = R[R["qualifying"]]
        obs = o[sd]["observed"]
        oq = obs["qualifying"].to_numpy()
        for s in ("D", "A"):
            tau_obs = o[sd]["observed_fit"][f"{s}_sd"]
            taus = np.array([d[f"{s}_sd"] for d in design[sd]])
            sup_rows.append({"outer_sd": sd, "score": s, "tau_obs": tau_obs, "tau_design_min": taus.min(),
                             "tau_design_q05": np.quantile(taus, 0.05), "tau_design_q95": np.quantile(taus, 0.95),
                             "tau_design_max": taus.max(), "datasets_within_htau": int((np.abs(taus - tau_obs) <= SUPPORT_HTAU).sum()),
                             "share_tau_zero": float((taus < 1e-6).mean())})
            for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)):
                qlo, qhi = np.quantile(taus, [lo, hi])
                g = R[(R[f"{s}_tau"] >= qlo) & (R[f"{s}_tau"] <= qhi)].dropna(subset=[f"{s}_est"])
                X = np.column_stack([np.ones(len(g)), g[f"{s}_theta"], g[f"{s}_theta"] ** 2])
                beta, *_ = np.linalg.lstsq(X, g[f"{s}_est"].to_numpy(), rcond=None)
                X1 = X[:, :2]
                b1, *_ = np.linalg.lstsq(X1, g[f"{s}_est"].to_numpy(), rcond=None)
                res = g[f"{s}_est"].to_numpy() - X1 @ b1
                shape_rows.append({"outer_sd": sd, "score": s, "tau_tertile": f"[{qlo:.4f}, {qhi:.4f}]",
                                   "records": len(g), "intercept": b1[0], "slope": b1[1], "resid_sd": res.std(ddof=2),
                                   "quadratic_coef": beta[2], "theta_sd": g[f"{s}_theta"].std()})
            # Per-catcher intercepts: centering makes the pooled intercept 0, so catcher-specific bias at theta = 0
            # shows up only catcher by catcher.
            ints, ses = [], []
            for _, g in R.dropna(subset=[f"{s}_est"]).groupby("catcher"):
                if len(g) < 10 or g[f"{s}_theta"].std() < 1e-9:
                    continue
                X = np.column_stack([np.ones(len(g)), g[f"{s}_theta"]])
                b, *_ = np.linalg.lstsq(X, g[f"{s}_est"].to_numpy(), rcond=None)
                res = g[f"{s}_est"].to_numpy() - X @ b
                cov = np.linalg.inv(X.T @ X) * res.var(ddof=2)
                ints.append(b[0])
                ses.append(np.sqrt(cov[0, 0]))
            ints, ses = np.array(ints), np.array(ses)
            if len(ints) > 1:
                intercept_rows.append({"outer_sd": sd, "score": s, "catchers": len(ints), "intercept_sd": ints.std(ddof=1),
                                       "mean_intercept_se": ses.mean(),
                                       "share_abs_z_gt_2": float((np.abs(ints / ses) > 2).mean()),
                                       "max_abs_intercept": np.abs(ints).max()})
            belt_rows += _belts(R, obs, oq, s, sd, tau_obs)
        # T* vs sigma
        for d in design[sd]:
            t_rows.append({"source": f"design (W_hat of outer {sd})", "sigma": d["sigma"], "D_sd": d["D_sd"], "A_sd": d["A_sd"]})
        for r in range(R_H):
            d = _load_task("h0", sd, r)
            t_rows.append({"source": f"design (W_hat of outer {sd})", "sigma": 0.0, "D_sd": d["D_sd"], "A_sd": d["A_sd"]})
    lines += ["Fitted catcher SD (tau, per 100 pitches) of the outer dataset vs. the design datasets:", "",
              _md(pd.DataFrame(sup_rows), "{:.5f}"), "",
              "Estimate (centered effect) against target (centered h0 at the catcher's kappa), qualifying catchers, "
              "pooled within tertiles of the design datasets' tau. A quadratic term is fitted only to check linearity. "
              "Both are centered within each dataset, so the pooled intercept is 0 by construction.", "",
              _md(pd.DataFrame(shape_rows), "{:.5f}"), "",
              "Per-catcher intercepts at theta = 0 (estimate on target, all design datasets, catchers with >= 10 "
              "records): their SD across catchers vs. their mean standard error. An SD well above the SE, or many "
              "|z| > 2, indicates catcher-specific bias that the belt's intercept must absorb.", "",
              _md(pd.DataFrame(intercept_rows), "{:.5f}"), ""]

    lines += ["## 4. Belts for selected catchers (experimental; debugging only)", "",
              "Quantile regression of the estimate on [1, theta, tau, theta x tau] at p = 0.025 and 0.975, evaluated "
              f"at the outer dataset's tau and inverted over the supported theta range. Endpoint variability from "
              f"{N_BELT_BOOT} bootstrap resamples of design datasets (no refits). Joint support: design records with "
              f"|theta - t| <= {SUPPORT_H} and |tau - tau_obs| <= {SUPPORT_HTAU}, minimum over the interval's t range; "
              f"'insufficient support' below {SUPPORT_MIN}. Category per note 1 §6a. `covers_true_h0` uses h0 from "
              "the true world at the catcher's true kappa (one dataset; not coverage evidence).", "",
              _md(pd.DataFrame(belt_rows), "{:.5f}"), ""]

    # T* vs s
    T = pd.DataFrame(t_rows)
    T["sigma_bin"] = pd.cut(T["sigma"], [-1e-9, 1e-9, 0.01, 0.02, 0.03, 0.04, 0.05], labels=["0", "(0, 0.01]", "(0.01, 0.02]", "(0.02, 0.03]", "(0.03, 0.04]", "(0.04, 0.05]"])
    agg = T.groupby(["source", "sigma_bin"], observed=True).agg(
        n=("D_sd", "size"), D_mean=("D_sd", "mean"), D_q05=("D_sd", lambda x: x.quantile(0.05)),
        D_q95=("D_sd", lambda x: x.quantile(0.95)), D_zero=("D_sd", lambda x: (x < 1e-6).mean()),
        A_mean=("A_sd", "mean"), A_zero=("A_sd", lambda x: (x < 1e-6).mean())).reset_index()
    agg["sigma_bin"] = agg["sigma_bin"].astype(str)
    rho = {src: g[["sigma", "D_sd"]].corr(method="spearman").iloc[0, 1] for src, g in T.groupby("source")}
    chk = []
    for dgp in S.DGPS:
        v = pd.DataFrame([_load_task("check", dgp, r) for r in range(N_CHECK)])
        chk.append({"world": dgp, "n": len(v), "D_mean": v["D_sd"].mean(), "D_q05": v["D_sd"].quantile(0.05),
                    "D_q95": v["D_sd"].quantile(0.95), "A_mean": v["A_sd"].mean(), "A_q05": v["A_sd"].quantile(0.05),
                    "A_q95": v["A_sd"].quantile(0.95), "converged": float(v[["D_converged", "A_converged"]].all(axis=1).mean())})
    lines += ["## 5. The verdict statistic T (fitted catcher SD of D) against the scientific SD", "",
              _md(agg, "{:.5f}"), "",
              "Spearman correlation of sigma with T: " + "; ".join(f"{k}: {v:.3f}" for k, v in rho.items()) + ".", "",
              f"True worlds at scientific SD {CHECK_SD} (fresh z, {N_CHECK} datasets each):", "",
              _md(pd.DataFrame(chk), "{:.5f}"), ""]

    conv = []
    for t in tasks():
        d = _load_task(*t)
        conv.append(bool(d.get("D_converged", True) and d.get("A_converged", True)))
    secs = [_load_task(*t)["seconds"] for t in tasks()]
    lines += ["## Run", "", f"{len(secs)} tasks, median {np.median(secs):.0f} s each; ranking fits converged in "
              f"{np.mean(conv):.1%} of tasks.", ""]
    return "\n".join(lines)


def belt_interval(y, th, tau, a_obs, tau_obs, t_grid):
    """Belt inversion: t values with q_0.025(t, tau_obs) <= a_obs <= q_0.975(t, tau_obs).

    Quantile regression of y on [1, theta, tau, theta x tau]. Returns None if a fit fails.
    """
    from statsmodels.regression.quantile_regression import QuantReg
    X = np.column_stack([np.ones(len(y)), th, tau, th * tau])
    try:
        b_lo = QuantReg(y, X).fit(q=0.025).params
        b_hi = QuantReg(y, X).fit(q=0.975).params
    except Exception:
        return None
    Xt = np.column_stack([np.ones_like(t_grid), t_grid, np.full_like(t_grid, tau_obs), t_grid * tau_obs])
    ok = (Xt @ b_lo <= a_obs) & (a_obs <= Xt @ b_hi)
    return t_grid[ok]


def _belts(R, obs, oq, s, sd, tau_obs) -> list:
    """Experimental belts for six qualifying catchers chosen by pitch count (two small, two medium, two large)."""
    q = obs[oq].dropna(subset=[f"{s}_effect"]).sort_values(f"{s}_n")
    picks = q.iloc[np.unique(np.round(np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95]) * (len(q) - 1)).astype(int))]
    est_obs = obs[f"{s}_effect"] - np.nanmean(obs.loc[oq, f"{s}_effect"])
    # True target: h0 of the true world at the true kappa, centered over qualifying catchers.
    kap_t, h_true = _curves([("h0true", r) for r in range(R_H)])
    n_grid = (len(kap_t) + 1) // 2
    th_true = np.interp(obs["kappa"], kap_t[:n_grid], h_true[s][:, :n_grid].mean(0))
    th_true = pd.Series(th_true - th_true[oq].mean(), index=obs["catcher"])
    rows = []
    rng = np.random.default_rng([SEED, 9])
    for _, c in picks.iterrows():
        g = R[(R["catcher"] == c["catcher"])].dropna(subset=[f"{s}_est"])
        y = g[f"{s}_est"].to_numpy()
        th, tau = g[f"{s}_theta"].to_numpy(), g[f"{s}_tau"].to_numpy()
        a_obs = float(est_obs[obs["catcher"] == c["catcher"]].iloc[0])
        t_grid = np.linspace(th.min(), th.max(), 401)

        def interval(idx):
            return belt_interval(y[idx], th[idx], tau[idx], a_obs, tau_obs, t_grid)

        base = interval(np.arange(len(y)))
        boots = []
        ds = g["m"].unique()
        for _ in range(N_BELT_BOOT):
            pick = rng.choice(ds, len(ds), replace=True)
            idx = np.concatenate([np.flatnonzero(g["m"].to_numpy() == m) for m in pick])
            b = interval(idx)
            if b is not None and len(b):
                boots.append((b.min(), b.max()))
        if base is None:
            cat, lo, hi, sup = "fit failed", np.nan, np.nan, np.nan
        elif len(base) == 0:
            cat, lo, hi, sup = "empty", np.nan, np.nan, np.nan
        else:
            lo, hi = float(base.min()), float(base.max())
            open_lo, open_hi = lo <= t_grid[0], hi >= t_grid[-1]
            ts = np.linspace(lo, hi, 25)
            sup = int(min(((np.abs(th - t) <= SUPPORT_H) & (np.abs(tau - tau_obs) <= SUPPORT_HTAU)).sum() for t in ts))
            cat = ("unbounded" if (open_lo and open_hi) else "open" if (open_lo or open_hi)
                   else "insufficient support" if sup < SUPPORT_MIN else "usable")
        bl = np.array(boots) if boots else np.full((1, 2), np.nan)
        truth = float(th_true.get(c["catcher"], np.nan))
        rows.append({"outer_sd": sd, "score": s, "catcher": int(c["catcher"]), "pitches": int(c[f"{s}_n"]),
                     "estimate": a_obs, "lo": lo, "hi": hi, "category": cat, "min_joint_support": sup,
                     "lo_boot_sd": float(np.nanstd(bl[:, 0])), "hi_boot_sd": float(np.nanstd(bl[:, 1])),
                     "pev_lo": a_obs - 1.96 * c[f"{s}_pev_se"], "pev_hi": a_obs + 1.96 * c[f"{s}_pev_se"],
                     "true_h0": truth, "covers_true_h0": bool(lo <= truth <= hi) if np.isfinite(lo) else False,
                     "theta_range": f"[{t_grid[0]:.4f}, {t_grid[-1]:.4f}]"})
    return rows


def main(workers: int = 8):
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.json"
    if man.exists() and json.loads(man.read_text()) != json.loads(json.dumps(manifest())):
        raise ValueError(f"{OUT} was made with a different configuration; use a new version")
    man.write_text(json.dumps(manifest(), indent=1))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(OUTER_SDS))) as ex:
        list(ex.map(prepare_outer, OUTER_SDS))
    todo = tasks()
    print(f"stage A done in {time.perf_counter() - t0:.0f} s; {len(todo)} tasks", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(run_task, todo), 1):
            if i % 20 == 0:
                print(f"{i}/{len(todo)} tasks done ({time.perf_counter() - t0:.0f} s)", flush=True)
    REPORT.write_text(summarize())
    print(f"debugging stage done in {time.perf_counter() - t0:.0f} s; report in {REPORT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
