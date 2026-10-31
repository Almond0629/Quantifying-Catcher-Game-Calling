"""Research program S7: batter ground-ball rate (M4), prediction gate, prediction-only.

Specification: research_notes/research_program/S7_ground_ball.md (revision 1); model code in batter.py.
- M4_off: ground_ball=False, which is M3a's existing fitting path; its outcome and contact predictions must equal
  M3a's stored ones exactly (test 1).
- M4: M3a's outcome model, unchanged (test 2), and M3a's contact model plus ground-ball rate, main effects only.
Gate: contact only, nats per ball in play, delta 0.0005, full precision. M4's contact component is adopted only if
it passes against both P0's and M3a's. Otherwise M3a's stays. One run of the gate. M1c stays separate until S8.
Descriptive; 2024-2025 are previously examined seasons (exploratory reuse).

Importing research_s6 first limits every process to one math thread before numpy initializes.

Run with: python3.12 -m gamecalling.research_s7 run [WORKERS]   (fit in parallel; timed)
          python3.12 -m gamecalling.research_s7 gate             (checks, gate and report)
"""

from gamecalling import research_s6  # noqa: I001 (first: sets the thread limit)

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd

from gamecalling import batter, config, research
from gamecalling.phase1 import _md
from gamecalling.research import Spec
from gamecalling.research_checks import _load, _save
from gamecalling.research_s4 import _same
from gamecalling.research_s6 import FIT_COLUMNS, verify_threads

RAW_COLUMNS = batter.BB_KEYS + ["bb_type"]
RUN_JSON = config.DATA_DIR / "research" / "s7_run.json"
S7_REPORT = config.REPORTS_DIR / "research_s7.md"
COMPARISONS = (("P0", "M4"), ("M3a", "M4"))
PRED = ("heldout_outcome", "heldout_contact", "forward_outcome", "forward_contact")


def attach_raw(pitches: pd.DataFrame, pas: pd.DataFrame):
    """Spec.transform: pitch rows with bb_type from the raw table (the pitch table itself is not rewritten)."""
    return batter.attach_bb_type(pitches, pd.read_parquet(config.RAW_PARQUET, columns=RAW_COLUMNS)), pas


SPECS = {
    "M4_off": Spec("M4_off", "M4 with ground-ball rate explicitly disabled (ground_ball=False): M3a's existing path",
                   fit_models=partial(batter.fit_models, variant="M3a", ground_ball=False), training=batter.training,
                   params={"ground_ball": False}),
    "M4": Spec("M4", "M3a's outcome model; M3a's contact model plus batter ground-ball rate, main effects only",
               fit_models=partial(batter.fit_models, ground_ball=True), training=batter.training,
               transform=attach_raw, extra_inputs=(config.RAW_PARQUET,),
               params={"ground_ball": True, "lambdas": list(batter.LAMBDAS)}),
}


def _fit_job(args):
    model_id, season, fold = args
    threads = verify_threads()
    t = time.perf_counter()
    p = pd.read_parquet(config.PITCHES_PARQUET, columns=FIT_COLUMNS + ["at_bat_number", "pitch_number"])
    rows = p[p["season"] == season]
    if fold != "forward":
        rows = rows[rows["fold"] == fold]
    if SPECS[model_id].transform is not None:
        rows, _ = SPECS[model_id].transform(rows, None)
    models_ = SPECS[model_id].fit_models(rows.copy())
    key = "forward" if fold == "forward" else (season, fold)
    return model_id, key, models_, {"model": model_id, "season": season, "fold": str(fold),
                                    "seconds": time.perf_counter() - t, **threads}


def stage_run(workers: int = 8):
    t0 = time.perf_counter()
    parent = verify_threads()
    jobs = [(m, s, f) for m in SPECS for s in config.SEASONS for f in range(config.N_FOLDS)]
    jobs += [(m, config.SEASONS[0], "forward") for m in SPECS]
    prefit, log = {m: {} for m in SPECS}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for model_id, key, models_, entry in ex.map(_fit_job, jobs):
            prefit[model_id][key] = models_
            log.append(entry)
            print(f"{model_id} {key}: {entry['seconds']:.0f} s", flush=True)
    t_fit = time.perf_counter() - t0
    for m, spec in SPECS.items():
        research.run_predictions(spec, prefit=prefit[m], seconds_fit=sum(e["seconds"] for e in log if e["model"] == m))
    RUN_JSON.write_text(json.dumps({"workers": workers, "parent_threads": parent, "jobs": log, "wall_seconds_fits": t_fit,
                                    "wall_seconds_total": time.perf_counter() - t0, "budget_seconds": 3600}, indent=1))
    print(f"total {time.perf_counter() - t0:.0f} s", flush=True)


# --- Checks and gate ---

def checks() -> pd.DataFrame:
    rows = []
    for mid, whats, label in (("M4_off", PRED, "test 1: M4_off (ground_ball=False)"),
                              ("M4", PRED[0::2], "test 2: M4 outcome (unchanged)")):
        for what in whats:
            a = research.load(mid, what)
            ok, detail = _same(a, research.load("M3a", what))
            rows.append({"check": f"{label}: {what} vs M3a stored", "rows": len(a), "exact": ok, "detail": detail})
    return pd.DataFrame(rows)


def adopted(vs_p0: bool, vs_m3a: bool) -> str:
    """The gate's M4 rule: M4 only if it passes against both P0 and M3a; otherwise M3a's contact stays."""
    return "M4" if vs_p0 and vs_m3a else "M3a"


def stage_gate():
    t0 = time.perf_counter()
    res = {"checks": checks()}
    if not res["checks"]["exact"].all():
        raise RuntimeError("a regression check failed; fix before applying the gate")
    res["cmp"] = {pair: research.prediction_compare(*pair, "contact") for pair in COMPARISONS}
    res["gate"] = {pair: research.gate(c) for pair, c in res["cmp"].items()}
    res["chosen"] = adopted(res["gate"]["P0", "M4"]["passes"], res["gate"]["M3a", "M4"]["passes"])
    man = research.load("M4", "manifest")
    res["alphas"], res["forward_alphas"] = pd.DataFrame(man["alphas"]), man["forward_alphas"]
    res["m3a_alphas"] = pd.DataFrame(research.load("M3a", "manifest")["alphas"])
    res["inputs"] = man["inputs"]
    res["abnormal"] = pd.DataFrame([e for r in man["alphas"] + [man["forward_alphas"]]
                                    for kind in ("outcome", "contact") for e in (r.get(f"{kind}_abnormal_log") or [])],
                                   columns=["variant", "component", "season", "code", "inner", "lambda", "status",
                                            "message", "max_grad"])
    res["missing"] = research_s6.missing_history()[["sample", "rows", "xwobacon"]].rename(
        columns={"xwobacon": "no_denominator_share"})
    res["run"] = json.loads(RUN_JSON.read_text())
    _save("s7", res)
    S7_REPORT.write_text(s7_report(res))
    print(f"stage s7 gate done ({time.perf_counter() - t0:.0f} s); contact component: {res['chosen']}", flush=True)
    return res


# --- Report ---

def _gate_table(res) -> pd.DataFrame:
    rows = []
    for pair, c in res["cmp"].items():
        g, s = res["gate"][pair], c["season"].set_index("season")
        rows.append({"comparison": f"M4 vs {pair[0]}", "gain_2024": s.loc[2024, "gain"], "gain_2025": s.loc[2025, "gain"],
                     "both_seasons": g["both_seasons"], "max_cal": float(s["cal_cand"].max()),
                     "calibration": g["calibration"], "forward_gain": c["forward"]["gain"], "forward": g["forward"],
                     "passes": g["passes"]})
    return pd.DataFrame(rows)


def _loss_table(res) -> pd.DataFrame:
    rows = []
    for (b, v), c in res["cmp"].items():
        for r in c["season"].itertuples():
            rows.append({"base": b, "sample": str(r.season), "n_matched": r.n, "base_loss": r.base, "M4_loss": r.cand,
                         "gain": r.gain, "max_cal_base": f"{r.cal_base:.6f}", "max_cal_M4": r.cal_cand})
        f = c["forward"]
        rows.append({"base": b, "sample": "forward 2024->2025", "n_matched": f["n"], "base_loss": f["base"],
                     "M4_loss": f["cand"], "gain": f["gain"], "max_cal_base": "", "max_cal_M4": f["cal_cand"]})
    return pd.DataFrame(rows)


def _fold_table(res) -> pd.DataFrame:
    return pd.concat([c["fold"].rename(columns={"base": "base_loss", "cand": "M4_loss"}).assign(base=b)
                      for (b, _), c in res["cmp"].items()], ignore_index=True)[
        ["base", "season", "fold", "n", "base_loss", "M4_loss", "gain"]]


def _penalty_table(res) -> pd.DataFrame:
    a = res["alphas"].to_dict("records") + [{"season": "forward", "fold": "", **res["forward_alphas"]}]
    m3 = res["m3a_alphas"].set_index(["season", "fold"])["contact_lambda"]
    rows = []
    for r in a:
        rows.append({"role": f"{r['season']} fold {r['fold']}".strip(), "lambda": f"{r['contact_lambda']:g}",
                     "M3a_lambda": f"{m3[r['season'], r['fold']]:g}" if r["season"] != "forward" else "",
                     **{f"cv_{lam:g}": r[f"contact_cv_{lam:g}"] for lam in batter.LAMBDAS}})
    return pd.DataFrame(rows)


def _feature_table(res) -> pd.DataFrame:
    keys = ["kappa_ground_ball", "mu_ground_ball", "mean_ground_ball", "sd_ground_ball", "corr_ground_ball_xwobacon",
            "shift_rms_ground_ball", "shift_rms_whiff", "shift_rms_chase", "shift_rms_xwobacon"]
    a = res["alphas"].to_dict("records") + [{"season": "forward", "fold": "", **res["forward_alphas"]}]
    return pd.DataFrame([{"season": r["season"], "fold": r["fold"],
                          **{k.replace("_ground_ball", "_gb"): r[f"contact_{k}"] for k in keys}} for r in a])


def _bin_table(res) -> pd.DataFrame:
    a = res["alphas"]
    return pd.DataFrame([{"contact_bin": b, "mean_shift": a[f"contact_shift_ground_ball_bin{b}"].mean(),
                          "min_shift": a[f"contact_shift_ground_ball_bin{b}"].min(),
                          "max_shift": a[f"contact_shift_ground_ball_bin{b}"].max()} for b in range(1, 9)])


def s7_report(res) -> str:
    ch, chosen, ab, r = res["checks"], res["chosen"], res["abnormal"], res["run"]
    raw = res["inputs"][config.RAW_PARQUET.name]
    jobs = pd.DataFrame(r["jobs"])
    result = (f"**Gate result: {chosen}'s contact component.** " + (
        "M4 passes against both P0 and M3a. M3a is still scored for the required descriptive comparison."
        if chosen == "M4" else "M4 does not pass against both bases, so M3a's contact component stays. This is a "
        "negative predictive result for these specifications; no other shape feature, bin or interaction is tried."))
    ab_lines = ["No fit ended with an accepted abnormal termination."] if ab.empty else [
        f"Accepted abnormal terminations: {len(ab)}.", "",
        _md(ab.assign(max_grad=ab["max_grad"].map("{:.2e}".format), **{"lambda": ab["lambda"].map("{:g}".format)}))]
    out = ["# Research program S7: batter ground-ball rate (M4)", "",
           "Generated by `python3.12 -m gamecalling.research_s7 gate` (versions fitted by `... run`). Specification: "
           "`research_notes/research_program/S7_ground_ball.md` (revision 1); model code `gamecalling/batter.py`. "
           "M3a's outcome model is retained; ground-ball rate is added to M3a's contact model, main effects only. M1c "
           "stays separate. Prediction-only. Descriptive; 2024-2025 are previously examined seasons (exploratory "
           "reuse). One run of the gate. A passing gate shows better probabilities, not better comparisons between "
           "pitch choices.", "",
           "## Checks", "", _md(ch.assign(exact=ch["exact"].map({True: "yes", False: "NO"}))), "",
           f"The raw parquet is a model input: `{config.RAW_PARQUET.name}` sha256 `{raw['sha256'][:12]}` in M4's manifest. "
           "The join keeps row order and index, and refuses duplicate keys, unmatched pitch rows and unexpected labels.", "",
           "## Prediction gate (contact; nats per contact-fit-eligible ball in play; delta 0.0005; full precision)", "",
           "M4 must pass against both P0 and M3a. Gain = base log loss minus M4 log loss on matched held-out "
           "observations.", "", _md(_gate_table(res), "{:.6f}"), "", result, "",
           "Matched held-out and forward log loss:", "", _md(_loss_table(res), "{:.6f}"), "",
           "Per fold (reported, not gated):", "", _md(_fold_table(res), "{:.6f}"), "",
           "## Penalties (reported, not gated)", "",
           "M4 selects its own lambda on M3a's inner folds.", "", _md(_penalty_table(res), "{:.6f}"), "",
           "## Ground-ball feature and coefficients (reported)", "",
           "`corr_gb_xwobacon`: unweighted correlation across batters with a denominator, of shrunk rates and means. "
           "`shift_rms`: root-mean-square centered log-probability shift per feature SD.", "",
           _md(_feature_table(res), "{:.4f}"), "",
           "Ground-ball rate's centered log-probability shift per SD, by contact bin, over the six roles:", "",
           _md(_bin_table(res), "{:.4f}"), "",
           "## Batters without history (reported)", "",
           "Share of held-out rows whose batter has no tracked, non-bunt ball in play in the role fold (league rate).",
           "", _md(res["missing"], "{:.4f}"), "",
           "## Convergence (S6 §4.3, revision 2)", "", *ab_lines, "",
           "## Runtime and math threads", "",
           f"Full run: {r['wall_seconds_total'] / 60:.1f} min wall-clock at {r['workers']} workers, against the 60-minute "
           f"budget. One math thread verified in the parent and every job; largest CPU over wall time "
           f"{jobs['cpu_over_wall'].max():.2f}.", "", _md(jobs[["model", "season", "fold", "seconds"]], "{:.1f}"), ""]
    return "\n".join(out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        {"gate": stage_gate}[stage]()
