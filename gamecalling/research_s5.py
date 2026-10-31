"""Research program S5, step 1: previous-pitch prediction gate (M2a, M2b), no solver work.

Specification: research_notes/research_program/S5_previous_pitch.md (revision 2); model code in previous.py.
- M2_off: M2's code path with prev=False, as a full scored run; must reproduce P0's stored scores and
  predictions exactly (test 1).
- P0_predonly: P0 through the prediction-only path (research.run_predictions); must reproduce P0's held-out
  and forward predictions exactly, which checks the path itself.
- M2a, M2b: prediction-only runs (outcome model changed; contact is P0's and must predict identically).
Gate: outcome component only (delta 0.0005 nats per pitch, full precision, both seasons, calibration <= 0.02,
forward no worse), with the nested rule: M2b only if it passes against both P0 and M2a; otherwise M2a if it
passes against P0; otherwise neither. Fold gains, the first-pitch / later-pitch split and the concentrations
are reported, not gated. Descriptive; 2024-2025 are previously examined seasons (exploratory reuse).

Run with: python3.12 -m gamecalling.research_s5 run [WORKERS]   (fit the four versions)
          python3.12 -m gamecalling.research_s5 gate             (checks, nested gate and report)
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from gamecalling import config, previous, research
from gamecalling.phase1 import SCORES_PARQUET, _md
from gamecalling.research import KEYS, Spec
from gamecalling.research_checks import _load, _save
from gamecalling.research_s4 import _same

SPECS = {
    "M2_off": Spec("M2_off", "M2 with the previous-pitch extension explicitly disabled (prev=False): P0's own "
                   "functions only (full scored run)", fit_models=partial(previous.fit_models, prev=False),
                   training=previous.training, params={"prev": False}),
    "P0_predonly": Spec("P0_predonly", "P0 through the prediction-only path (check of the path)",
                        params={"prediction_only": True}),
    "M2a": Spec("M2a", "Previous pitch type on the outcome model (ratio-shifted on P0); contact P0's",
                fit_models=partial(previous.fit_models, variant="M2a"), training=previous.training,
                params={"variant": "M2a", "extra": list(previous.EXTRA["M2a"])}),
    "M2b": Spec("M2b", "Previous pitch type and result on the outcome model (ratio-shifted on M2a, with the "
                "none / non_action exception); contact P0's", fit_models=partial(previous.fit_models, variant="M2b"),
                training=previous.training, params={"variant": "M2b", "extra": list(previous.EXTRA["M2b"])}),
}
FULL_RUNS = ("M2_off",)
COMPARISONS = (("P0", "M2a"), ("P0", "M2b"), ("M2a", "M2b"))


def _run(model_id):
    t = time.perf_counter()
    spec = SPECS[model_id]
    man = research.run_model(spec) if model_id in FULL_RUNS else research.run_predictions(spec)
    return model_id, time.perf_counter() - t


def stage_run(workers: int = 8, ids=None):
    ids = ids or list(SPECS)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
        for model_id, sec in ex.map(_run, ids):
            print(f"{model_id}: {sec:.0f} s", flush=True)


# --- Checks ---

PRED = ("heldout_outcome", "heldout_contact", "forward_outcome", "forward_contact")


def checks() -> pd.DataFrame:
    rows = []
    off = research.load("M2_off", "scores")
    ok, detail = _same(off, pd.read_parquet(SCORES_PARQUET))
    rows.append({"check": "test 1: M2_off scores vs data/phase1_scores.parquet", "rows": len(off), "exact": ok,
                 "detail": detail})
    for mid, whats, label in (("M2_off", PRED, "test 1: M2_off"), ("P0_predonly", PRED, "prediction-only path: P0"),
                              ("M2a", PRED[1::2], "M2a contact (unchanged)"),
                              ("M2b", PRED[1::2], "M2b contact (unchanged)")):
        for what in whats:
            a = research.load(mid, what)
            ok, detail = _same(a, research.load("P0", what))
            rows.append({"check": f"{label}: {what} vs P0_repro", "rows": len(a), "exact": ok, "detail": detail})
    return pd.DataFrame(rows)


# --- Gate ---

def first_pitch_split(base_id: str, cand_id: str) -> pd.DataFrame:
    """Matched outcome gain on first pitches of the PA (prev none at 0-0) and on all other pitches; reported,
    not gated."""
    m = research.matched(research.load(base_id, "heldout_outcome"), research.load(cand_id, "heldout_outcome"))
    p = research.base_tables()[0]
    first = p[KEYS].assign(first=(~p["has_prev_pitch"].astype(bool) & (p["balls"] == 0) & (p["strikes"] == 0))
                           .to_numpy())
    m = m.merge(first, on=KEYS, how="left", validate="one_to_one")
    if m["first"].isna().any():
        raise ValueError("matched rows missing from the pitch table")
    rows = []
    for (season, f), g in m.groupby(["season", "first"]):
        rows.append({"base": base_id, "version": cand_id, "season": season,
                     "pitches": "first of PA (none, 0-0)" if f else "later", "n": len(g),
                     "share": len(g) / (m["season"] == season).sum(), "gain": (g["lp_cand"] - g["lp_base"]).mean()})
    return pd.DataFrame(rows)


def stage_gate():
    t0 = time.perf_counter()
    res = {"checks": checks()}
    if not res["checks"]["exact"].all():
        raise RuntimeError("a regression or unchanged-component check failed; fix before applying the gate")
    res["cmp"] = {pair: research.prediction_compare(*pair, "outcome") for pair in COMPARISONS}
    res["gate"] = {pair: research.gate(c) for pair, c in res["cmp"].items()}
    passes = {pair: g["passes"] for pair, g in res["gate"].items()}
    choice = research.nested_choice(passes["P0", "M2a"], passes["P0", "M2b"], passes["M2a", "M2b"])
    res["chosen"] = {"complex": "M2b", "simple": "M2a", "base": None}[choice]
    res["split"] = pd.concat([first_pitch_split(*pair) for pair in COMPARISONS], ignore_index=True)
    res["alphas"] = {m: pd.DataFrame(research.load(m, "manifest")["alphas"]) for m in ("M2a", "M2b")}
    res["forward_alphas"] = {m: research.load(m, "manifest")["forward_alphas"] for m in ("M2a", "M2b")}
    res["seconds"] = {m: research.load(m, "manifest")["seconds"]["total"] for m in SPECS}
    _save("s5", res)
    S5_REPORT.write_text(s5_report(res))
    print(f"stage s5 gate done ({time.perf_counter() - t0:.0f} s); chosen: {res['chosen'] or 'neither'}", flush=True)
    return res


# --- Report ---

S5_REPORT = config.REPORTS_DIR / "research_s5.md"


def _gate_table(res) -> pd.DataFrame:
    rows = []
    for pair, c in res["cmp"].items():
        g, s = res["gate"][pair], c["season"].set_index("season")
        rows.append({"comparison": f"{pair[1]} vs {pair[0]}", "gain_2024": s.loc[2024, "gain"],
                     "gain_2025": s.loc[2025, "gain"], "both_seasons": g["both_seasons"],
                     "max_cal": float(s["cal_cand"].max()), "calibration": g["calibration"],
                     "forward_gain": c["forward"]["gain"], "forward": g["forward"], "passes": g["passes"]})
    return pd.DataFrame(rows)


def _loss_table(res) -> pd.DataFrame:
    rows = []
    for (b, v), c in res["cmp"].items():
        for r in c["season"].itertuples():
            rows.append({"base": b, "version": v, "sample": str(r.season), "n_matched": r.n, "base_loss": r.base,
                         "version_loss": r.cand, "gain": r.gain, "max_cal_base": r.cal_base,
                         "max_cal_version": r.cal_cand})
        f = c["forward"]
        rows.append({"base": b, "version": v, "sample": "forward 2024->2025", "n_matched": f["n"],
                     "base_loss": f["base"], "version_loss": f["cand"], "gain": f["gain"],
                     "max_cal_version": f["cal_cand"]})
    out = pd.DataFrame(rows)
    out["max_cal_base"] = out["max_cal_base"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    return out


def _fold_table(res) -> pd.DataFrame:
    return pd.concat([c["fold"].rename(columns={"base": "base_loss", "cand": "version_loss"}).assign(base=b, version=v)
                      for (b, v), c in res["cmp"].items()], ignore_index=True)[
        ["base", "version", "season", "fold", "n", "base_loss", "version_loss", "gain"]]


def _alpha_table(res) -> pd.DataFrame:
    a = res["alphas"]["M2b"]
    cols = ["season", "fold"] + [c for c in a if c.startswith("outcome_")]
    return a[cols].rename(columns=lambda c: c.replace("outcome_", ""))


def _bound_lines(res) -> list:
    bound = float(np.exp(research.models.pooling.LOG_ALPHA_BOUNDS[1]))
    a = res["alphas"]["M2b"]
    out = []
    for key, name in (("outcome_p", "alpha_p (M2a cell, refitted in M2b's roles)"), ("outcome_pr", "alpha_pr (M2b cell)"),
                      ("outcome_league_p", "alpha_league_p"), ("outcome_league_pr", "alpha_league_pr")):
        v = a[key]
        near = (v >= 0.95 * bound).all()
        out.append(f"- {name}: {v.min():,.1f} to {v.max():,.1f}"
                   + (f", at or within {1 - v.min() / bound:.2%} of the optimizer's upper bound ({bound:,.0f}): "
                      "deviations at this level are heavily shrunk, not shown to be absent." if near else "."))
    m2a, m2b = res["alphas"]["M2a"], res["alphas"]["M2b"]
    same = all(np.array_equal(m2a[c].to_numpy(), m2b[c].to_numpy()) for c in m2a if c.startswith("outcome_"))
    fa = res["forward_alphas"]["M2b"]
    out.append(f"- **Forward models** (fitted on all of 2024, about three times a role fold's data): league_p "
               f"{fa['outcome_league_p']:,.1f}, p {fa['outcome_p']:,.1f}, league_pr {fa['outcome_league_pr']:,.1f}, "
               f"pr {fa['outcome_pr']:,.1f}. The within-season roles put alpha_league_p at the bound, and M2a's mean "
               "held-out gain over P0 is about 0.000001 nats per pitch; the forward fit keeps a league previous-type "
               "shift (forward gain above). Reported as a description of the fits; the gate is unchanged.")
    out.append(f"- M2a's concentrations refitted inside M2b's roles are {'identical to' if same else 'NOT identical to'} "
               "the M2a run's (same rows, same parent).")
    return out


def _first_pitch_line(res) -> str:
    s = res["split"]
    f = s[s["pitches"].str.startswith("first") & (s["base"] == "P0")]
    vals = ", ".join(f"{r.version} {r.season}: {r.gain:.1e}" for r in f.itertuples())
    return f"Against P0 the first-pitch gains round to zero in the table but are not exactly zero ({vals})."


def s5_report(res) -> str:
    ch = res["checks"]
    chosen = res["chosen"]
    result = (f"**Nested rule: {chosen} is chosen.** Step 2 (solver and scoring extension, tests 5-6, criterion 2 "
              f"again) applies to {chosen} only; it is not started in this report." if chosen else
              "**Nested rule: neither variant is chosen.** S5 stops after step 1 with a negative predictive "
              "result: no solver work and no further variant (§6).")
    out = ["# Research program S5, step 1: previous-pitch prediction gate (M2a, M2b)", "",
           "Generated by `python3.12 -m gamecalling.research_s5 gate` (versions fitted by `... run`). "
           "Specification: `research_notes/research_program/S5_previous_pitch.md` (revision 2); model code "
           "`gamecalling/previous.py`. Outcome component only; the contact model is P0's. Built on P0 alone (not on "
           "M1c). Descriptive; 2024-2025 are previously examined seasons (exploratory reuse). One run of the gate.", "",
           "## Checks", "", _md(ch.assign(exact=ch["exact"].map({True: "yes", False: "NO"}))), "",
           "Unit tests (`tests/test_previous.py`): M2a and M2b hierarchy formulas against a hand computation "
           "(1e-12), the sparse-cell fallbacks with positivity and summing to 1, the M2b exception (exact equality "
           "with M2a for `none` and `non_action`; M2b's additional counts and concentrations unchanged when "
           "exception rows are altered or removed, holding the M2a parent fixed; M2a's counts include them), the "
           "prev categories, and the disabled switch. The policy part of test 4 and the solver tests come with "
           "step 2.", "",
           "## Prediction gate (outcome; nats per fit-eligible pitch; delta 0.0005; full precision)", "",
           "Gain = base log loss minus version log loss on matched held-out observations (positive = the version "
           "predicts better).", "", _md(_gate_table(res), "{:.6f}"), "", result, "",
           "Matched held-out and forward log loss:", "", _md(_loss_table(res), "{:.6f}"), "",
           "Per fold (reported, not gated):", "", _md(_fold_table(res), "{:.6f}"), "",
           "## First pitch of the PA vs. later pitches (reported, not gated)", "",
           "First pitches have prev = `none` at 0-0. M2b equals M2a exactly on them (the exception), so only the "
           "M2b-vs-M2a gain there is exactly 0, by construction. " + _first_pitch_line(res), "",
           _md(res["split"], "{:.6f}"), "",
           "## Fitted concentrations by role (outcome model, from M2b's roles)", "",
           "`league`/`cell`: P0's levels; `league_p`/`p`: M2a's; `league_pr`/`pr`: M2b's additional levels.", "",
           _md(_alpha_table(res), "{:.1f}"), "", *_bound_lines(res), "",
           "## Runtime", "",
           "; ".join(f"{m}: {s:.0f} s" for m, s in res["seconds"].items()) + " (M2_off is a full scored run; the "
           "others are prediction-only).", ""]
    return "\n".join(out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 8, sys.argv[3:] or None)
    else:
        {"gate": stage_gate}[stage]()
