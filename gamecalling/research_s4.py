"""Research program S4: runners-on flag (M1) under the prediction gate (project_plan.md, second amendment).

Specification: research_notes/research_program/S4_runners_on.md (revision 1); model code in runners.py.
- M1_off: the extension explicitly disabled (runners=False); must reproduce P0's stored scores and
  predictions exactly (§3.5 test 1, full-run form).
- M1: runner levels on both the outcome and the contact model.
The gate is applied component by component (outcome per fit-eligible pitch, contact per ball in play;
delta 0.0005; full precision; both seasons; calibration <= 0.02; forward log loss no higher than P0's). Each
passing component is adopted and each failing one reverts to P0's; if neither passes, M1 is not adopted and
not scored. One run of the gate; no variant is tried after a failure (§6). Descriptive; 2024-2025 are
previously examined seasons (exploratory reuse).

For the scored version (the adopted components; M1c when only contact passes), descriptively: D, A and B
changes against P0 on the common catcher set, decision relevance, R4 rerun under the version, coverage.

Run with: python3.12 -m gamecalling.research_s4 run [WORKERS]   (fit M1_off and M1)
          python3.12 -m gamecalling.research_s4 gate             (checks, gate and report)
          python3.12 -m gamecalling.research_s4 describe         (scored version: run, analyses, report)
"""

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, research, runners, scoring
from gamecalling.phase1 import SCORES_PARQUET, _md
from gamecalling.research import P0, Spec
from gamecalling.research_checks import (GATE_NOTE, RANK_NOTE, SCORES2, _coverage_diff, _fold_table, _load,
                                         _pred_table, _rank_md, _save, bases_empty, predictions)

SPECS = {
    "M1_off": Spec("M1_off", "M1 with the runners-on extension explicitly disabled (runners=False): P0's own "
                   "functions only", fit_models=partial(runners.fit_models, runners=False),
                   solve_role=partial(runners.solve_role, runners=False), training=runners.training,
                   params={"runners": False}),
    "M1": Spec("M1", "Runners-on flag: league runner level and ratio-shifted runner cells on the outcome and "
               "contact models", fit_models=runners.fit_models, solve_role=runners.solve_role,
               training=runners.training, params={"runners": True, "components": list(runners.COMPONENTS)}),
}
# Component-wise adoption: the scored version for each set of passing components.
SCORED = {
    ("contact", "outcome"): "M1",
    ("contact",): "M1c",
    ("outcome",): "M1o",
}
SPECS["M1c"] = Spec("M1c", "Runners-on flag on the contact model only (component-wise adoption: the outcome "
                    "component failed the gate and is P0's)",
                    fit_models=partial(runners.fit_models, components=("contact",)), solve_role=runners.solve_role,
                    training=runners.training, params={"runners": True, "components": ["contact"]})
SPECS["M1o"] = Spec("M1o", "Runners-on flag on the outcome model only (component-wise adoption: the contact "
                    "component failed the gate and is P0's)",
                    fit_models=partial(runners.fit_models, components=("outcome",)), solve_role=runners.solve_role,
                    training=runners.training, params={"runners": True, "components": ["outcome"]})


def _run(model_id):
    t = time.perf_counter()
    man = research.run_model(SPECS[model_id])
    return model_id, time.perf_counter() - t, man["seconds"]


GATE_IDS = ("M1_off", "M1")   # the runs the gate needs; the scored version is run by describe if it differs


def stage_run(workers: int = 8, ids=None):
    ids = ids or list(GATE_IDS)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
        for model_id, sec, s in ex.map(_run, ids):
            print(f"{model_id}: {sec:.0f} s (pipeline {s['pipeline']:.0f} s)", flush=True)


# --- Checks and gate ---

def _same(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False, f"shape/columns differ: {a.shape} vs {b.shape}"
    diff = [c for c in b.columns if not a[c].equals(b[c])]
    return not diff, ", ".join(diff) or "identical"


def disabled_checks() -> pd.DataFrame:
    """Test 1, full-run form: M1_off against P0's stored scores and the verified P0 predictions."""
    rows = []
    ok, detail = _same(research.load("M1_off", "scores"), pd.read_parquet(SCORES_PARQUET))
    rows.append({"check": "scores vs data/phase1_scores.parquet (all rows, all columns)",
                 "rows": len(research.load("M1_off", "scores")), "exact": ok, "detail": detail})
    for what in ("heldout_outcome", "heldout_contact", "forward_outcome", "forward_contact"):
        a, b = research.load("M1_off", what), research.load("P0", what)
        ok, detail = _same(a, b)
        rows.append({"check": f"{what} vs P0_repro", "rows": len(a), "exact": ok, "detail": detail})
    return pd.DataFrame(rows)


def alpha_table(model_id: str) -> pd.DataFrame:
    return pd.DataFrame(research.load(model_id, "manifest")["alphas"])


def stage_gate():
    t0 = time.perf_counter()
    res = {"disabled": disabled_checks()}
    if not res["disabled"]["exact"].all():
        raise RuntimeError("M1_off does not reproduce P0 exactly; fix before applying the gate")
    res["pred"] = predictions("M1")
    res["gate"] = pd.DataFrame([research.gate(c) for c in res["pred"].values()])
    res["adopted"] = [k for k, g in zip(res["gate"]["kind"], res["gate"]["passes"]) if g]
    res["alphas"] = alpha_table("M1")
    res["manifests"] = {m: research.load(m, "manifest") for m in GATE_IDS}   # the scored version's: describe
    res["seconds"] = time.perf_counter() - t0
    _save("s4", res)
    S4_REPORT.write_text(s4_report(res))
    print(f"stage s4 gate done ({res['seconds']:.0f} s); adopted components: {res['adopted'] or 'none'}",
          flush=True)
    return res


# --- Scored version: descriptive measures (never used for adoption) ---

def pair_change_sd(Qb: np.ndarray, Qc: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pitch: mean over its eligible ordered pairs (a, b) of [(Qc(a) - Qc(b)) - (Qb(a) - Qb(b))]^2, and the
    same for Qb(a) - Qb(b) itself (scale). With both orderings the mean change is 0, and the mean square is
    2 x the sample variance over the eligible actions. NaN for pitches with one eligible action."""
    def msq(x):
        x = np.where(eligible, x, np.nan)
        m = eligible.sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # rows with no eligible pair
            return np.where(m >= 2, 2 * np.nanvar(x, axis=1) * m / np.maximum(m - 1, 1), np.nan)
    Qb, Qc = np.where(eligible, Qb, 0.0), np.where(eligible, Qc, 0.0)   # ineligible Q is inf
    return msq(Qc - Qb), msq(Qb)


def decision_relevance(cand: Spec) -> pd.DataFrame:
    """Share of scored decisions whose recommended pitch changes, and the pitch-weighted SD (each scored pitch
    weight 1) of the change in Q_value(a) - Q_value(b) over its eligible pairs; P0's own SD of Q(a) - Q(b) as
    scale. Runs per pitch decision."""
    rows = []
    pitches = research.base_tables()[0]   # neither version transforms the data
    for season in config.SEASONS:
        sols = {}
        for name, spec in (("P0", P0), ("cand", cand)):
            pp, pas, support = research.prepared(spec)
            run = scoring.run_pilot(pp, pas, support, seasons=(season,), fit_role=research.role_fitter(spec),
                                    fit_policy=spec.fit_policy, solve_role=spec.solve_role, keep_solutions=True)
            sols[name] = run.solutions
        for k in range(config.N_FOLDS):
            b, c = sols["P0"][season, k], sols["cand"][season, k]
            units = b["units"]
            if not units.slots.equals(c["units"].slots):
                raise ValueError("versions differ in eligible sets")
            p = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
            p = p.merge(units.slots, on=["pitcher", "stand", "action"], how="left", validate="many_to_one")
            u, bo = p["u"].to_numpy(int), p["base_out"].to_numpy(int)
            cc = (p["balls"] * 3 + p["strikes"]).to_numpy(int)
            elig = units.eligible[u]
            rec_b = b["select"].Q_ref[u, bo, cc].argmin(1)
            rec_c = c["select"].Q_ref[u, bo, cc].argmin(1)
            ch, scale = pair_change_sd(b["value"].Q_ref[u, bo, cc], c["value"].Q_ref[u, bo, cc], elig)
            rows.append({"season": season, "fold": k, "pitches": len(p), "rec_changed": int((rec_b != rec_c).sum()),
                         "pitches_with_pairs": int(np.isfinite(ch).sum()), "msq_change": float(np.nanmean(ch)),
                         "msq_P0": float(np.nanmean(scale))})
        del sols
    f = pd.DataFrame(rows)
    out = []
    for season, g in f.groupby("season"):
        w = g["pitches_with_pairs"]
        out.append({"season": season, "pitches": int(g["pitches"].sum()),
                    "share_rec_changed": g["rec_changed"].sum() / g["pitches"].sum(),
                    "sd_pair_change": float(np.sqrt((g["msq_change"] * w).sum() / w.sum())),
                    "sd_pair_P0": float(np.sqrt((g["msq_P0"] * w).sum() / w.sum()))})
    return pd.DataFrame(out).assign(ratio=lambda d: d["sd_pair_change"] / d["sd_pair_P0"])


def stage_describe():
    t0 = time.perf_counter()
    res = _load("s4")
    if not res["adopted"]:
        print("no component adopted; M1 is not scored", flush=True)
        return res
    mid = SCORED[tuple(sorted(res["adopted"]))]
    if mid != "M1":
        stage_run(1, [mid])
    spec = SPECS[mid]
    P0s, V = research.load("P0", "scores"), research.load(mid, "scores")
    cache = {}
    d = {"id": mid, "description": spec.description}
    d["rank"] = research.rank_compare(P0s, P0s, V, SCORES2 + (("B", "B"),), "scored", cache, "P0", mid)
    d["r4_version"] = research.rank_compare(V, V, V[bases_empty(V)], SCORES2, "r4_v", cache, f"{mid}_all",
                                            f"{mid}_bases_empty")
    d["r4_P0"] = research.rank_compare(P0s, P0s, P0s[bases_empty(P0s)], SCORES2, "r4_p0", cache, "P0_all",
                                       "P0_bases_empty")
    d["relevance"] = decision_relevance(spec)
    d["coverage"] = _coverage_diff(mid, config.SEASONS)
    d["alphas"] = alpha_table(mid)
    d["manifest"] = research.load(mid, "manifest")
    d["seconds"] = time.perf_counter() - t0
    res["scored"] = d
    res["manifests"][mid] = d["manifest"]
    _save("s4", res)
    S4_REPORT.write_text(s4_report(res))
    print(f"stage s4 describe done ({d['seconds']:.0f} s)", flush=True)
    return res


# --- Report ---

S4_REPORT = config.REPORTS_DIR / "research_s4.md"


def _gate_rows(res) -> pd.DataFrame:
    rows = []
    for g, (kind, c) in zip(res["gate"].itertuples(), res["pred"].items()):
        s = c["season"].set_index("season")
        rows.append({"component": kind, "gain_2024": s.loc[2024, "gain"], "gain_2025": s.loc[2025, "gain"],
                     "delta": g.delta, "both_seasons": g.both_seasons,
                     "max_cal": float(s["cal_cand"].max()), "calibration": g.calibration,
                     "forward_gain": c["forward"]["gain"], "forward": g.forward, "passes": g.passes})
    return pd.DataFrame(rows)


def gate_reading(res) -> list:
    al = res["alphas"]
    out = []
    for kind, c in res["pred"].items():
        g = res["gate"].set_index("kind").loc[kind]
        if not g["passes"]:
            folds = c["fold"]
            out.append(f"- **{kind} fails the gate** and reverts to P0's: matched gain "
                       f"{c['season']['gain'].min():.6f} to {c['season']['gain'].max():.6f} against delta "
                       f"{g['delta']:.4f} (forward gain {c['forward']['gain']:.6f}; fold gains "
                       f"{folds['gain'].min():.6f} to {folds['gain'].max():.6f}, reported, not gated). Recorded as a "
                       "negative result under the stopping rule; no variant is tried (§6).")
    bound = float(np.exp(models.pooling.LOG_ALPHA_BOUNDS[1]))
    for kind in ("outcome", "contact"):
        a = al[f"{kind}_alpha_r"]
        out.append(f"- {kind} runner-cell concentration alpha_r: {a.min():,.0f} to {a.max():,.0f} across the six "
                   f"role folds, at or within {1 - a.min() / bound:.1%} of the optimizer's upper bound ({bound:,.0f}). "
                   "A runner cell's own counts enter with weight N_r / (N_r + alpha_r), which is small but not zero "
                   "at these values: pitcher-specific runner deviations are heavily shrunk, and predictions are "
                   "approximately the league runner adjustment L_r / L applied to each pitcher's P0 cell. This does "
                   "not establish that pitcher-specific runner effects are absent.")
    return out


def s4_report(res) -> str:
    dis = res["disabled"]
    al = res["alphas"]
    cols = ["season", "fold", "outcome_league", "outcome_cell", "outcome_alpha_league_r", "outcome_alpha_r",
            "contact_league", "contact_cell", "contact_alpha_league_r", "contact_alpha_r"]
    secs = {m: man["seconds"]["total"] for m, man in res["manifests"].items()}
    adopted = res["adopted"]
    out = ["# Research program S4: runners-on flag (M1)", "",
           "Generated by `python3.12 -m gamecalling.research_s4 gate` (versions fitted by `... run`). "
           "Specification: `research_notes/research_program/S4_runners_on.md` (revision 1); model code "
           "`gamecalling/runners.py`. Descriptive; 2024-2025 are previously examined seasons (exploratory reuse). "
           "One run of the gate (§6).", "",
           "## Test 1, full-run form: extension disabled = P0", "",
           "`M1_off` runs M1's code path with `runners=False` (P0's own fitting and solve functions, M1's "
           "observation hook).", "",
           _md(dis.assign(exact=dis["exact"].map({True: "yes", False: "NO"}))), "",
           "Unit tests (`tests/test_runners.py`): runner-cell formula against a hand computation (1e-12), the three "
           "sparse-cell fallbacks with positivity and summing to 1, solver slices (identical slices give P0's "
           "`solve_role` exactly; states 0-2 use runners_on = 0 and 3-23 runners_on = 1), the disabled switch, and "
           "the fold-checked matched set.", "",
           "## Prediction gate (component by component)", "", GATE_NOTE.replace(
               "These S1-S3 versions are sensitivity checks, not candidates, so the prediction gate is not applied "
               "to them.", "M1 is a candidate: the gate applies."), "",
           _md(_gate_rows(res), "{:.6f}"), "",
           f"**Result: {'adopted components: ' + ', '.join(adopted) if adopted else 'neither component passes; M1 is not adopted and not scored'}.**",
           "", *gate_reading(res), "", "Matched held-out and forward log loss:", "", _md(_pred_table(res["pred"]), "{:.6f}"), "",
           "Per fold (reported, not gated):", "", _md(_fold_table(res["pred"]), "{:.6f}"), "",
           "## Fitted concentrations by role", "",
           "P0's league and cell concentrations (unchanged, refit inside M1) and M1's league runner (`alpha_league_r`) "
           "and runner-cell (`alpha_r`) concentrations, per role fold.", "",
           _md(al[[c for c in cols if c in al]], "{:.1f}"), "",
           "## Runtime", "",
           "; ".join(f"{m}: {s:.0f} s" for m, s in secs.items()) + " (full run: fit, solve, score, held-out and "
           "forward predictions; budget 60 min).", ""]
    if "scored" in res:
        out += scored_report(res["scored"])
    return "\n".join(out)


def adopted_checks(d) -> list:
    """The scored version's adopted components predict exactly as in the gated M1 run."""
    out = []
    for kind in runners.COMPONENTS:
        if kind in d["manifest"]["params"]["components"]:
            for what in (f"heldout_{kind}", f"forward_{kind}"):
                ok, detail = _same(research.load(d["id"], what), research.load("M1", what))
                out.append(f"- {what} of {d['id']} vs. M1 (gated): {'identical' if ok else 'DIFFERS: ' + detail}.")
    return out


def scored_report(d) -> list:
    mid = d["id"]
    cov = d["coverage"]
    rel = d["relevance"]
    return ["## Scored version: " + mid, "", f"{d['description']}. Run `data/models/{mid}/` "
            f"({d['manifest']['seconds']['total']:.0f} s). Everything below is descriptive and was not used for "
            "adoption.", "", *adopted_checks(d), "",
            "### Catcher scores against P0 (D, A, B)", "", RANK_NOTE, "", _rank_md(d["rank"]), "",
            "### Decision relevance", "",
            "Share of scored decisions whose recommended pitch (selection role) changes; pitch-weighted SD (each "
            "scored pitch weight 1, spread over its eligible ordered pairs) of the change in Q_value(a) - "
            "Q_value(b), in runs per decision, with P0's own SD of Q_value(a) - Q_value(b) as scale. Pitches with "
            "one eligible action have no pairs.", "", _md(rel, "{:.6f}"), "",
            "### R4 under the scored version (all pitches vs. bases-empty pitches)", "",
            "R4 for P0 on the same rules, recomputed here for a like-for-like comparison:", "",
            _rank_md(d["r4_P0"]), "", f"R4 under {mid}:", "", _rank_md(d["r4_version"]), "",
            "### Coverage (expected unchanged)", "",
            _md(cov[["season", "status", "P0", "version", "change"]]), "",
            f"Total change: {int(cov['change'].abs().sum())} pitch rows.", "",
            "### Reading (descriptive)", "", *scored_reading(d), ""]


def scored_reading(d) -> list:
    r = d["rank"]["table"].set_index(["season", "x"])
    rel = d["relevance"].set_index("season")
    r4p, r4v = (d[k]["table"].assign(s=lambda t: t["x"].str.split(":").str[1]).set_index(["season", "s"])
                for k in ("r4_P0", "r4_version"))
    mid = d["id"]
    pear = lambda s: "/".join(f"{r.loc[(y, f'P0:{s}'), 'pearson']:.3f}" for y in config.SEASONS)
    r4 = lambda t, s: "/".join(f"{t.loc[(y, s), 'pearson']:.2f}" for y in config.SEASONS)
    return [f"- Catcher effects against P0 (Pearson, 2024/2025): D {pear('D')}, A {pear('A')}, B {pear('B')}.",
            f"- Recommended pitch changes in {rel['share_rec_changed'].min():.1%} to "
            f"{rel['share_rec_changed'].max():.1%} of scored decisions; the SD of the change in Q(a) - Q(b) is "
            f"{rel['ratio'].min():.1%} to {rel['ratio'].max():.1%} of P0's own SD of Q(a) - Q(b).",
            f"- R4 (all vs. bases-empty, Pearson 2024/2025): D {r4(r4p, 'D')} under P0, {r4(r4v, 'D')} under {mid}; "
            f"A {r4(r4p, 'A')} under P0, {r4(r4v, 'A')} under {mid}. The contact runner flag leaves the bases-empty "
            "sensitivity essentially where it was. This describes the comparison only; it does not establish why "
            "the rankings differ."]


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 8, sys.argv[3:] or None)
    else:
        {"gate": stage_gate, "describe": stage_describe}[stage]()
