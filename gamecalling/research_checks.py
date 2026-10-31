"""Research program S1-S3: bounded sensitivity checks on the frozen pilot P0 (project_plan.md, second amendment).

- S1: sample-restriction and training-exclusion checks. Each check states which of evaluated situations,
  probability training and payoff estimation it restricts; arsenal eligibility stays P0's.
  (a) bases-empty vs. runners-on ranking fits; (b) the addendum's sample restriction; (c) dugout-called games
  removed from everything (2025); (d) RE24 from all complete half-innings; (e) bases-empty probability training.
- S2: pooling strength: pitcher-cell concentrations (outcome and contact) or the policy concentration
  multiplied by 0.5 or 2 after estimation. A sensitivity analysis, not a correction.
- S3: pitch families instead of pitch types (F1, sensitivity only).

These checks identify sensitivity; they cannot by themselves establish why rankings differ. Everything is
descriptive, and 2024-2025 are previously examined seasons (exploratory reuse).

Run with: python3.12 -m gamecalling.research_checks models [WORKERS]   (fit the S1-S3 versions)
          python3.12 -m gamecalling.research_checks s1|s2|s3            (analysis and report)
"""

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from gamecalling import config, loader, models, payoff, research
from gamecalling import validation_pilot as vp
from gamecalling.phase1 import _md
from gamecalling.research import Spec

OUT = config.DATA_DIR / "research"
CONTEXT_SEED = 20261104
SCORES2 = (("D", "D"), ("A", "A"))


# --- Specification pieces (module level, so they pickle into worker processes) ---

def bases_empty(df: pd.DataFrame) -> np.ndarray:
    return df["base_out"].to_numpy() < 3


def drop_dugout_games(pitches, pas):
    games = set(pitches.loc[pitches["dugout_called"].astype(bool), "game_pk"])
    return pitches[~pitches["game_pk"].isin(games)], pas[~pas["game_pk"].isin(games)]


def re24_all_complete(pitches, pas):
    pas["re24_eligible"] = pas["half_inning_complete"]
    return pitches, pas


def to_families(pitches, pas):
    loader.set_action_unit(pitches, "family")
    return pitches, pas


def scaled_cell_models(train, factor):
    o, c = models.fit_outcome_model(train), models.fit_contact_model(train)
    return (models.fit_outcome_model(train, alphas=(o.alpha_league, factor * o.alpha_cell)),
            models.fit_contact_model(train, alphas=(c.alpha_league, factor * c.alpha_cell)))


def scaled_policy(train, units, factor):
    alpha = models.fit_policy(train, units).alpha
    return models.fit_policy(train, units, alpha=factor * alpha)


SPECS = {
    "S1c": Spec("S1c", "Dugout-called games removed from probability training, payoffs and evaluation (2025)",
                seasons=(2025,), transform=drop_dugout_games,
                params={"evaluated": "restricted", "probability_training": "restricted", "payoffs": "restricted"}),
    "S1d": Spec("S1d", "RE24 from all complete half-innings (including the 9th and later)",
                transform=re24_all_complete,
                params={"evaluated": "P0", "probability_training": "P0", "payoffs": "changed (RE24 sample)"}),
    "S1e": Spec("S1e", "Outcome, contact and policy models trained on bases-empty pitches; bases-empty evaluated",
                train_mask=bases_empty, eval_mask=bases_empty,
                params={"evaluated": "bases-empty", "probability_training": "bases-empty", "payoffs": "P0"}),
    "S2_cell_x0.5": Spec("S2_cell_x0.5", "Pitcher-cell alpha (outcome and contact) x 0.5 after estimation",
                         fit_models=partial(scaled_cell_models, factor=0.5), params={"cell_alpha_factor": 0.5}),
    "S2_cell_x2": Spec("S2_cell_x2", "Pitcher-cell alpha (outcome and contact) x 2 after estimation",
                       fit_models=partial(scaled_cell_models, factor=2.0), params={"cell_alpha_factor": 2.0}),
    "S2_policy_x0.5": Spec("S2_policy_x0.5", "Reference-policy alpha x 0.5 after estimation",
                           fit_policy=partial(scaled_policy, factor=0.5), params={"policy_alpha_factor": 0.5}),
    "S2_policy_x2": Spec("S2_policy_x2", "Reference-policy alpha x 2 after estimation",
                         fit_policy=partial(scaled_policy, factor=2.0), params={"policy_alpha_factor": 2.0}),
    "F1": Spec("F1", "Pitch families as the action unit (sensitivity only; never a candidate)",
               transform=to_families, recompute_support=True, params={"action_unit": "family"}),
}


def _run(model_id):
    t = time.perf_counter()
    man = research.run_model(SPECS[model_id])
    return model_id, time.perf_counter() - t, man["seconds"]


def stage_models(workers: int = 4, ids=None):
    ids = ids or list(SPECS)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
        for model_id, sec, s in ex.map(_run, ids):
            print(f"{model_id}: {sec:.0f} s (pipeline {s['pipeline']:.0f} s)", flush=True)


# --- Shared pieces ---

def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def _load(name):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def predictions(cand_id: str) -> dict:
    return {kind: research.prediction_compare("P0", cand_id, kind) for kind in ("outcome", "contact")}


def _pred_table(pred: dict) -> pd.DataFrame:
    rows = []
    for kind, c in pred.items():
        for r in c["season"].itertuples():
            rows.append({"model": kind, "sample": str(r.season), "n_matched": r.n, "n_P0": r.n_base,
                         "n_version": r.n_cand, "P0": r.base, "version": r.cand, "gain": r.gain,
                         "max_cal_P0": r.cal_base, "max_cal_version": r.cal_cand})
        if "forward" in c:
            f = c["forward"]
            rows.append({"model": kind, "sample": "forward 2024->2025", "n_matched": f["n"], "P0": f["base"],
                         "version": f["cand"], "gain": f["gain"], "max_cal_version": f["cal_cand"]})
    out = pd.DataFrame(rows)
    for c in ("n_P0", "n_version"):
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    out["max_cal_P0"] = out["max_cal_P0"].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    return out


def _fold_table(pred: dict) -> pd.DataFrame:
    return pd.concat([c["fold"].assign(model=k) for k, c in pred.items()], ignore_index=True)[
        ["model", "season", "fold", "n", "base", "cand", "gain"]].rename(columns={"base": "P0", "cand": "version"})


RANK_COLS = ["season", "x", "y", "n", "pearson", "spearman", "median_abs_diff", "max_abs_diff", "rank_moves",
             "sd_x", "sd_y", "converged"]


def _rank_md(res: dict) -> str:
    t = res["table"]
    return _md(t[[c for c in RANK_COLS if c in t]])


def _coverage_diff(cand_id: str, seasons) -> pd.DataFrame:
    a = research.load("P0", "coverage").rename(columns={"pitches": "P0"})
    b = research.load(cand_id, "coverage").rename(columns={"pitches": "version"})
    m = a[a["season"].isin(seasons)].merge(b, on=["season", "status"], how="outer").fillna(0)
    m["change"] = m["version"] - m["P0"]
    tot = m.groupby("season")[["P0", "version"]].transform("sum")
    m["share_P0"], m["share_version"] = m["P0"] / tot["P0"], m["version"] / tot["version"]
    return m.sort_values(["season", "P0"], ascending=[True, False])


GATE_NOTE = ("Prediction columns are log loss on matched held-out observations (outcome: nats per fit-eligible "
             "pitch; contact: nats per contact-fit-eligible ball in play); gain = P0 minus version (positive = "
             "the version predicts better). These S1-S3 versions are sensitivity checks, not candidates, so the "
             "prediction gate is not applied to them.")
RANK_NOTE = ("Ranking columns: catcher effects (runs saved per 100 pitches, random-effects model) fitted on "
             "samples x and y, compared on one common set (qualifying in the season, in the main group of the "
             "full season and of each restricted sample, at least 500 pitches in each restricted sample). "
             "Differences are after centering; rank moves count catchers moving 10 or more places within the "
             "common set. Descriptive only; seasons previously examined.")


# --- S1 ---

def addendum_flags(pitches: pd.DataFrame, pas: pd.DataFrame) -> pd.DataFrame:
    """Per PA: base-out change within the PA, censored endpoint, and an excluded decision (any pitch row
    that is not scored: exclusion or arsenal reason)."""
    pa = pitches["pa_id"]
    within = (pitches["bases_changed_before"] | pitches["outs_added_before"].ne(0)).groupby(pa).any()
    excluded = (~pitches["scorable"].astype(bool)).groupby(pa).any()
    out = pd.DataFrame({"base_out_change": within, "excluded_decision": excluded})
    out = out.join(pas.set_index("pa_id")["censored"])
    out["any"] = out.any(axis=1)
    return out


def stage_s1():
    t0 = time.perf_counter()
    pitches, pas, _ = research.base_tables()
    P0 = research.load("P0", "scores")
    res, cache = {}, {}

    # (a) complementary situation split, plus rough context
    empty = bases_empty(P0)
    res["a"] = research.rank_compare(P0, P0[empty], P0[~empty], SCORES2, "a", cache, "bases_empty", "runners_on")
    rng_parts = []
    for season in config.SEASONS:
        s = P0[P0["season"] == season]
        rng = np.random.default_rng([CONTEXT_SEED, season])
        rng_parts.append(pd.Series(rng.random(len(s)) < bases_empty(s).mean(), index=s.index))
    rnd = pd.concat(rng_parts).reindex(P0.index)
    res["a_context_random"] = research.rank_compare(P0, P0[rnd], P0[~rnd], SCORES2, "a_ctx", cache,
                                                    "random_share_empty", "random_rest")
    res["a_shares"] = P0.assign(e=empty).groupby("season")["e"].mean()
    v2 = vp._load("v2")
    res["a_context_halves"] = pd.DataFrame([{"season": se, **r} for se, d in v2["split_random"].items()
                                            for r in d["compare"] if r["model"] == "random" and r["score"] in "DA"])
    res["r4"] = v2["sensitivity"]["table"]

    # (b) addendum sample restriction (ranking refit only)
    flags = addendum_flags(pitches, pas)
    pa_flag = P0["pa_id"].map(flags["any"]).fillna(False).astype(bool)
    res["b_counts"] = pd.DataFrame([{"season": se, "reason": col, "PAs": int(flags.loc[ids, col].sum()),
                                     "scored_pitches_removed": int(P0.loc[(P0["season"] == se)
                                                                          & P0["pa_id"].map(flags[col]).fillna(False)
                                                                          .astype(bool)].shape[0])}
                                    for se in config.SEASONS
                                    for ids in [P0.loc[P0["season"] == se, "pa_id"].unique()]
                                    for col in ("base_out_change", "censored", "excluded_decision", "any")])
    res["b"] = research.rank_compare(P0, P0, P0[~pa_flag], SCORES2, "b", cache, "P0", "restricted")

    # (c) dugout-called games removed everywhere (2025)
    c = research.load("S1c", "scores")
    P0_25 = P0[P0["season"] == 2025]
    res["c_pred"] = predictions("S1c")
    res["c"] = research.rank_compare(P0_25, P0_25, c, SCORES2, "c", cache, "P0", "S1c")
    res["c_vs_R3"] = research.rank_compare(P0_25, P0_25[~P0_25["dugout_called"].astype(bool)], c, SCORES2,
                                           "c_r3", cache, "R3_sample", "S1c")
    res["c_coverage"] = _coverage_diff("S1c", (2025,))

    # (d) RE24 from all complete half-innings
    d = research.load("S1d", "scores")
    res["d_pred"] = predictions("S1d")
    res["d"] = research.rank_compare(P0, P0, d, SCORES2 + (("D_1p", "D_1p"),), "d", cache, "P0", "S1d")
    rows = []
    for season in config.SEASONS:
        for f in range(config.N_FOLDS):
            a = pas[(pas["season"] == season) & (pas["fold"] == f)]
            r0 = payoff.re24_table(a)
            r1 = payoff.re24_table(a.assign(re24_eligible=a["half_inning_complete"]))
            rows.append({"season": season, "fold": f, "pas_P0": int(a["re24_eligible"].sum()),
                         "pas_S1d": int(a["half_inning_complete"].sum()), "re24_empty0_P0": r0[0],
                         "re24_empty0_S1d": r1[0], "max_abs_change": float(np.abs(r1 - r0).max()),
                         "state_of_max": int(np.abs(r1 - r0).argmax())})
    res["d_re24"] = pd.DataFrame(rows)

    # (e) bases-empty probability training, bases-empty evaluated
    e = research.load("S1e", "scores")
    e_empty = e[bases_empty(e)]
    res["e_pred"] = predictions("S1e")
    res["e_vs_P0"] = research.rank_compare(P0, P0, e_empty, SCORES2, "e", cache, "P0", "S1e")
    res["e_vs_R4"] = research.rank_compare(P0, P0[empty], e_empty, SCORES2, "e_r4", cache, "R4_sample", "S1e")
    res["seconds"] = time.perf_counter() - t0
    _save("s1", res)
    S1_REPORT.write_text(s1_report(res))
    print(f"stage s1 done ({res['seconds']:.0f} s)", flush=True)


S1_REPORT = config.REPORTS_DIR / "research_s1.md"


def s1_report(res) -> str:
    out = ["# Research program S1: sample-restriction and training-exclusion checks on P0", "",
           "Generated by `python3.12 -m gamecalling.research_checks s1` (versions fitted by `... models`). "
           "Plan: `project_plan.md`, second amendment, S1. **These checks identify sensitivity; on their own they "
           "cannot establish why the original rankings differed.** All results are descriptive, on seasons "
           "previously examined (exploratory reuse). Arsenal eligibility is P0's in every check.", "",
           "| Check | Evaluated situations | Probability training | Payoff estimation |", "|---|---|---|---|",
           "| (a) bases-empty vs. runners-on ranking fits | restricted (each subset) | P0 | P0 |",
           "| (b) addendum sample restriction | restricted | P0 | P0 |",
           "| (c) dugout-called games removed (2025) | restricted | restricted | restricted |",
           "| (d) RE24 from all complete half-innings | P0 | P0 | changed |",
           "| (e) bases-empty probability training | bases-empty | bases-empty | P0 |", "",
           RANK_NOTE, "", GATE_NOTE, ""]

    sh = res["a_shares"]
    out += ["## (a) Complementary situation split", "",
            f"Bases-empty share of scored pitches: {sh[2024]:.3f} (2024), {sh[2025]:.3f} (2025). Catcher effects "
            "fitted separately on the two disjoint subsets of P0's scores.", "", _rank_md(res["a"]), "",
            "**Rough context only** (neither matches the situation split's pitcher assignments or state "
            "distributions; the random halves also differ in size):", "",
            "- A seeded random pitch split with the same proportions (seed "
            f"{CONTEXT_SEED}, per season):", "", _rank_md(res["a_context_random"]), "",
            "- The random game halves of the validation run (§3a, already reported):", "",
            _md(res["a_context_halves"][["season", "score", "n", "pearson", "spearman", "median_abs_diff"]]), "",
            "For reference, R4 (P0 on all pitches vs. P0 on bases-empty pitches; validation §5b):", "",
            _md(res["r4"][res["r4"]["check"].str.startswith("R4")][
                ["season", "check", "n", "pearson", "spearman", "median_abs_diff", "rank_moves"]]), ""]

    out += ["## (b) Addendum sample restriction", "",
            "PAs removed from the ranking sample: a base-out change between pitches within the PA (bases or outs; "
            "runs alone are covered by these), a censored endpoint (walk-off or called game), or an excluded "
            "decision (any pitch row of the PA that is not scored, e.g. automatic calls, intentional walks or "
            "out-of-arsenal pitches). Models and payoffs are P0's.", "", _md(res["b_counts"]), "",
            _rank_md(res["b"]), ""]

    out += ["## (c) Dugout-called games removed everywhere (2025)", "",
            "The nine MIA games flagged as dugout-called (from 2025-09-19, per config) are removed from probability "
            "training, payoff estimation and evaluation. 2024 is unaffected.", "",
            "Prediction (matched; the removed games' pitches are not in the version's held-out set):", "",
            _md(_pred_table(res["c_pred"]), "{:.6f}"), "", "Against P0 on all of 2025:", "", _rank_md(res["c"]), "",
            "Against P0's scores without those games (R3's ranking sample), which isolates the training and payoff "
            "change:", "", _rank_md(res["c_vs_R3"]), "", "Coverage:", "",
            _md(res["c_coverage"][["season", "status", "P0", "version", "change"]]), ""]

    out += ["## (d) RE24 from all complete half-innings", "",
            "`re24_eligible` widened from complete half-innings in innings 1-8 to all complete half-innings. This "
            "changes RE24, the strikeout and walk values, the ball-in-play payoffs (through RE24 after the play) "
            "and the RE288 table of the one-pitch benchmark. Probability training is unchanged, so held-out "
            "predictions should be identical (check below).", "", _md(res["d_re24"]), "",
            _md(_pred_table(res["d_pred"]), "{:.6f}"), "", _rank_md(res["d"]), ""]

    out += ["## (e) Bases-empty probability training", "",
            "Outcome, contact and policy models trained on bases-empty pitches of each role's fold; payoffs (and "
            "RE288) from the whole fold, because bases-empty decisions still end in runner states. Evaluated on "
            "bases-empty pitches. Prediction is compared with P0 on the same bases-empty held-out pitches.", "",
            _md(_pred_table(res["e_pred"]), "{:.6f}"), "", "Per fold:", "",
            _md(_fold_table(res["e_pred"]), "{:.6f}"), "",
            "Against P0 on all pitches (the R4 comparison, now with restricted training too):", "",
            _rank_md(res["e_vs_P0"]), "",
            "Against P0's scores on bases-empty pitches (R4's sample), which isolates the training change:", "",
            _rank_md(res["e_vs_R4"]), "",
            "Reading allowed: how much of R4's difference remains when probability training is also restricted. "
            "Not allowed: why the rankings differ.", ""]
    return "\n".join(out)


# --- S2 ---

S2_IDS = ("S2_cell_x0.5", "S2_cell_x2", "S2_policy_x0.5", "S2_policy_x2")
S2_REPORT = config.REPORTS_DIR / "research_s2.md"


def stage_s2():
    t0 = time.perf_counter()
    P0 = research.load("P0", "scores")
    res, cache = {"pred": {}, "rank": {}, "alphas": {}}, {}
    for mid in S2_IDS:
        if mid.startswith("S2_cell"):
            res["pred"][mid] = predictions(mid)
        res["rank"][mid] = research.rank_compare(P0, P0, research.load(mid, "scores"), SCORES2, mid, cache,
                                                 "P0", mid)
        man = research.load(mid, "manifest")
        res["alphas"][mid] = (pd.DataFrame(man["alphas"]), pd.DataFrame(man["policy_alphas"]))
    p0 = research.load("P0", "manifest")
    res["alphas"]["P0"] = (pd.DataFrame(p0["alphas"]), pd.DataFrame(p0["policy_alphas"]))
    res["seconds"] = time.perf_counter() - t0
    _save("s2", res)
    S2_REPORT.write_text(s2_report(res))
    print(f"stage s2 done ({res['seconds']:.0f} s)", flush=True)


def s2_report(res) -> str:
    a0, pol0 = res["alphas"]["P0"]
    summ = a0.groupby("season")[["outcome_cell", "contact_cell"]].agg(["min", "max"]).round(1)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    pol = pol0.groupby("season")["alpha"].agg(["min", "max"]).round(1).rename(columns=lambda c: f"policy_{c}")
    out = ["# Research program S2: pooling-strength sensitivity on P0", "",
           "Generated by `python3.12 -m gamecalling.research_checks s2`. After the usual marginal-likelihood "
           "estimate in each role, the pitcher-cell concentration of the outcome and contact models (league "
           "level unchanged), or the reference-policy concentration, is multiplied by a fixed factor. The "
           "production estimator is unchanged. **A sensitivity analysis, not a correction factor**, and unrelated "
           "to the uncertainty specifications' rule against fixing alpha. Descriptive; seasons previously "
           "examined.", "", "P0's estimated concentrations (range over the three role folds):", "",
           _md(summ.join(pol).reset_index(), "{:.1f}"), "", GATE_NOTE, "",
           "Policy factors change no predicted probability (the policy is only the continuation and the A "
           "baseline), so only the cell factors have a prediction table.", ""]
    for mid in S2_IDS:
        out += [f"## {mid}", ""]
        if mid in res["pred"]:
            out += [_md(_pred_table(res["pred"][mid]), "{:.6f}"), "", "Per fold:", "",
                    _md(_fold_table(res["pred"][mid]), "{:.6f}"), ""]
        out += [RANK_NOTE if mid == S2_IDS[0] else "", "", _rank_md(res["rank"][mid]), ""]
    out += ["## Not run", "",
            "The optional add-on (observable checks (c) dispersion against cell size and (d) correlation across "
            "counts from `research_notes/uncertainty_design/07_synthesis.md`) was not run; the user listed no "
            "optional work as required now.", ""]
    return "\n".join(out)


# --- S3 ---

S3_REPORT = config.REPORTS_DIR / "research_s3.md"


def relabel_pitcher_seasons(pitches: pd.DataFrame, max_share: float = 0.25) -> pd.DataFrame:
    """Pitcher-seasons throwing both SL and ST, with both in fewer than max_share of the games in which they
    threw either (Phase 0's definition: 334 pitcher-seasons throw both, median 63%, 92 under 25%)."""
    t = pitches[pitches["thrown"] & pitches["pitch_type"].isin(["SL", "ST"])]
    g = t.groupby(["season", "pitcher", "game_pk"])["pitch_type"].nunique()
    per = pd.DataFrame({"games_sl_or_st": g.groupby(level=[0, 1]).size(),
                        "games_both": (g == 2).groupby(level=[0, 1]).sum()})
    throws_both = t.groupby(["season", "pitcher"])["pitch_type"].nunique() == 2
    per = per[throws_both.reindex(per.index)]
    per["share_both"] = per["games_both"] / per["games_sl_or_st"]
    return per[per["share_both"] < max_share].reset_index()


def stage_s3():
    t0 = time.perf_counter()
    pitches = research.base_tables()[0]
    P0, F1 = research.load("P0", "scores"), research.load("F1", "scores")
    res, cache = {}, {}
    res["pred"] = predictions("F1")
    res["coverage"] = _coverage_diff("F1", config.SEASONS)
    res["rank"] = research.rank_compare(P0, P0, F1, SCORES2, "F1", cache, "P0", "F1")
    rel = relabel_pitcher_seasons(pitches)
    res["relabel_n"] = rel.groupby("season").size()
    key = set(zip(rel["season"], rel["pitcher"]))
    P0e = P0.assign(rel=[k in key for k in zip(P0["season"], P0["pitcher"])])
    expo = P0e.groupby(["season", "catcher"])["rel"].mean().rename("exposure")
    rows = []
    for season in config.SEASONS:
        for sx, _ in SCORES2:
            fx, fy = cache["F1", season, "P0", sx]["saved"], cache["F1", season, "F1", sx]["saved"]
            common = res["rank"]["common"][season]  # the main comparison's common set (§5b rules)
            a, b = fx[common] - fx[common].mean(), fy[common] - fy[common].mean()
            e = expo.loc[season].reindex(common)
            ch = (b - a).abs()
            rows.append({"season": season, "score": sx, "n": len(common), "median_exposure": float(e.median()),
                         "max_exposure": float(e.max()), "corr_exposure_abs_change": float(np.corrcoef(e, ch)[0, 1]),
                         "median_abs_change_top10_exposure": float(ch[e.nlargest(10).index].median()),
                         "median_abs_change_rest": float(ch.drop(e.nlargest(10).index).median())})
    res["relabel"] = pd.DataFrame(rows)
    res["seconds"] = time.perf_counter() - t0
    _save("s3", res)
    S3_REPORT.write_text(s3_report(res))
    print(f"stage s3 done ({res['seconds']:.0f} s)", flush=True)


def s3_report(res) -> str:
    rn = res["relabel_n"]
    cov = res["coverage"]
    out = ["# Research program S3: pitch families vs. pitch types (F1, sensitivity)", "",
           "Generated by `python3.12 -m gamecalling.research_checks s3`. F1 rebuilds the action unit as pitch "
           "families (FF/SI, FC, SL/ST/SV, CU/KC/CS, CH/FS/FO; EP, KN and SC have no family) through the whole "
           "pipeline: arsenal support (recomputed under the same rule), models, policy, solver and scoring. "
           "**Pitch types stay the locked action unit**; F1 is a robustness check only. Descriptive; seasons "
           "previously examined.", "",
           "## Coverage (reported separately from prediction)", "",
           _md(cov[["season", "status", "P0", "version", "change", "share_P0", "share_version"]], "{:.4f}"), "",
           "## Prediction on matched observations", "",
           "The two versions predict different conditional quantities (outcome given pitch type vs. given "
           "family); the comparison is on the same pitches, held out under both.", "",
           GATE_NOTE, "", _md(_pred_table(res["pred"]), "{:.6f}"), "", "Per fold:", "",
           _md(_fold_table(res["pred"]), "{:.6f}"), "",
           "## Catcher scores", "", RANK_NOTE, "", _rank_md(res["rank"]), "",
           "## Slider/sweeper relabeling exposure", "",
           f"Pitcher-seasons throwing both SL and ST, with both in under 25% of the games in which they threw either: {int(rn.get(2024, 0))} "
           f"(2024), {int(rn.get(2025, 0))} (2025); {int(rn.sum())} in total (Phase 0's figure, same definition). A catcher's "
           "exposure is the share of the catcher's P0 scored pitches thrown by those pitcher-seasons. Absolute change = "
           "|F1 - P0| after centering on, and computed over, the same common catcher set as the main F1 comparison above "
           "(qualifying, in the main connected group of both versions). Descriptive.", "",
           _md(res["relabel"], "{:.4f}"), ""]
    return "\n".join(out)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "models":
        stage_models(int(sys.argv[2]) if len(sys.argv) > 2 else 4, sys.argv[3:] or None)
    else:
        {"s1": stage_s1, "s2": stage_s2, "s3": stage_s3}[stage]()
