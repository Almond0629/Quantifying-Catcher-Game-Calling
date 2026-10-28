"""Pilot validation runner (research_notes/validation/01_validation_protocol.md, revision 3, locked 2026-10-29).

Evaluates the existing pilot scores (D primary, A comparison, B diagnostic) through held-out prediction,
within-season and 2024-2025 repeatability, robustness, team and staff dependence, and baseball
interpretation. Everything is descriptive: no confidence intervals, no tests, no claims of validated
uncertainty. The Phase 1 checks stay in validation.py.

Stages (protocol §0):
- v0: game halves, calendar halves, mover groups (pitch counts and seeds only) and the frozen rules,
  saved with a manifest before any catcher score is joined to them;
- v2: forward prediction (§2b), within-season reliability (§3a, §3b), 2024-2025 repeatability (§4),
  ranking-sample sensitivity (§5b), team and staff dependence (§6);
- v3: examples selected by the frozen rules (§7);
- v4: report (with the D-vs.-A evidence table, §8).
(v1, the implementation checks, is tests/test_validation_pilot.py plus `smoke`.)

Run with: python3.12 -m gamecalling.validation_pilot STAGE [WORKERS]   (STAGE v0, smoke, v2, v3 or v4)
Outputs: data/validation_pilot/v1/ and reports/validation_pilot.md. Display: runs saved vs. average per
100 pitches (higher = better); internal effects are costs.
"""

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from gamecalling import arsenal, config, models, network, ranking, scoring, solver, validation
from gamecalling.phase1 import SCORES_PARQUET, _md

VERSION = "v1"
OUT = config.DATA_DIR / "validation_pilot" / VERSION
REPORT = config.REPORTS_DIR / "validation_pilot.md"
SEED = 20261103
SEASONS = tuple(config.SEASONS)
QUALIFY = network.QUALIFY          # 1,000 scorable pitches in the season
HALF_MIN = 500                     # per half (§3) and in a check's sample (§5b)
MOVER_TEAM_MIN = 300               # within-season mover: pitches with each of 2+ teams (§6)
NONMOVER_SHARE = 0.90              # non-mover: share with the primary team, each season (§6)
MIN_SET = 10                       # correlations only on sets of at least 10 catchers
RANK_MOVE = 10                     # rank moves counted at 10 places or more (§5b)
ZERO_SD = 1e-4                     # zero-variance fit: catcher SD below this (runs per 100 pitches)
SCORES = ("D", "A", "B")
CHECKS = {"R1": ("D", "D_opt", "all"), "R2": ("D", "D_1p", "all"), "R3_D": ("D", "D", "no_dugout"),
          "R3_A": ("A", "A", "no_dugout"), "R4_D": ("D", "D", "bases_empty"), "R4_A": ("A", "A", "bases_empty")}
N_CATCHER_EX, N_PITCH_EX = 3, 4
PITCH_EX_SEASON = 2024
UNDEFINED, UNAVAILABLE = "undefined (zero catcher variance)", "unavailable (zero catcher variance)"


def rules() -> dict:
    """Every frozen rule, threshold, seed and tie-break (protocol §1-§7), written to the v0 manifest."""
    return {"protocol": "research_notes/validation/01_validation_protocol.md revision 3 (locked 2026-10-29)",
            "version": VERSION, "seed": SEED, "seasons": list(SEASONS), "qualify": QUALIFY, "half_min": HALF_MIN,
            "mover_team_min": MOVER_TEAM_MIN, "nonmover_share": NONMOVER_SHARE, "min_set": MIN_SET,
            "rank_move": RANK_MOVE, "zero_sd": ZERO_SD, "min_pair": network.MIN_PAIR, "checks": CHECKS,
            "halves": "arsenal.assign_folds(pitches, n_folds=2, seed=SEED) -> rng [SEED, season]",
            "calendar": "game_date >= the season's median game date -> half 1",
            "half_refit_folds": "arsenal.assign_folds(half_pitches, seed=SEED + 10 + half)",
            "catcher_examples": f"{N_CATCHER_EX} largest and most negative rank_D - rank_A (ties: catcher id); "
                                f"{N_CATCHER_EX} random from top third on both, bottom third on both: "
                                "rng [SEED, 7, season, group], candidates sorted by catcher id",
            "pitch_examples": f"{PITCH_EX_SEASON}; {N_PITCH_EX} per category; rng [SEED, 8, category], candidates "
                              "sorted by (game_pk, at_bat_number, pitch_number); strict > 95th, < 5th, < median",
            "ranks": "ordinal by saved descending, ties by ascending catcher id; Spearman uses average ranks",
            "primary_team_ties": "more games caught, then alphabetical team code",
            "centering": "effects are centered on the compared catcher set's mean before differences"}


# --- Small pure helpers (tested in tests/test_validation_pilot.py) ---

def ordinal_ranks(saved: pd.Series) -> pd.Series:
    """Rank 1 = highest saved; ties by ascending catcher id."""
    order = sorted(saved.index, key=lambda c: (-saved[c], c))
    return pd.Series(np.arange(1, len(order) + 1), index=order).reindex(saved.index)


def compare(x: pd.Series, y: pd.Series, zero: bool) -> dict:
    """Descriptive agreement of two effect vectors (saved scale) on their common index."""
    idx = x.index.intersection(y.index)
    n = len(idx)
    if zero:
        return {"n": n, "pearson": UNDEFINED, "spearman": UNDEFINED, "median_abs_diff": np.nan}
    if n < MIN_SET:
        return {"n": n, "pearson": f"listed (n < {MIN_SET})", "spearman": f"listed (n < {MIN_SET})",
                "median_abs_diff": np.nan}
    a, b = x[idx] - x[idx].mean(), y[idx] - y[idx].mean()
    return {"n": n, "pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(stats.spearmanr(a, b)[0]),
            "median_abs_diff": float((a - b).abs().median())}


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))


def team_share(y: pd.Series, team: pd.Series):
    """Sum_t n_t (ybar_t - ybar)^2 / sum_c (y_c - ybar)^2; 'undefined' at a zero denominator."""
    y = y.astype(float)
    ybar = y.mean()
    den = float(((y - ybar) ** 2).sum())
    if den <= 0:
        return "undefined (zero denominator)"
    g = y.groupby(team.reindex(y.index))
    return float((g.size() * (g.mean() - ybar) ** 2).sum() / den)


def group_gap(sel: pd.Series, ev: pd.Series) -> dict:
    """Top and bottom thirds chosen on `sel` (ordinal ranks), their mean on `ev` (saved scale)."""
    idx = sel.index.intersection(ev.index)
    k = len(idx) // 3
    r = ordinal_ranks(sel[idx]).sort_values()
    top, bottom = r.index[:k], r.index[-k:]
    return {"k": k, "top_mean": float(ev[top].mean()), "bottom_mean": float(ev[bottom].mean()),
            "gap": float(ev[top].mean() - ev[bottom].mean())}


def primary_teams(counts: pd.DataFrame) -> pd.DataFrame:
    """counts: catcher, fielding_team, pitches, games (one season). Primary team with the protocol's ties."""
    c = counts.sort_values(["catcher", "pitches", "games", "fielding_team"], ascending=[True, False, False, True])
    first = c.groupby("catcher").head(1).set_index("catcher")
    tot = counts.groupby("catcher")["pitches"].sum()
    n_big = counts[counts["pitches"] >= MOVER_TEAM_MIN].groupby("catcher").size()
    return pd.DataFrame({"primary_team": first["fielding_team"], "primary_share": first["pitches"] / tot,
                         "pitches": tot, "within_mover": n_big.reindex(first.index).fillna(0) >= 2})


def mover_groups(prim: dict) -> pd.DataFrame:
    """prim: season -> primary_teams table. Groups for catchers qualifying in both seasons (§6)."""
    a, b = (prim[s] for s in SEASONS)
    both = a.index[a["pitches"] >= QUALIFY].intersection(b.index[b["pitches"] >= QUALIFY])
    out = pd.DataFrame(index=both)
    out["team_0"], out["team_1"] = a.loc[both, "primary_team"], b.loc[both, "primary_team"]
    out["share_0"], out["share_1"] = a.loc[both, "primary_share"], b.loc[both, "primary_share"]
    out["group"] = np.where(out["team_0"] != out["team_1"], "mover",
                            np.where((out["share_0"] >= NONMOVER_SHARE) & (out["share_1"] >= NONMOVER_SHARE),
                                     "non_mover", "neither"))
    return out


def main_group(df: pd.DataFrame, candidates) -> set:
    """Largest connected group among `candidates` (passed explicitly, never the 1,000 default)."""
    comps = network.components(network.catcher_graph(df, qualifying=set(candidates)))
    return set(comps[0]) if comps else set()


def draw(rng, candidates: list, k: int) -> list:
    """k without replacement from sorted candidates (all of them if fewer: a shortfall)."""
    candidates = sorted(candidates)
    if len(candidates) <= k:
        return candidates
    return [candidates[i] for i in sorted(rng.choice(len(candidates), k, replace=False))]


# --- Fits ---

def season_fit(df: pd.DataFrame, score: str, fixed: bool = False) -> dict:
    f = ranking.fit(df, score, pitcher_fixed=fixed)
    c = f.catchers.set_index("catcher")
    sd = float(np.sqrt(f.variances["catcher"]))
    return {"saved": -c["effect"], "n": c["n_pitches"], "sd": sd, "zero": sd < ZERO_SD, "converged": f.converged,
            "pitchers": None if f.pitchers is None else f.pitchers.set_index("pitcher")["effect"]}


def _fits(df, scores=SCORES, models_=(False, True)) -> dict:
    return {(s, fx): season_fit(df, s, fx) for s in scores for fx in models_}


# --- Data ---

_DATA = {}


def scores_table() -> pd.DataFrame:
    if "scores" not in _DATA:
        s = pd.read_parquet(SCORES_PARQUET)
        ab = pd.read_parquet(config.PITCHES_PARQUET, columns=["pa_id", "pitch_number", "at_bat_number"])
        _DATA["scores"] = s.merge(ab, on=["pa_id", "pitch_number"], how="left", validate="one_to_one")
    return _DATA["scores"]


def pitches_table() -> pd.DataFrame:
    if "pitches" not in _DATA:
        _DATA["pitches"] = pd.read_parquet(config.PITCHES_PARQUET)
    return _DATA["pitches"]


def pas_table() -> pd.DataFrame:
    if "pas" not in _DATA:
        _DATA["pas"] = pd.read_parquet(config.PAS_PARQUET)
    return _DATA["pas"]


def _save(name: str, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def _load(name: str):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# --- V0 ---

def stage_v0() -> dict:
    """Halves and mover groups from pitch counts and seeds only; no score column is read.

    Frozen once written: a rerun with the same rules leaves the existing files untouched, and a rerun with
    different rules is refused.
    """
    man = OUT / "manifest_v0.json"
    if man.exists() or (OUT / "v0.pkl").exists():
        _check_manifest()
        print("stage v0 already frozen with these rules; existing files kept", flush=True)
        return _load("v0")
    p = pitches_table()
    halves = arsenal.assign_folds(p, n_folds=2, seed=SEED).rename("half")
    games = p.drop_duplicates("game_pk")[["game_pk", "season", "game_date"]].set_index("game_pk")
    med = games.groupby("season")["game_date"].transform(lambda d: d.sort_values().iloc[(len(d) - 1) // 2])
    calendar = (games["game_date"] >= med).astype("int8").rename("calendar_half")
    counts = pd.read_parquet(SCORES_PARQUET, columns=["season", "catcher", "fielding_team", "game_pk"])  # no scores
    prim = {}
    for season in SEASONS:
        c = counts[counts["season"] == season].groupby(["catcher", "fielding_team"]).agg(
            pitches=("game_pk", "size"), games=("game_pk", "nunique")).reset_index()
        prim[season] = primary_teams(c)
    v0 = {"halves": halves, "calendar": calendar, "primary": prim, "movers": mover_groups(prim), "rules": rules()}
    _save("v0", v0)
    (OUT / "manifest_v0.json").write_text(json.dumps(rules(), indent=1, default=str))
    return v0


def _check_manifest():
    man = OUT / "manifest_v0.json"
    if not man.exists():
        raise RuntimeError("stage v0 has no manifest: run it first (or restore the frozen files)")
    if json.loads(man.read_text()) != json.loads(json.dumps(rules(), default=str)):
        raise ValueError("rules differ from the v0 manifest; the protocol is locked")


# --- §2b forward check ---

def forward_check(scorable_only: bool = False) -> pd.DataFrame:
    """§2b on all 2025 fit-eligible pitches (the protocol); `scorable_only` is the labeled supplementary check."""
    p = pitches_table()
    test = p[p["season"] == SEASONS[1]]
    train, test = p[p["season"] == SEASONS[0]], (test[test["scorable"]] if scorable_only else test)
    rows = []
    for kind in ("outcome", "contact"):
        if kind == "outcome":
            tr, obs = models.outcome_training(train), models.outcome_training(test)
            model, K = models.fit_outcome_model(train), models.N_OUTCOMES
            base = validation._league_baseline(tr, ["cidx", "action"], ["cidx"], K)
            keys, names = ["cidx", "action"], list(solver.OUTCOMES)
        else:
            tr, obs = models.contact_training(train), models.contact_training(test)
            model, K = models.fit_contact_model(train), models.N_BINS
            n = models._counts(tr, ["action"], K)
            base = (n + n.sum(axis=0) / n.to_numpy().sum()).div(n.sum(axis=1) + 1, axis=0)
            keys, names = ["action"], [f"bin_{b + 1}" for b in range(K)]
        pred, league = model.predict(obs), model.league_mean(obs)
        bp = models._lookup(base, obs, keys, fill=np.nan)
        overall = models._counts(tr.assign(_all=0), ["_all"], K).to_numpy()[0]
        bp = np.where(np.isnan(bp), overall / overall.sum(), bp)
        seen_pairs = set(map(tuple, tr[["pitcher", "action"]].drop_duplicates().to_numpy()))
        seen = np.array([(a, b) in seen_pairs for a, b in obs[["pitcher", "action"]].to_numpy()])
        y = obs["y"].to_numpy()
        for subset, m in (("all", np.ones(len(y), bool)), ("seen_cells", seen), ("unseen_cells", ~seen)):
            if m.sum() == 0:
                continue
            i = np.arange(m.sum())
            yy = y[m]
            row = {"model": kind, "subset": subset, "n": int(m.sum()), "share": float(m.mean()),
                   "log_loss": -np.log(pred[m][i, yy]).mean(), "baseline_log_loss": -np.log(bp[m][i, yy]).mean(),
                   "league_pooled_log_loss": -np.log(league[m][i, yy]).mean()}
            for j, nm in enumerate(names):
                row[f"cal_{nm}"] = validation.calibration_error(pred[m][:, j], yy == j)
            row["max_cal"] = max(row[f"cal_{nm}"] for nm in names)
            rows.append(row)
    return pd.DataFrame(rows)


# --- §3 within-season ---

def _half_eligible(df_half: pd.DataFrame, season_qual: set) -> tuple[set, pd.Series]:
    n = df_half.groupby("catcher").size()
    cand = set(n[n >= HALF_MIN].index) & season_qual
    return main_group(df_half, cand), n


def within_split(scores: pd.DataFrame, halves: pd.Series, label: str) -> dict:
    """§3a: the existing scores split by game half; per-half eligibility and connectivity."""
    out = {}
    for season in SEASONS:
        s = scores[scores["season"] == season]
        qual = set(s.groupby("catcher").size().loc[lambda n: n >= QUALIFY].index)
        h = s["game_pk"].map(halves)
        parts = {k: s[h == k] for k in (0, 1)}
        out[season] = _within_compare(parts, qual, label)
    return out


def _within_compare(parts: dict, qual: set, label: str) -> dict:
    elig = {k: _half_eligible(parts[k], qual) for k in (0, 1)}
    eligible = elig[0][0] & elig[1][0]
    dropped = []
    for c in sorted(qual):
        why = [f"half {k}: {'pitches' if elig[k][1].get(c, 0) < HALF_MIN else 'not in main group'}"
               for k in (0, 1) if c not in elig[k][0]]
        if why:
            dropped.append({"catcher": c, "reason": "; ".join(why),
                            "pitches_h0": int(elig[0][1].get(c, 0)), "pitches_h1": int(elig[1][1].get(c, 0))})
    fits = {k: _fits(parts[k]) for k in (0, 1)}
    res = {"label": label, "n_eligible": len(eligible), "dropped": pd.DataFrame(dropped), "fits_meta": {},
           "compare": [], "gap": [], "effects": {}}
    idx = sorted(eligible)
    for (score, fixed) in fits[0]:
        f0, f1 = fits[0][score, fixed], fits[1][score, fixed]
        model = "fixed" if fixed else "random"
        res["fits_meta"][score, model] = {k: {"sd": fits[k][score, fixed]["sd"],
                                              "converged": fits[k][score, fixed]["converged"]} for k in (0, 1)}
        x, y = f0["saved"].reindex(idx), f1["saved"].reindex(idx)
        res["effects"][score, model] = pd.DataFrame({"h0": x, "h1": y})
        allq = sorted(qual)  # every qualifying catcher, for the separate list of those not compared
        res["effects_all"] = res.get("effects_all", {})
        res["effects_all"][score, model] = pd.DataFrame({"h0": f0["saved"].reindex(allq),
                                                          "h1": f1["saved"].reindex(allq)})
        zero = f0["zero"] or f1["zero"]
        res["compare"].append({"score": score, "model": model, **compare(x, y, zero),
                               "sd_h0": f0["sd"], "sd_h1": f1["sd"], "converged": f0["converged"] and f1["converged"]})
        if model == "random" and score in ("D", "A"):
            for sel, ev, name in ((x, y, "choose h0, evaluate h1"), (y, x, "choose h1, evaluate h0")):
                if zero:
                    res["gap"].append({"score": score, "direction": name, "result": UNAVAILABLE})
                elif len(idx) < MIN_SET:
                    res["gap"].append({"score": score, "direction": name, "result": f"listed (n < {MIN_SET})"})
                else:
                    res["gap"].append({"score": score, "direction": name, **group_gap(sel, ev)})
    return res


def _half_pipeline(args) -> dict:
    """§3b: the whole pilot pipeline on one season-half's games only."""
    season, half, game_ids = args
    t0 = time.perf_counter()
    p, a = pitches_table(), pas_table()
    hp = p[(p["season"] == season) & p["game_pk"].isin(game_ids)].copy()
    folds = arsenal.assign_folds(hp, seed=SEED + 10 + half)
    hp["fold"] = hp["game_pk"].map(folds).astype("int8")
    ha = a[(a["season"] == season) & a["game_pk"].isin(game_ids)].copy()
    ha["fold"] = ha["game_pk"].map(folds).astype("int8")
    support = arsenal.support_table(hp)
    marked = arsenal.mark_arsenal(hp, arsenal.eligible_sets(support))
    run = scoring.run_pilot(marked, ha, support, seasons=(season,), with_re288=True)
    cand = marked["exclusion_reason"].isna()
    meta = {"season": season, "half": half, "games": len(game_ids), "candidates": int(cand.sum()),
            "scored": len(run.scores), "scored_share": len(run.scores) / int(cand.sum()),
            "out_of_arsenal_share": float(marked.loc[cand, "arsenal_reason"].notna().mean()),
            "alphas": {f: {"outcome_cell": r.outcome.alpha_cell, "contact_cell": r.contact.alpha_cell}
                       for (_, f), r in run.roles.items()}, "seconds": time.perf_counter() - t0}
    return {"scores": run.scores, "meta": meta}


def within_refit(workers: int, game_frac: float = 1.0, only=None) -> dict:
    halves = _load("v0")["halves"]
    jobs = []
    for season in SEASONS:
        for half in (0, 1):
            g = np.sort(halves.index[(halves == half).to_numpy()
                                     & halves.index.isin(pitches_table().loc[pitches_table()["season"] == season,
                                                                             "game_pk"].unique())])
            if game_frac < 1:
                g = g[: max(1, int(len(g) * game_frac))]
            if only is None or (season, half) in only:
                jobs.append((season, half, g))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        runs = {(r["meta"]["season"], r["meta"]["half"]): r for r in ex.map(_half_pipeline, jobs)}
    return runs


def within_refit_compare(runs: dict) -> dict:
    s = scores_table()
    out = {}
    for season in SEASONS:
        if (season, 0) not in runs or (season, 1) not in runs:
            continue
        qual = set(s[s["season"] == season].groupby("catcher").size().loc[lambda n: n >= QUALIFY].index)
        parts = {k: runs[season, k]["scores"] for k in (0, 1)}
        res = _within_compare(parts, qual, "independent refit")
        for (score, model), m in res["fits_meta"].items():
            for k in (0, 1):
                m[k]["main_group"] = len(_half_eligible(parts[k], qual)[0])
        res["meta"] = {k: runs[season, k]["meta"] for k in (0, 1)}
        out[season] = res
    return out


# --- §4 and §6 ---

def season_fits() -> dict:
    s = scores_table()
    return {season: _fits(s[s["season"] == season]) for season in SEASONS}


def season_sets() -> dict:
    s = scores_table()
    out = {}
    for season in SEASONS:
        d = s[s["season"] == season]
        n = d.groupby("catcher").size()
        qual = set(n[n >= QUALIFY].index)
        out[season] = {"qual": qual, "main": main_group(d, qual), "n": n}
    return out


def repeatability(fits: dict, sets: dict, movers: pd.DataFrame) -> dict:
    a, b = SEASONS
    both = sets[a]["qual"] & sets[b]["qual"]
    eligible = sorted(both & sets[a]["main"] & sets[b]["main"])
    disc = sorted(both - set(eligible))
    w = 2 * sets[a]["n"] * sets[b]["n"] / (sets[a]["n"] + sets[b]["n"])
    rows, listed, effects = [], [], {}
    for (score, fixed), fa in fits[a].items():
        fb = fits[b][score, fixed]
        model = "fixed" if fixed else "random"
        x, y = fa["saved"].reindex(eligible), fb["saved"].reindex(eligible)
        effects[score, model] = pd.DataFrame({"y0": x, "y1": y, "w": w.reindex(eligible)})
        zero = fa["zero"] or fb["zero"]
        row = {"score": score, "model": model, **compare(x, y, zero), "sd_0": fa["sd"], "sd_1": fb["sd"],
               "converged": fa["converged"] and fb["converged"]}
        row["weighted_pearson"] = (UNDEFINED if zero else weighted_pearson(
            x.to_numpy(), y.to_numpy(), w.reindex(eligible).to_numpy())) if len(eligible) >= MIN_SET else np.nan
        rows.append(row)
        for c in disc:
            listed.append({"score": score, "model": model, "catcher": c, "saved_0": fa["saved"].get(c, np.nan),
                           "saved_1": fb["saved"].get(c, np.nan),
                           "disconnected_in": ", ".join(str(s) for s in SEASONS if c not in sets[s]["main"])})
    by_group = []
    for grp in ("mover", "non_mover", "neither"):
        ids = sorted(set(movers.index[movers["group"] == grp]) & set(eligible))
        for score in ("D", "A"):
            fa, fb = fits[a][score, False], fits[b][score, False]
            x, y = fa["saved"].reindex(ids), fb["saved"].reindex(ids)
            by_group.append({"group": grp, "score": score, **compare(x, y, fa["zero"] or fb["zero"])})
    return {"eligible": eligible, "disconnected": pd.DataFrame(listed), "table": pd.DataFrame(rows),
            "by_group": pd.DataFrame(by_group), "effects": effects}


def team_staff(fits: dict, sets: dict, prim: dict) -> dict:
    s = scores_table()
    share_rows, staff_rows, pair_rows = [], [], []
    for season in SEASONS:
        ids = sorted(sets[season]["qual"] & sets[season]["main"])
        team = prim[season]["primary_team"].reindex(ids)
        d = s[s["season"] == season]
        pc = d.groupby(["catcher", "pitcher"]).size().rename("n").reset_index()
        pc = pc[pc["catcher"].isin(ids)]
        for score in ("D", "A"):
            f = fits[season][score, False]
            y = f["saved"].reindex(ids)
            share_rows.append({"season": season, "score": score, "catchers": len(ids), "teams": team.nunique(),
                               "catchers_per_team": len(ids) / team.nunique(), "share": team_share(y, team)})
            u = f["pitchers"]
            m = pc.assign(u=pc["pitcher"].map(u)).dropna()
            staff = (m["u"] * m["n"]).groupby(m["catcher"]).sum() / m["n"].groupby(m["catcher"]).sum()
            cost = -y
            staff_rows.append({"season": season, "score": score,
                               **compare(cost, staff.reindex(ids), f["zero"])})
            for t, g in team.groupby(team):
                if len(g) < 2:
                    continue
                cs = sorted(g.index)
                for i, c1 in enumerate(cs):
                    for c2 in cs[i + 1:]:
                        pair_rows.append({"season": season, "score": score, "team": t, "catcher_1": c1,
                                          "catcher_2": c2, "saved_1": y[c1], "saved_2": y[c2],
                                          "difference": y[c1] - y[c2] if not f["zero"] else np.nan})
    return {"share": pd.DataFrame(share_rows), "staff": pd.DataFrame(staff_rows), "pairs": pd.DataFrame(pair_rows)}


# --- §5b ---

def sensitivity(fits: dict, sets: dict) -> dict:
    s = scores_table()
    rows, excluded, listed = [], [], []
    for season in SEASONS:
        d = s[s["season"] == season]
        samples = {"all": d, "no_dugout": d[~d["dugout_called"].astype(bool)], "bases_empty": d[d["base_out"] < 3]}
        for name, (prim_score, check_score, sample) in CHECKS.items():
            sub = samples[sample]
            n_check = sub.groupby("catcher").size()
            cand = sets[season]["qual"] & set(n_check[n_check >= HALF_MIN].index)
            main_check = main_group(sub, cand)
            common = sorted(cand & sets[season]["main"] & main_check)
            for c in sorted(sets[season]["qual"] - set(common)):
                why = []
                if n_check.get(c, 0) < HALF_MIN:
                    why.append(f"{int(n_check.get(c, 0))} pitches in the check sample")
                if c not in sets[season]["main"]:
                    why.append("not in the primary main group")
                if c in cand and c not in main_check:
                    why.append("not in the check's main group")
                excluded.append({"season": season, "check": name, "catcher": c, "reason": "; ".join(why)})
            fp = fits[season][prim_score, False]
            fc = season_fit(sub, check_score) if not (sample == "all" and check_score == prim_score) else fp
            x, y = fp["saved"].reindex(common), fc["saved"].reindex(common)
            zero = fp["zero"] or fc["zero"]
            row = {"season": season, "check": name, "removed_pitches": len(d) - len(sub),
                   **compare(x, y, zero), "sd_primary": fp["sd"], "sd_check": fc["sd"],
                   "converged": fp["converged"] and fc["converged"]}
            if zero:
                row.update(max_abs_diff=np.nan, rank_moves=UNAVAILABLE)
            elif len(common) >= MIN_SET:
                a, b = x - x.mean(), y - y.mean()
                row["max_abs_diff"] = float((a - b).abs().max())
                row["rank_moves"] = int(((ordinal_ranks(x) - ordinal_ranks(y)).abs() >= RANK_MOVE).sum())
            rows.append(row)
    return {"table": pd.DataFrame(rows), "excluded": pd.DataFrame(excluded)}


# --- V3 examples ---

def catcher_examples(fits: dict, sets: dict) -> pd.DataFrame:
    s = scores_table()
    rows = []
    for season in SEASONS:
        ids = sorted(sets[season]["qual"] & sets[season]["main"])
        fd, fa, fb = (fits[season][k, False] for k in ("D", "A", "B"))
        if fd["zero"] or fa["zero"]:
            rows.append({"season": season, "group": "all", "note": UNAVAILABLE})
            continue
        D, A = fd["saved"].reindex(ids), fa["saved"].reindex(ids)
        rD, rA = ordinal_ranks(D), ordinal_ranks(A)
        diff = rD - rA
        order = sorted(ids, key=lambda c: (-diff[c], c))
        picks = [("D ranks lower than A (largest rank_D - rank_A)", order[:N_CATCHER_EX]),
                 ("D ranks higher than A (most negative)", sorted(ids, key=lambda c: (diff[c], c))[:N_CATCHER_EX])]
        third = len(ids) // 3
        top = [c for c in ids if rD[c] <= third and rA[c] <= third]
        bottom = [c for c in ids if rD[c] > len(ids) - third and rA[c] > len(ids) - third]
        for gi, (name, cand) in enumerate((("agree: top third on both", top), ("agree: bottom third on both", bottom))):
            picks.append((name, draw(np.random.default_rng([SEED, 7, season, gi]), cand, N_CATCHER_EX)))
            if len(cand) < N_CATCHER_EX:
                rows.append({"season": season, "group": name, "note": f"shortfall: {len(cand)} candidates"})
        d = s[s["season"] == season]
        for name, cs in picks:
            for c in cs:
                g = d[d["catcher"] == c]
                ahead = np.sign(g["strikes"].astype(int) - g["balls"].astype(int))
                by = g.groupby(ahead)[["D", "A", "B"]].mean().mul(-100)
                top_p = g["pitcher"].value_counts().head(3)
                rows.append({"season": season, "group": name, "catcher": c, "rank_D": int(rD[c]), "rank_A": int(rA[c]),
                             "D_saved": D[c], "A_saved": A[c], "B_saved": fb["saved"].get(c, np.nan),
                             "pitches": len(g), "top_pitchers": ", ".join(f"{p} ({n})" for p, n in top_p.items()),
                             **{f"{sc}_per100_{lab}": by[sc].get(k, np.nan) for sc in ("D", "A", "B")
                                for k, lab in ((1, "ahead"), (0, "even"), (-1, "behind"))}})
    return pd.DataFrame(rows)


def pitch_examples(sets: dict) -> pd.DataFrame:
    s = scores_table()
    season = PITCH_EX_SEASON
    ids = sets[season]["qual"] & sets[season]["main"]
    d = s[(s["season"] == season) & s["catcher"].isin(ids)]
    qD, qA = d["D"].quantile([0.05, 0.5, 0.95]).to_numpy(), d["A"].quantile([0.05, 0.5, 0.95]).to_numpy()
    cats = {"agree-costly": (d["D"] > qD[2]) & (d["A"] > qA[2]),
            "agree-good": (d["D"] < qD[0]) & (d["A"] < qA[0]),
            "disagree": ((d["D"] > qD[2]) & (d["A"] < qA[1])) | ((d["A"] > qA[2]) & (d["D"] < qD[1]))}
    key = ["game_pk", "at_bat_number", "pitch_number"]
    picked = []
    for ci, (cat, m) in enumerate(cats.items()):
        cand = d[m].sort_values(key)
        rng = np.random.default_rng([SEED, 8, ci])
        idx = np.arange(len(cand)) if len(cand) <= N_PITCH_EX else np.sort(rng.choice(len(cand), N_PITCH_EX, False))
        picked.append(cand.iloc[idx].assign(category=cat, candidates=len(cand)))
    ex = pd.concat(picked, ignore_index=True)
    # Q for every eligible action from the valuation role, re-solved (2024 pipeline, keep_solutions)
    run = scoring.run_pilot(pitches_table(), pas_table(), pd.read_parquet(config.DATA_DIR / "arsenal_support.parquet"),
                            seasons=(season,), keep_solutions=True)
    rows = []
    for _, r in ex.iterrows():
        sol = run.solutions[season, int(r["fold"])]
        units, val = sol["units"], sol["value"]
        u = int(units.table.set_index(["pitcher", "stand"]).loc[(r["pitcher"], r["stand"]), "u"])
        c = int(r["balls"]) * 3 + int(r["strikes"])
        bo = int(r["base_out"])
        acts = [a for a in units.actions[u] if a is not None]
        q = val.Q_ref[u, bo, c, :len(acts)]
        pi = val.policy.probs[u, c, :len(acts)]
        actual = acts.index(r["action"])
        d_re = q[actual] - q[acts.index(r["rec_action"])]
        rows.append({"category": r["category"], "candidates": r["candidates"], "game_pk": r["game_pk"],
                     "at_bat_number": r["at_bat_number"], "pitch_number": r["pitch_number"],
                     "pitcher": r["pitcher"], "catcher": r["catcher"], "stand": r["stand"],
                     "count": f"{int(r['balls'])}-{int(r['strikes'])}", "base_out": bo, "actual": r["action"],
                     "recommended": r["rec_action"],
                     "eligible (pi_ref, Q x100)": "; ".join(f"{a} ({p:.2f}, {100 * v:.2f})"
                                                           for a, p, v in zip(acts, pi, q)),
                     "V_x100": 100 * val.V_ref[u, bo, c], "D_x100": 100 * r["D"], "A_x100": 100 * r["A"],
                     "B_x100": 100 * r["B"], "D_recomputed_diff": abs(d_re - r["D"])})
    return pd.DataFrame(rows), {"D_q": qD, "A_q": qA, "counts": {k: int(v.sum()) for k, v in cats.items()}}


# --- Stages ---

def stage_v2(workers: int = 8):
    _check_manifest()
    v0 = _load("v0")
    t0 = time.perf_counter()
    res = {"forward": forward_check(), "forward_scorable": forward_check(scorable_only=True)}
    print(f"forward check done ({time.perf_counter() - t0:.0f} s)", flush=True)
    s = scores_table()
    res["split_random"] = within_split(s, v0["halves"], "split of existing scores (random halves)")
    res["split_calendar"] = within_split(s, v0["calendar"], "split of existing scores (calendar halves; secondary)")
    print(f"split halves done ({time.perf_counter() - t0:.0f} s)", flush=True)
    runs = within_refit(workers)
    _save("half_runs", runs)
    res["refit"] = within_refit_compare(runs)
    print(f"independent halves done ({time.perf_counter() - t0:.0f} s)", flush=True)
    fits, sets = season_fits(), season_sets()
    res["sets"] = {k: {"qual": len(v["qual"]), "main": len(v["main"])} for k, v in sets.items()}
    res["repeatability"] = repeatability(fits, sets, v0["movers"])
    res["sensitivity"] = sensitivity(fits, sets)
    res["team_staff"] = team_staff(fits, sets, v0["primary"])
    res["season_sd"] = pd.DataFrame([{"season": se, "score": sc, "model": "fixed" if fx else "random", "sd": f["sd"],
                                      "zero": f["zero"], "converged": f["converged"]}
                                     for se, ff in fits.items() for (sc, fx), f in ff.items()])
    _save("v2", res)
    print(f"stage v2 done ({time.perf_counter() - t0:.0f} s)", flush=True)


def stage_forward():
    """Recompute only §2b (protocol sample, plus the scorable-only supplementary check) into the stored v2 results."""
    _check_manifest()
    res = _load("v2")
    res["forward"], res["forward_scorable"] = forward_check(), forward_check(scorable_only=True)
    _save("v2", res)


def stage_v3():
    _check_manifest()
    fits, sets = season_fits(), season_sets()
    pex, pmeta = pitch_examples(sets)
    _save("v3", {"catchers": catcher_examples(fits, sets), "pitches": pex, "pitch_meta": pmeta})


# --- V4 report ---

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"


def _scatter_grid(panels: list, path, xlabel: str, ylabel: str, movers=None):
    """Small multiples of catcher effects (runs saved per 100 pitches); identity and zero lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 4.2 * rows), squeeze=False)
    for ax, (title, df) in zip(axes.flat, panels):
        x, y = df.iloc[:, 0], df.iloc[:, 1]
        lim = 1.1 * max(x.abs().max(), y.abs().max())
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="-", zorder=0)
        mv = df.index.isin(movers) if movers is not None else np.zeros(len(df), bool)
        size = 22 if "w" not in df else 10 + 40 * df["w"] / df["w"].max()
        ax.scatter(x[~mv], y[~mv], s=size if np.isscalar(size) else size[~mv], color=BLUE, edgecolor="white",
                   linewidth=1, label="other catchers", zorder=2)
        if mv.any():
            ax.scatter(x[mv], y[mv], s=size if np.isscalar(size) else size[mv], color=ORANGE, marker="^",
                       edgecolor="white", linewidth=1, label="between-season movers", zorder=3)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"{xlabel}: runs saved per 100 pitches", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{ylabel}: runs saved per 100 pitches", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def _half_connectivity(key: str, season: int, v0: dict) -> pd.DataFrame:
    """Per half: candidates (season-qualifying, >= 500 of the half's scorable pitches) and main-group size/share."""
    s = scores_table()
    d = s[s["season"] == season]
    qual = set(d.groupby("catcher").size().loc[lambda n: n >= QUALIFY].index)
    if key == "refit":
        runs = _load("half_runs")
        parts = {k: runs[season, k]["scores"] for k in (0, 1)}
    else:
        h = d["game_pk"].map(v0["halves" if key == "split_random" else "calendar"])
        parts = {k: d[h == k] for k in (0, 1)}
    rows = []
    for k, part in parts.items():
        n = part.groupby("catcher").size()
        cand = set(n[n >= HALF_MIN].index) & qual
        main = main_group(part, cand)
        rows.append({"half": k, "scorable_pitches": len(part), "candidates": len(cand), "main_group": len(main),
                     "main_group_share": len(main) / len(cand) if cand else np.nan})
    return pd.DataFrame(rows)


def _fmt(v):
    return v if isinstance(v, str) else f"{v:.3f}"


def _cmp_table(rows: list) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    for c in ("pearson", "spearman", "weighted_pearson"):
        if c in t:
            t[c] = t[c].map(_fmt)
    return t


def stage_v4():
    _check_manifest()
    v0, v2, v3 = _load("v0"), _load("v2"), _load("v3")
    lists = OUT / "lists"
    lists.mkdir(exist_ok=True)
    fig_dir = config.REPORTS_DIR
    L = ["# Pilot validation: prediction, repeatability, robustness and interpretation", "",
         "Protocol: `research_notes/validation/01_validation_protocol.md`, revision 3 (locked 2026-10-29). Runner: "
         "`python3.12 -m gamecalling.validation_pilot` (stages v0, v2, v3, v4); frozen rules in "
         "`data/validation_pilot/v1/manifest_v0.json`. Effects are shown as **runs saved vs. average per 100 pitches "
         "(higher = better)**; differences are taken after centering each fit on the compared catchers' mean. "
         "**Everything here is descriptive.** No confidence intervals are shown: ordinary intervals would treat "
         "catchers as independent and ignore the shared fitted models and pitcher networks. Catcher estimates are "
         "exploratory point estimates; there are no validated uncertainty intervals, significance claims, causal "
         "claims, upper bounds on skill or claims that skill is absent (plan, \"Claim limits\"). A repeatable score "
         "is not thereby shown to measure the intended skill.", "",
         "**Examined before the 2026-10-29 amendment** (reused, labeled): the within-season held-out prediction "
         "(§2a), the pitcher-fixed-effects and bridge-removal results (§5a), the season catcher SDs and leaderboard "
         "point estimates, all simulation results. **Examined for the first time here, under the locked rules:** "
         "everything else, including the 2024-2025 repeatability.", "",
         "All ranking fits converged; no fit has zero catcher variance (season SDs below).", "",
         _md(v2["season_sd"], "{:.4f}"), ""]

    f, fs = v2["forward"], v2.get("forward_scorable")
    keep = [c for c in ("model", "subset", "n", "share", "log_loss", "baseline_log_loss", "league_pooled_log_loss",
                        "max_cal") if c in f]
    L += ["## 2. Held-out prediction", "",
          "**(a) Reused (already examined), within season, cross-fitted:** outcome calibration error at most 0.0024 "
          "(2024) and 0.0023 (2025) per outcome against the 0.02 threshold; log loss 1.4799 vs. 1.4823 (league count "
          "x pitch type) and 1.4817 (the model's own league-pooled level) in 2024, 1.4756 vs. 1.4776 and 1.4772 in "
          "2025 (`reports/phase1_pilot.md`).", "",
          "**(b) Forward check (new; the protocol's sample):** models fit on all 2024 fit-eligible pitches, evaluated "
          "on **all 2025 fit-eligible pitches**. `seen_cells`: the pitcher threw that pitch type in the 2024 training "
          "data; `unseen_cells`: they did not (the model then relies on league levels). Calibration error = "
          "sample-weighted mean absolute gap over 10 equal-count bins, the largest over outcomes (contact bins).", "",
          _md(f[keep], "{:.4f}"), ""]
    if fs is not None:
        L += ["*Supplementary (labeled; not the protocol's sample):* the same check restricted to 2025 **scorable** "
              "pitches, as first run on 2026-10-29 before the sample was corrected. It leaves out "
              f"{int(f.iloc[0]['n'] - fs.iloc[0]['n']):,} outcome and "
              f"{int(f[f['model'] == 'contact'].iloc[0]['n'] - fs[fs['model'] == 'contact'].iloc[0]['n']):,} contact "
              "observations.", "", _md(fs[keep], "{:.4f}"), ""]
    L += [
          "Limitations: the gains are small (about 0.002-0.004 nats per pitch); this validates outcome "
          "probabilities, not decision values or catcher scores; calibration is per outcome, not joint; payoffs are "
          "validated only through the RE288 diagnostic.", ""]

    L += ["## 3. Within-season reliability", "",
          "Game halves: random (seeded, fixed) as primary; calendar halves (before / from the median game date) as a "
          "secondary split of the existing scores. A catcher is compared if, in each half, they have at least 500 of "
          "that half's scorable pitches and are in that half's main connected group (computed on that half's pitches). "
          "(a) splits the existing cross-fitted scores (shared fitted models, so optimistic); (b) reruns the whole "
          "pipeline inside each half (no shared fitted models; pitchers and the network are still shared). The group "
          "gap chooses the top and bottom thirds on one half and reports their mean on the other.", ""]
    panels = []
    for key, title in (("split_random", "(a) Split of existing scores, random halves"),
                       ("refit", "(b) Independent whole-pipeline refit in each half"),
                       ("split_calendar", "(a') Split of existing scores, calendar halves (secondary)")):
        L += [f"### {title}", ""]
        for season, r in v2[key].items():
            reasons = r["dropped"]["reason"].str.contains("main group").sum() if len(r["dropped"]) else 0
            L += [f"**{season}:** {r['n_eligible']} catchers compared; {len(r['dropped'])} qualifying catchers not "
                  f"compared ({reasons} of them outside a half's main group). Lists with both halves' scores: "
                  f"`data/validation_pilot/v1/lists/{key}_{season}.csv`.", ""]
            L += [_md(_cmp_table(r["compare"]), "{:.4f}"), ""]
            if r["gap"]:
                L += [_md(pd.DataFrame(r["gap"]), "{:.4f}"), ""]
            L += ["Per half (from the saved results): catchers qualifying in the season with at least 500 of the "
                  "half's scorable pitches, and how many of them form the half's main connected group:", "",
                  _md(_half_connectivity(key, season, v0), "{:.3f}"), ""]
            if "meta" in r:
                meta = pd.DataFrame([{k: m[k] for k in ("half", "games", "scored", "scored_share",
                                                        "out_of_arsenal_share")} for m in r["meta"].values()])
                L += [_md(meta, "{:.3f}"), "",
                      "Pooling strengths (pitcher-cell α) of each half's three role folds, against 103-127 (outcome) "
                      "and 114-139 (contact) in the full-season roles (`reports/phase1_pilot.md`):", "",
                      _md(pd.DataFrame([{"half": m["half"], "role_fold": rf, **a}
                                        for m in r["meta"].values() for rf, a in m["alphas"].items()]), "{:.1f}"), ""]
            ea = r["effects_all"]
            dl = pd.concat({f"{sc}_{mo}": d for (sc, mo), d in ea.items()}, axis=1)
            dl.columns = [f"{a}_{b}" for a, b in dl.columns]
            dl = dl.assign(compared=dl.index.isin(r["effects"]["D", "random"].index))
            dl = dl.join(r["dropped"].set_index("catcher")["reason"]) if len(r["dropped"]) else dl
            dl.to_csv(lists / f"{key}_{season}.csv")
            if r["n_eligible"] < MIN_SET:
                lst = r["effects"]["D", "random"].join(r["effects"]["A", "random"], lsuffix="_D", rsuffix="_A")
                L += [f"Fewer than {MIN_SET} catchers: listed instead of correlated (D and A random effects, half 0 / "
                      "half 1):", "", _md(lst.reset_index().rename(columns={"index": "catcher"}), "{:.4f}"), ""]
            if key != "split_calendar":
                for sc in ("D", "A"):
                    panels.append((f"{title.split(',')[0][:3]} {season} {sc}", r["effects"][sc, "random"]))
    L += ["Why the calendar split compares so few catchers: cross-team links in the catcher network come largely "
          "from pitchers who changed teams, and most of a traded pitcher's pitches for each team fall on opposite "
          "sides of the season's midpoint. Within each calendar half most catchers are therefore outside the main "
          "connected group, and the protocol lists them instead of correlating them.", ""]
    _scatter_grid([p for p in panels if p[0].startswith("(a)")], fig_dir / "validation_halves_split.png",
                  "half 0", "half 1")
    _scatter_grid([p for p in panels if p[0].startswith("(b)")], fig_dir / "validation_halves_refit.png",
                  "half 0", "half 1")
    L += ["Figures: `reports/validation_halves_split.png` (a), `reports/validation_halves_refit.png` (b); random "
          "effects, compared catchers.", ""]

    r = v2["repeatability"]
    a, b = SEASONS
    L += ["## 4. 2024-2025 repeatability", "",
          f"Catchers qualifying in both seasons and in the main connected group in both: **{len(r['eligible'])}**. "
          "Seasons fitted independently. Unweighted Pearson and Spearman (primary); a Pearson weighted by the "
          "harmonic mean of the two seasons' pitch counts (secondary, same set); median absolute change after "
          "centering.", "", _md(_cmp_table(r["table"].to_dict("records")), "{:.4f}"), "",
          "Catchers qualifying in both seasons but outside a main group (listed, not correlated):", "",
          _md(r["disconnected"], "{:.4f}"), ""]
    eff = r["effects"]
    _scatter_grid([(f"{sc} ({mo} effects)", eff[sc, mo]) for sc in ("D", "A") for mo in ("random",)],
                  fig_dir / "validation_repeatability.png", f"{a}", f"{b}",
                  movers=v0["movers"].index[v0["movers"]["group"] == "mover"])
    L += ["Figure: `reports/validation_repeatability.png` (random effects; point size by harmonic-mean pitches; "
          "triangles are between-season movers).", "",
          "**Reading (descriptive):** positive repeatability could come from stable team, staff or assignment "
          "patterns as well as skill; low repeatability does not show that skill is absent. Within-season and "
          "between-season repeatability measure different things (§3 vs. §4).", ""]

    sens = v2["sensitivity"]
    L += ["## 5. Robustness", "",
          "**(a) Reused (already examined):** D random vs. pitcher-fixed-effects correlation 0.95 / 0.94; largest "
          "identifiable bridge-removal change (D random / D fixed / A random / A fixed) 0.0095 / 0.0107 / 0.0068 / "
          "0.0114 (2024) and 0.0085 / 0.0117 / 0.0043 / 0.0091 (2025), medians about 0.001-0.003 per 100 pitches; "
          "main-group shares 100% and 97.7% (`reports/network_pilot.md`).", "",
          "**(b) Ranking-sample sensitivity (new).** R1 optimal continuation (D_opt), R2 one-pitch RE288 benchmark "
          "(D_1p); R3 without dugout-called games and R4 bases-empty pitches only change **only the ranking "
          "sample**: the outcome, contact and policy models and the payoffs still learned from the excluded "
          "observations. Common set: qualifying in the season, in the main group of both the primary and the "
          "check's sample, and at least 500 pitches in the check's sample; ranks recomputed within it. "
          "`removed_pitches`: pitches outside the check's sample. Units: runs saved per 100 pitches.", "",
          _md(_cmp_table(sens["table"].to_dict("records")), "{:.4f}"), "",
          f"Catchers excluded from a common set: `data/validation_pilot/v1/lists/sensitivity_excluded.csv` "
          f"({len(sens['excluded'])} rows).", ""]
    sens["excluded"].to_csv(lists / "sensitivity_excluded.csv", index=False)

    ts = v2["team_staff"]
    mv = v0["movers"]["group"].value_counts()
    neither = sorted(set(v0["movers"].index[v0["movers"]["group"] == "neither"]) & set(r["eligible"]))
    nl = pd.DataFrame({"catcher": neither, f"D_{a}": eff["D", "random"]["y0"].reindex(neither).to_numpy(),
                       f"D_{b}": eff["D", "random"]["y1"].reindex(neither).to_numpy(),
                       f"A_{a}": eff["A", "random"]["y0"].reindex(neither).to_numpy(),
                       f"A_{b}": eff["A", "random"]["y1"].reindex(neither).to_numpy()})
    share = ts["share"].copy()
    share["context_(T-1)/(N-1)"] = (share["teams"] - 1) / (share["catchers"] - 1)
    within = {se: int((t["within_mover"] & (t["pitches"] >= QUALIFY)).sum()) for se, t in v0["primary"].items()}
    L += ["## 6. Team and staff dependence", "",
          f"Groups (defined from pitch counts in stage v0, before scores were joined), among catchers qualifying in "
          f"both seasons: {mv.get('mover', 0)} between-season movers, {mv.get('non_mover', 0)} non-movers, "
          f"{mv.get('neither', 0)} neither. Within-season movers (qualifying, >= 300 pitches with each of 2+ teams): "
          + ", ".join(f"{se}: {n}" for se, n in within.items()) + ". These are mover comparisons, not natural "
          "experiments.", "",
          "**Repeatability by group** (§4's eligible set; random effects):", "",
          _md(_cmp_table(r["by_group"].to_dict("records")), "{:.4f}"), "",
          "Group with fewer than 10 catchers, listed:", "", _md(nl, "{:.4f}"), "",
          "**Between-team share of fitted-score variation** (qualifying main-group catchers grouped by primary team; "
          "sum_t n_t (ybar_t - ybar)^2 / sum_c (y_c - ybar)^2). This is the descriptive share of fitted-score "
          "variation between primary-team groups, **not** the fraction caused by teams. For context only (not a "
          "test): with catchers assigned to teams at random and these group sizes, the expected share is about "
          "(T-1)/(N-1).", "", _md(share, "{:.3f}"), "",
          "**Staff diagnostic:** correlation of each catcher's effect (cost scale) with the pitch-weighted mean of "
          "their pitchers' pitcher-season effects from the same fit (cost scale). A diagnostic of how that fitted model "
          "divides variation between catchers and pitchers, not independent evidence about staff quality.", "",
          _md(_cmp_table(ts["staff"].to_dict("records")), "{:.4f}"), "",
          f"**Same-team pairs** (same team; pitcher assignments may differ): {len(ts['pairs'])} rows in "
          "`data/validation_pilot/v1/lists/same_team_pairs.csv`.", ""]
    ts["pairs"].to_csv(lists / "same_team_pairs.csv", index=False)

    ce, pe, pm = v3["catchers"], v3["pitches"], v3["pitch_meta"]
    cols = [c for c in ("season", "group", "catcher", "rank_D", "rank_A", "D_saved", "A_saved", "B_saved", "pitches",
                        "top_pitchers") if c in ce]
    per = [c for c in ce if "_per100_" in c]
    L += ["## 7. Baseball interpretation", "",
          "Selected by the rules frozen in stage v0 (seeds and tie-breaks in the manifest); **every selected case is "
          "reported.** Ranks: 1 = most runs saved, among qualifying main-group catchers.", "",
          _md(ce[cols], "{:.4f}"), "",
          "Mean per pitch by count group (runs saved per 100 pitches; ahead = more strikes than balls):", "",
          _md(ce[["season", "catcher"] + per], "{:.3f}"), "",
          f"**Pitch level ({PITCH_EX_SEASON}).** Percentiles over qualifying main-group catchers' scored pitches "
          f"(runs per pitch): D 5th/50th/95th {pm['D_q'][0]:.4f} / {pm['D_q'][1]:.4f} / {pm['D_q'][2]:.4f}; A "
          f"{pm['A_q'][0]:.4f} / {pm['A_q'][1]:.4f} / {pm['A_q'][2]:.4f}. Candidates per category: "
          + ", ".join(f"{k} {v:,}" for k, v in pm["counts"].items()) + ". Costs x 100 (runs for the batting team, "
          "lower = better for the pitcher). Q for every eligible action re-solved from the valuation role; it "
          f"reproduces the stored D exactly (largest difference {pe['D_recomputed_diff'].max():.1e}). The recommended "
          "pitch comes from the independent selection role, so it need not have the lowest valuation Q.", "",
          _md(pe.drop(columns=["D_recomputed_diff"]), "{:.3f}"), ""]

    rr = r["table"].set_index(["score", "model"])
    sp = {se: pd.DataFrame(v2["split_random"][se]["compare"]).set_index(["score", "model"]) for se in SEASONS}
    rf = {se: pd.DataFrame(v2["refit"][se]["compare"]).set_index(["score", "model"]) for se in SEASONS}
    st = sens["table"].set_index(["season", "check"])
    g = lambda d, sc: " / ".join(_fmt(d[se].loc[(sc, "random"), "pearson"]) for se in SEASONS)  # noqa: E731
    ev = [
        ("Construct", "Difference from an independently selected recommendation", "Difference from the pooled "
         "reference policy's value", "plan", "yes"),
        ("Recovery simulation (corr. with targets at SD 0.01 / 0.02 / 0.05 / 0.08)", "0.30-0.35 / 0.50-0.52 / "
         "0.80-0.83 / 0.90-0.92; spread under identical calling 0.0075; offsets at zero effect sensitive to game "
         "assignment", "0.56-0.58 / 0.76-0.81 / 0.92-0.93 / 0.95; spread under identical calling 0.0008",
         "reports/phase23_pilot.md; spec 04 §5", "yes"),
        ("Uncertainty procedures", "none validated", "none validated (first-copy A under-covered)", "uncertainty_design",
         "yes"),
        ("Held-out prediction", "shared models", "shared models", "§2", "within season yes; forward no"),
        ("Within-season reliability, split of existing scores (Pearson 2024 / 2025)", g(sp, "D"), g(sp, "A"), "§3a", "no"),
        ("Within-season reliability, independent refit (Pearson 2024 / 2025)", g(rf, "D"), g(rf, "A"), "§3b", "no"),
        ("2024-2025 repeatability (Pearson, random / fixed)",
         f"{_fmt(rr.loc[('D', 'random'), 'pearson'])} / {_fmt(rr.loc[('D', 'fixed'), 'pearson'])}",
         f"{_fmt(rr.loc[('A', 'random'), 'pearson'])} / {_fmt(rr.loc[('A', 'fixed'), 'pearson'])}", "§4", "no"),
        ("Robustness: pitcher fixed effects; bridges", "0.95 / 0.94; max change 0.0095 / 0.0085",
         "max change 0.0068 / 0.0043", "§5a", "yes"),
        ("Robustness: bases-empty ranking sample (Pearson 2024 / 2025)",
         f"{_fmt(st.loc[(a, 'R4_D'), 'pearson'])} / {_fmt(st.loc[(b, 'R4_D'), 'pearson'])}",
         f"{_fmt(st.loc[(a, 'R4_A'), 'pearson'])} / {_fmt(st.loc[(b, 'R4_A'), 'pearson'])}", "§5b", "no"),
        ("Team share (2024 / 2025; context about 0.35)",
         " / ".join(_fmt(x) for x in share[share["score"] == "D"]["share"]),
         " / ".join(_fmt(x) for x in share[share["score"] == "A"]["share"]), "§6", "no"),
        ("Interpretation", "see §7", "see §7", "§7", "no"),
    ]
    L += ["## 8. D-vs.-A evidence table", "",
          "The addendum's rule: D stays primary unless the table as a whole supports a documented revision; no single "
          "row decides, and reliability or persistence alone never selects a score. Outcome options: \"no revision "
          "justified\", \"revision justified\" (documented as post hoc, D still reported), \"inconclusive\". "
          "**Outcome (decided 2026-10-29, from the review the user pasted): inconclusive.** D remains the declared "
          "primary score and A is reported prominently. A has stronger recovery and within-season evidence, but it "
          "answers a different comparison question, and its year-to-year correlation changes from +0.11 to -0.19 with "
          "pitcher fixed effects; that does not justify presenting A as a validated replacement.", "",
          _md(pd.DataFrame(ev, columns=["evidence", "D", "A", "source", "examined before 2026-10-29"])), ""]
    REPORT.write_text("\n".join(L))
    print(f"report in {REPORT}", flush=True)


def smoke(workers: int = 2):
    """V1: §3b on a small game subset (one season-half), checking the pipeline path only."""
    runs = within_refit(workers, game_frac=0.1, only=[(SEASONS[0], 0)])
    m = runs[SEASONS[0], 0]["meta"]
    print({k: v for k, v in m.items() if k != "alphas"}, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    stage, workers = (args[0] if args else "v0"), (int(args[1]) if len(args) > 1 else 8)
    OUT.mkdir(parents=True, exist_ok=True)
    {"v0": lambda: stage_v0(), "smoke": lambda: smoke(workers), "v2": lambda: stage_v2(workers),
     "v3": stage_v3, "v4": stage_v4, "forward": stage_forward}[stage]()
