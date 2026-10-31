"""Per-identity solves and scoring for the batter models M3a and M4 (research program S6 steps 2-4).

Specification: research_notes/research_program/S6_solver_addendum.md (revision 2).

Within one fitted run, a solve is identified by (season, scoring fold, role, pitcher, batter hand, batter, base-out
state). Those fields determine every input: the probabilities (pitcher, hand, action, count, the batter's features in
that role), the payoffs and RE288 (role, base-out state), pi (unit, count, role) and the eligible set (scoring fold,
unit). Only observed identities are solved. Reuse is scoped to one fit_id: nothing is shared across versions, run
labels or fits, and solutions do not outlive the pipeline call.

The solver's stopping tolerance is not a bound on value error: it compares successive iterates, so a batch's
composition can change the result by more than the tolerance when the two-strike foul loop contracts slowly. 1e-8 is
a required empirical comparison (compare_values), and a failure stops the work. The stopping rule is unchanged.

Scores are scoring.score_fold's definitions, indexed by identity, with P0's action order and first-minimum tie-break.
Pi is P0's and batter-independent, so the reference continuation does not adapt to the batter.
"""

import hashlib
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, scoring, solver

CHUNK = 4000
COMPARE_TOL = 1e-8
IDENTITY_KEYS = ["pitcher", "stand", "batter", "base_out"]
VALUE_ARRAYS = ("V_ref", "Q_ref", "V_opt", "Q_opt", "Q_1p")
PITCH_KEYS = ["season", "pa_id", "pitch_number"]
ROLE_NAMES = ("select", "value")


class NumericalDiscrepancy(RuntimeError):
    """Two computations of the same values differ by more than the comparison tolerance."""


def compare_values(a: np.ndarray, b: np.ndarray, tol: float = COMPARE_TOL, what: str = "values") -> float:
    """Largest absolute difference; infinite entries must coincide. Raises above tol: never widened automatically."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isinf(a), np.isinf(b)):
        raise NumericalDiscrepancy(f"{what}: shapes or ineligible entries differ")
    finite = np.isfinite(a)
    diff = float(np.abs(a[finite] - b[finite]).max(initial=0.0))
    if not np.isfinite(b[finite]).all() or diff > tol:
        raise NumericalDiscrepancy(f"{what}: max difference {diff:.3e} exceeds {tol:.1e}")
    return diff


def peak_rss_gb() -> float:
    """This process's peak resident set (bytes on macOS, kilobytes on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


# --- Fitted-run scope ---

def _hash_model(h, model) -> None:
    if hasattr(model, "theta"):
        f = model.features
        for a in (model.theta, f.mean, f.sd, f.totals):
            h.update(np.asarray(a, float).tobytes())
        model = model.p0
    h.update(np.array([model.alpha_league, model.alpha_cell]).tobytes())
    h.update(model.cell_n.to_numpy().tobytes())


def make_fit_id(model_id: str, run_label: str, season: int, fold: int, role) -> str:
    """Identifies one fitted role: version, run label (as in artifact paths), season, fold, and a hash of the fit."""
    h = hashlib.sha256()
    _hash_model(h, role.outcome)
    _hash_model(h, role.contact)
    h.update(role.payoffs.contact.tobytes())
    h.update(np.asarray(role.re288, float).tobytes())
    if role.policy_train is not None:
        h.update(pd.util.hash_pandas_object(role.policy_train, index=False).to_numpy().tobytes())
    return f"{model_id}/{run_label}/{season}/{fold}/{h.hexdigest()[:16]}"


def _scope(fit_id: str) -> tuple:
    return tuple(fit_id.split("/")[:3])


# --- Identities ---

def scored_pitches(pitches: pd.DataFrame, season: int, k: int, units: models.Units) -> pd.DataFrame:
    """A scoring fold's scorable pitches with their unit, action slot, count and base-out state."""
    p = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
    p = p.merge(units.slots, on=["pitcher", "stand", "action"], how="left", validate="many_to_one")
    if p["u"].isna().any():
        raise ValueError("scorable pitch outside the eligible set")
    return p.assign(u=p["u"].astype(int), slot=p["slot"].astype(int),
                    cidx=(p["balls"] * 3 + p["strikes"]).astype(int), base_out=p["base_out"].astype(int))


def identity_table(p: pd.DataFrame, units: models.Units) -> tuple[pd.DataFrame, np.ndarray]:
    """The observed solve identities, in sorted order (the same in both roles), and each pitch's identity."""
    ids = p[IDENTITY_KEYS].drop_duplicates().sort_values(IDENTITY_KEYS).reset_index(drop=True)
    ids = ids.merge(units.table, on=["pitcher", "stand"], how="left", validate="many_to_one")
    ids["A"] = units.eligible[ids["u"].to_numpy()].sum(axis=1)
    ids["id"] = np.arange(len(ids))
    pitch_id = p[IDENTITY_KEYS].merge(ids[IDENTITY_KEYS + ["id"]], on=IDENTITY_KEYS, how="left")["id"].to_numpy()
    return ids, pitch_id


# --- Solves ---

def context_probabilities(role, ctx: pd.DataFrame, units: models.Units, A: int) -> tuple[np.ndarray, np.ndarray]:
    """Outcome (N, 12, A, 6) and contact (N, A, 8) probabilities of contexts (u, batter) with arsenal size A."""
    N = len(ctx)
    slots = units.slots[units.slots["slot"] < A]
    base = ctx[["u", "batter"]].assign(pos=np.arange(N)).merge(slots, on="u")
    pc = np.zeros((N, A, models.N_BINS))
    pc[base["pos"], base["slot"]] = role.contact.predict(base)
    q = base.merge(pd.DataFrame({"cidx": np.arange(solver.N_COUNTS)}), how="cross")
    p = np.zeros((N, solver.N_COUNTS, A, models.N_OUTCOMES))
    p[q["pos"], q["cidx"], q["slot"]] = role.outcome.predict(q)
    return p, pc


def solve_chunk(role, policy: np.ndarray, p: np.ndarray, pc: np.ndarray, b: np.ndarray, stats: dict | None = None) -> dict:
    """All value types for N identities: p (N, 12, A, 6), pc (N, A, 8), policy (N, 12, A), b (N,). Q_ref continues with
    V_ref, Q_opt with V_opt and Q_1p with the role's RE288; terminal values come from each identity's base-out state."""
    pay = role.payoffs
    bip = np.einsum("iak,ik->ia", pc, pay.contact[b])[:, None, :]
    walk, strikeout = pay.walk[b], pay.strikeout[b]
    eligible = np.ones((len(b), 1, p.shape[2]), bool)
    V_ref, Q_ref = solver.evaluate(p, bip, walk, strikeout, policy, eligible=eligible, stats=stats)
    V_opt, Q_opt = solver.optimize(p, bip, walk, strikeout, eligible, stats=stats)
    Q_1p = solver.action_values(np.asarray(role.re288)[b], p, bip, walk, strikeout)
    return {"V_ref": V_ref, "Q_ref": Q_ref, "V_opt": V_opt, "Q_opt": Q_opt, "Q_1p": Q_1p}


@dataclass
class PitchValues:
    """One role's values at a scoring fold's pitches. Q arrays are (n, A_max), inf outside the eligible set."""
    V_ref: np.ndarray
    Q_ref: np.ndarray
    V_opt: np.ndarray
    Q_opt: np.ndarray
    Q_1p: np.ndarray
    fit_id: str
    log: dict = field(default_factory=dict)


def solve_role_values(role, units: models.Units, policy: models.Policy, ids: pd.DataFrame, pitch_id: np.ndarray,
                      cidx: np.ndarray, fit_id: str = "", chunk: int = CHUNK) -> PitchValues:
    """Solve every identity once, in sorted fixed-size batches by arsenal size; keep only the scored states."""
    n, A_max = len(pitch_id), units.eligible.shape[1]
    out = {"V_ref": np.full(n, np.nan), "V_opt": np.full(n, np.nan), "Q_ref": np.full((n, A_max), np.inf),
           "Q_opt": np.full((n, A_max), np.inf), "Q_1p": np.full((n, A_max), np.inf)}
    t_prob = t_solve = 0.0
    batches, stats, id_pos = 0, {"iterations": 0, "max_iterations": 0}, np.full(len(ids), -1)
    for A, group in ids.groupby("A", sort=True):
        t = time.perf_counter()
        ctx = group[["u", "batter"]].drop_duplicates().reset_index(drop=True)
        p_all, pc_all = context_probabilities(role, ctx, units, int(A))
        ctx_of = group[["u", "batter"]].merge(ctx.assign(pos=np.arange(len(ctx))), on=["u", "batter"])["pos"].to_numpy()
        t_prob += time.perf_counter() - t
        t = time.perf_counter()
        group_ids, group_b, group_u = group["id"].to_numpy(), group["base_out"].to_numpy(), group["u"].to_numpy()
        for start in range(0, len(group), chunk):
            sel = np.arange(start, min(start + chunk, len(group)))
            pos = ctx_of[sel]
            values = solve_chunk(role, policy.probs[group_u[sel]][:, :, :int(A)], p_all[pos], pc_all[pos], group_b[sel],
                                 stats)
            batches += 1
            id_pos[:] = -1
            id_pos[group_ids[sel]] = np.arange(len(sel))
            where = id_pos[pitch_id]
            rows = np.flatnonzero(where >= 0)
            at, c = where[rows], cidx[rows]
            out["V_ref"][rows] = values["V_ref"][at, c]
            out["V_opt"][rows] = values["V_opt"][at, c]
            for name in ("Q_ref", "Q_opt", "Q_1p"):
                out[name][rows[:, None], np.arange(int(A))[None, :]] = values[name][at, c]
        t_solve += time.perf_counter() - t
    log = {"identities": len(ids), "contexts": len(ids[["u", "batter"]].drop_duplicates()), "batches": batches,
           "iterations": stats["iterations"], "max_iterations": stats["max_iterations"], "seconds_probabilities": t_prob,
           "seconds_solve": t_solve, "fit_id": fit_id}
    return PitchValues(fit_id=fit_id, log=log, **out)


# --- Scoring ---

def score_values(p: pd.DataFrame, sel: PitchValues, val: PitchValues, units: models.Units,
                 policy_value: models.Policy) -> pd.DataFrame:
    """scoring.score_fold's columns from per-pitch values. Both roles must belong to one version, run and season."""
    if _scope(sel.fit_id) != _scope(val.fit_id):
        raise ValueError(f"selection and valuation come from different runs: {sel.fit_id} / {val.fit_id}")
    u, a, c = p["u"].to_numpy(), p["slot"].to_numpy(), p["cidx"].to_numpy()
    rows = np.arange(len(p))
    out = p[scoring.SCORE_COLUMNS + [col for col in ("w", "w_hyper") if col in p]].copy()

    def scored(q_sel, q_val):
        rec = q_sel.argmin(axis=1)     # P0's action order and first-minimum tie-break
        return q_val[rows, a] - q_val[rows, rec], rec

    out["D"], rec = scored(sel.Q_ref, val.Q_ref)
    out["Q_actual"] = val.Q_ref[rows, a]
    out["V_value"] = val.V_ref
    out["A"] = out["Q_actual"] - out["V_value"]
    out["B"] = out["D"] - out["A"]
    out["rec_action"] = units.actions[u, rec]
    out["agree"] = rec == a
    out["pi_actual"] = policy_value.probs[u, c, a]
    out["D_opt"], rec_opt = scored(sel.Q_opt, val.Q_opt)
    out["agree_opt"] = rec_opt == a
    out["D_1p"], rec_1p = scored(sel.Q_1p, val.Q_1p)
    out["agree_1p"] = rec_1p == a
    out["n_eligible"] = units.eligible[u].sum(axis=1)
    return out


def action_value_rows(p: pd.DataFrame, q_select: np.ndarray, q_value: np.ndarray, units: models.Units,
                      k: int) -> pd.DataFrame:
    """Long form: one row per scored pitch and eligible action, in pitch and slot order."""
    pitch, slot = np.nonzero(units.eligible[p["u"].to_numpy()])
    out = p.iloc[pitch][PITCH_KEYS].reset_index(drop=True)
    return out.assign(scoring_fold=k, action=units.actions[p["u"].to_numpy()[pitch], slot],
                      Q_select=q_select[pitch, slot], Q_value=q_value[pitch, slot])


def write_action_values(rows: pd.DataFrame, root: Path, model_id: str, run_label: str, season: int, k: int) -> dict:
    """Write one fold's file, refusing to overwrite; returns the artifact reference."""
    path = Path(root) / model_id / "runs" / run_label / "action_values" / f"season={season}" / f"scoring_fold={k}.parquet"
    if path.exists():
        raise FileExistsError(f"{path} exists: use a new run label for a deliberate rerun")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    return {"path": str(path), "model": model_id, "run_label": run_label, "season": int(season), "scoring_fold": int(k),
            "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# --- The per-action reader and decision relevance (addendum §5.3) ---

def read_action_values(refs: list) -> pd.DataFrame:
    """One version's rows. Duplicate (pitch, action) keys raise, as does a file that does not match its reference."""
    parts = []
    for r in refs:
        f = pd.read_parquet(r["path"])
        if len(f) != r["rows"] or (f["scoring_fold"] != r["scoring_fold"]).any() or (f["season"] != r["season"]).any():
            raise ValueError(f"{r['path']} does not match its reference")
        parts.append(f)
    rows = pd.concat(parts, ignore_index=True)
    if rows.duplicated(PITCH_KEYS + ["action"]).any():
        raise ValueError("duplicate pitch and action keys in the per-action values")
    return rows


def decision_relevance_rows(base_refs: list, candidate_refs: list) -> pd.DataFrame:
    """Share of changed recommendations and the pitch-weighted SD of the change in Q_value(a) - Q_value(b), by season
    (S4's accepted convention). Refuses mixed run labels, mismatched folds and differing eligible-action sets: it never
    silently intersects."""
    from gamecalling.research_s4 import pair_change_sd
    labels = {r["run_label"] for r in base_refs} | {r["run_label"] for r in candidate_refs}
    if len(labels) != 1:
        raise ValueError(f"per-action comparisons need one run label, got {sorted(labels)}")
    keys = PITCH_KEYS + ["action"]
    m = read_action_values(base_refs).merge(read_action_values(candidate_refs), on=keys, how="outer",
                                            suffixes=("_b", "_c"), indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError("the versions differ in their eligible-action sets or scored pitches")
    if (m["scoring_fold_b"] != m["scoring_fold_c"]).any():
        raise ValueError("a pitch is in different scoring folds in the two versions")
    m = m.sort_values(keys, kind="stable")
    out = []
    for season, g in m.groupby("season"):
        codes = g.groupby(PITCH_KEYS[1:], sort=False).ngroup().to_numpy()
        slots = g.groupby(PITCH_KEYS[1:], sort=False).cumcount().to_numpy()
        n, A = codes.max() + 1, slots.max() + 1
        arrays = {}
        for col in ("Q_select_b", "Q_select_c", "Q_value_b", "Q_value_c"):
            arr = np.full((n, A), np.inf)
            arr[codes, slots] = g[col].to_numpy()
            arrays[col] = arr
        eligible = np.isfinite(arrays["Q_value_b"])
        changed = arrays["Q_select_b"].argmin(1) != arrays["Q_select_c"].argmin(1)
        ch, scale = pair_change_sd(arrays["Q_value_b"], arrays["Q_value_c"], eligible)
        out.append({"season": season, "pitches": int(n), "share_rec_changed": float(changed.mean()),
                    "sd_pair_change": float(np.sqrt(np.nanmean(ch))), "sd_pair_base": float(np.sqrt(np.nanmean(scale)))})
    return pd.DataFrame(out).assign(ratio=lambda d: d["sd_pair_change"] / d["sd_pair_base"])


# --- Jobs and the pipeline ---

def fit_job(args) -> dict:
    """One role's fit in a worker: only that fold's rows are passed in."""
    from gamecalling.research_s6 import verify_threads
    fit_role, rows, pas, season, fold, model_id, run_label = args
    threads = verify_threads()
    t = time.perf_counter()
    role = fit_role(rows, pas, season, fold)
    seconds = time.perf_counter() - t
    return {"season": season, "fold": fold, "role": role, "fit_id": make_fit_id(model_id, run_label, season, fold, role),
            "seconds": seconds, "peak_rss_gb": peak_rss_gb(), **threads}


def score_fold_values(rows, season, k, sel_role, val_role, sel_id, val_id, fit_policy, units):
    """Both roles' solves for a scoring fold: the pitches, their values, and the value role's policy."""
    p = scored_pitches(rows, season, k, units)
    ids, pitch_id = identity_table(p, units)
    cidx = p["cidx"].to_numpy()
    policies = {"select": fit_policy(sel_role.policy_train, units), "value": fit_policy(val_role.policy_train, units)}
    sel = solve_role_values(sel_role, units, policies["select"], ids, pitch_id, cidx, sel_id)
    val = solve_role_values(val_role, units, policies["value"], ids, pitch_id, cidx, val_id)
    return p, ids, pitch_id, sel, val, policies


def score_job(args) -> dict:
    """One scoring fold in a worker: both roles' solves, scoring and the per-action file."""
    from gamecalling.research_s6 import verify_threads
    rows, season, k, sel_role, val_role, sel_id, val_id, fit_policy, units, root, model_id, run_label = args
    threads = verify_threads()
    t0 = time.perf_counter()
    p, ids, pitch_id, sel, val, policies = score_fold_values(rows, season, k, sel_role, val_role, sel_id, val_id,
                                                             fit_policy, units)
    t = time.perf_counter()
    scores = score_values(p, sel, val, units, policies["value"])
    t_score = time.perf_counter() - t
    t = time.perf_counter()
    ref = write_action_values(action_value_rows(p, sel.Q_ref, val.Q_ref, units, k), root, model_id, run_label, season, k)
    t_write = time.perf_counter() - t
    log = [{"season": season, "scoring_fold": k, "role": nm, **v.log} for nm, v in (("select", sel), ("value", val))]
    return {"season": season, "k": k, "scores": scores, "ref": ref, "solve_log": log, "seconds_score": t_score,
            "seconds_write": t_write, "seconds": time.perf_counter() - t0, "peak_rss_gb": peak_rss_gb(),
            "policy_alphas": [{"season": season, "scoring_fold": k, "role": nm, "alpha": pol.alpha}
                              for nm, pol in policies.items()], **threads}


def batter_pipeline(pitches, pas, support, spec, seasons, fit_role, run_label="full", root=None, workers=1,
                    keep_solutions=False) -> scoring.PipelineRun:
    """Fit all roles, then score each fold (research.Spec.pipeline). Reuse lives inside this call; `workers` > 1 runs
    the role fits and the scoring-fold jobs in a process pool, passing each job only its own rows."""
    t0 = time.perf_counter()
    eligible = arsenal.eligible_sets(support)
    fit_args = []
    for season in seasons:
        for f in range(config.N_FOLDS):
            fit_args.append((fit_role, pitches[(pitches["season"] == season) & (pitches["fold"] == f)],
                             pas[(pas["season"] == season) & (pas["fold"] == f)], season, f, spec.id, run_label))
    ex = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    mapper = ex.map if ex is not None else map
    try:
        fits = {(r["season"], r["fold"]): r for r in mapper(fit_job, fit_args)}
        t_fit = time.perf_counter() - t0
        score_args = []
        for season in seasons:
            for k in range(config.N_FOLDS):
                s, v = arsenal.fold_roles(k)
                units = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == k)])
                rows = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
                score_args.append((rows, season, k, fits[season, s]["role"], fits[season, v]["role"],
                                   fits[season, s]["fit_id"], fits[season, v]["fit_id"], spec.fit_policy, units, root,
                                   spec.id, run_label))
        results = list(mapper(score_job, score_args))
    finally:
        if ex is not None:
            ex.shutdown()
    return scoring.PipelineRun(
        scores=pd.concat([r["scores"] for r in results], ignore_index=True),
        roles={key: r["role"] for key, r in fits.items()},
        policy_alphas=[row for r in results for row in r["policy_alphas"]],
        action_values=[r["ref"] for r in results],
        solve_log=[row for r in results for row in r["solve_log"]],
        abnormal_log=[e for r in fits.values() for m in (r["role"].outcome, r["role"].contact)
                      for e in getattr(m, "abnormal", [])],
        fit_ids={f"{key[0]}/{key[1]}": r["fit_id"] for key, r in fits.items()},
        timings={"wall_fit": t_fit, "wall_total": time.perf_counter() - t0,
                 "fit_jobs": [r["seconds"] for r in fits.values()], "score_jobs": [r["seconds"] for r in results],
                 "peak_rss_gb": max(r["peak_rss_gb"] for r in [*fits.values(), *results])})


# --- Validation helpers (addendum §6, §8) ---

def values_from_solution(sol: scoring.RoleSolution, p: pd.DataFrame, fit_id: str = "") -> PitchValues:
    """A P0-style array solution's values at the scored pitches."""
    u, b, c = p["u"].to_numpy(), p["base_out"].to_numpy(), p["cidx"].to_numpy()
    return PitchValues(sol.V_ref[u, b, c], sol.Q_ref[u, b, c], sol.V_opt[u, b, c], sol.Q_opt[u, b, c], sol.Q_1p[u, b, c],
                       fit_id)


def compare_to_base(base_sel, base_val, cand_sel, cand_val, slot: np.ndarray, tol: float = COMPARE_TOL) -> pd.DataFrame:
    """Zero-coefficient comparison with P0 (addendum §8, test 2). All values must agree within tol. Where the
    recommendations agree, D must agree. Each numerical-tie flip is counted, and its D difference must equal the
    valuation difference between the two recommendations: two pitches tied in selection can differ in valuation."""
    for name in VALUE_ARRAYS:
        for b, c, role in ((base_sel, cand_sel, "selection"), (base_val, cand_val, "valuation")):
            compare_values(getattr(b, name), getattr(c, name), tol, f"{role} {name}")
    rows, out = np.arange(len(slot)), []
    for label, q in (("D", "Q_ref"), ("D_opt", "Q_opt"), ("D_1p", "Q_1p")):
        qsb, qvb, qsc, qvc = getattr(base_sel, q), getattr(base_val, q), getattr(cand_sel, q), getattr(cand_val, q)
        rb, rc = qsb.argmin(1), qsc.argmin(1)
        db, dc = qvb[rows, slot] - qvb[rows, rb], qvc[rows, slot] - qvc[rows, rc]
        same, flip = rb == rc, rb != rc
        compare_values(db[same], dc[same], tol, f"{label} where recommendations agree")
        compare_values((dc - db)[flip], (qvc[rows, rb] - qvc[rows, rc])[flip], tol, f"{label} at tie flips")
        out.append({"score": label, "pitches": len(slot), "flips": int(flip.sum()),
                    "max_abs_diff_agreeing": float(np.abs(db[same] - dc[same]).max(initial=0.0)),
                    "max_flip_abs_valuation_gap": float(np.abs(qvc[rows, rb] - qvc[rows, rc])[flip].max(initial=0.0))})
    return pd.DataFrame(out)


def simulation_contexts(p: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The 4 contexts with the most scored pitches (ties by sorted key), then 4 uniform without replacement from the
    remaining observed contexts, sorted by key. There is no minimum pitch count."""
    keys = ["pitcher", "stand", "batter"]
    n = p.groupby(keys).size().rename("scored_pitches").reset_index().sort_values(keys).reset_index(drop=True)
    top = n.sort_values("scored_pitches", ascending=False, kind="stable").head(4)
    rest = n.drop(top.index).reset_index(drop=True)
    drawn = rest.iloc[np.sort(rng.choice(len(rest), 4, replace=False))]
    return pd.concat([top.assign(kind="top"), drawn.assign(kind="random")], ignore_index=True)


def reuse_sample(p: pd.DataFrame, ids: pd.DataFrame, pitch_id: np.ndarray, sel: PitchValues, season_seed: int,
                 k: int, n_main: int = 500, n_extra: int = 100, no_history=None) -> dict:
    """Pitch positions for the reuse check: `main` uniform from the fold's pitches sorted by key; then, from the same
    generator, up to n_extra each of batters without history in a role, switch hitters and numerical ties."""
    rng = np.random.default_rng([20261031, season_seed, k])
    order = p.sort_values(PITCH_KEYS, kind="stable").index.to_numpy()
    main = np.sort(rng.choice(order, min(n_main, len(order)), replace=False))

    def extra(mask):
        where = order[mask[order]]
        return np.sort(rng.choice(where, min(n_extra, len(where)), replace=False)) if len(where) else where

    sides = p.groupby("batter")["stand"].transform("nunique").to_numpy() > 1
    two = np.sort(sel.Q_ref, axis=1)[:, :2]
    ties = two[:, 1] - two[:, 0] <= 1e-12
    missing = np.zeros(len(p), bool) if no_history is None else p["batter"].isin(no_history).to_numpy()
    return {"main": main, "missing_history": extra(missing), "switch_hitter": extra(sides), "numerical_tie": extra(ties)}


def reuse_check(role, units, policy, p, ids, pitch_id, values: PitchValues, positions: np.ndarray) -> dict:
    """Solve each sampled pitch's identity alone and compare all five value arrays with the reused solution. A failure
    raises and stops the work."""
    worst = {name: 0.0 for name in VALUE_ARRAYS}
    for i in positions:
        one = ids.iloc[[pitch_id[i]]].assign(id=0)
        alone = solve_role_values(role, units, policy, one, np.zeros(1, int), p["cidx"].to_numpy()[[i]], values.fit_id)
        for name in VALUE_ARRAYS:
            worst[name] = max(worst[name], compare_values(getattr(alone, name)[0], getattr(values, name)[i], COMPARE_TOL,
                                                          f"{name} of pitch position {i}"))
    return {"pitches": len(positions), **{f"max_diff_{k}": v for k, v in worst.items()}}


def simulate_identity(role, units, policy, ctx: pd.Series, b: int, seed: list, n_sims: int = 4_000_000) -> dict:
    """Criterion 2 for one context and base-out state. A validation-only solve: the context need not have appeared in
    that state, and it enters no score. The solver's V(0-0) against simulated at-bats, contact bins drawn."""
    from gamecalling import validation
    u = units.table[(units.table["pitcher"] == ctx["pitcher"]) & (units.table["stand"] == ctx["stand"])]["u"].iloc[0]
    A = int(units.eligible[u].sum())
    one = pd.DataFrame({"u": [u], "batter": [ctx["batter"]]})
    p, pc = context_probabilities(role, one, units, A)
    pol = policy.probs[u][:, :A]
    v = solve_chunk(role, pol[None], p, pc, np.array([b]))["V_ref"][0, 0]
    sims = validation.simulate_pa(np.random.default_rng(seed), p[0], pc[0], role.payoffs.contact[b], pol,
                                  role.payoffs.walk[b], role.payoffs.strikeout[b], n_sims)
    se = sims.std() / np.sqrt(n_sims)
    return {"solver": float(v), "simulated": float(sims.mean()), "diff": float(sims.mean() - v), "se": float(se),
            "z": float((sims.mean() - v) / se), "within_tol": bool(abs(sims.mean() - v) <= validation.SIM_TOL)}
