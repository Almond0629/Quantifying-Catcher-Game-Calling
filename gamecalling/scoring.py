"""Cross-fitted scored differences for the count-based pilot (project_plan.md, Scored regret).

For scoring fold k of a season, fold k+1 is the selection role and fold k+2 the
valuation role (arsenal.fold_roles). Both are solved over the scoring fold's
eligible sets, for every pitcher x batter hand x base-out state x count.

Per scorable pitch, with s its state and a its actual pitch type:
- D  = Q_value(s, a) - Q_value(s, a_hat), a_hat = argmin_a Q_select(s, a)   (main)
- A  = Q_value(s, a) - V_value(s), V = sum_a pi_value(a|s) Q_value(s, a)    (addendum)
- B  = D - A                                                               (addendum)
- D_opt: D with both roles solved under optimal continuation (robustness)
- D_1p: D with league RE288 of each role's fold as the continuation after the pitch
  (one-pitch benchmark)
All are costs in runs (lower = better for the pitcher); display flips the sign.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gamecalling import arsenal, config, models, solver


@dataclass
class RoleSolution:
    V_ref: np.ndarray   # (U, 24, 12)
    Q_ref: np.ndarray   # (U, 24, 12, A), inf where ineligible
    V_opt: np.ndarray
    Q_opt: np.ndarray
    Q_1p: np.ndarray
    policy: models.Policy
    p_outcome: np.ndarray  # (U, 12, A, 6)
    p_contact: np.ndarray  # (U, A, 8)
    bip_value: np.ndarray  # (U, 24, 1, A)


def solve_role(role: models.Role, units: models.Units) -> RoleSolution:
    p = models.unit_outcome_probs(role.outcome, units)
    pc = models.unit_contact_probs(role.contact, units)
    policy = models.fit_policy(role.policy_train, units)
    pay = role.payoffs
    bip = np.einsum("uak,bk->uba", pc, pay.contact)[:, :, None, :]
    elig = units.eligible[:, None, None, :]
    p5 = p[:, None]
    V_ref, Q_ref = solver.evaluate(p5, bip, pay.walk, pay.strikeout, policy.probs[:, None], eligible=elig)
    V_opt, Q_opt = solver.optimize(p5, bip, pay.walk, pay.strikeout, elig)
    Q_1p = np.where(elig, solver.action_values(role.re288, p5, np.where(elig, bip, 0.0), pay.walk, pay.strikeout),
                    np.inf)
    return RoleSolution(V_ref, Q_ref, V_opt, Q_opt, Q_1p, policy, p, pc, bip)


SCORE_COLUMNS = [
    "season", "fold", "game_pk", "pa_id", "pitch_number", "game_date", "fielding_team", "catcher", "pitcher",
    "batter", "stand", "balls", "strikes", "base_out", "action", "dugout_called",
]


def _pick(Q, u, bo, c):
    return Q[u, bo, c]  # (n, A)


def score_fold(pitches: pd.DataFrame, season: int, k: int, sel: RoleSolution, val: RoleSolution,
               units: models.Units) -> pd.DataFrame:
    """Scores for the scorable pitches of one season's scoring fold."""
    p = pitches[(pitches["season"] == season) & (pitches["fold"] == k) & pitches["scorable"]]
    p = p.merge(units.slots, on=["pitcher", "stand", "action"], how="left", validate="many_to_one")
    if p["u"].isna().any():
        raise ValueError("scorable pitch outside the eligible set")
    u, a = p["u"].to_numpy(int), p["slot"].to_numpy(int)
    bo = p["base_out"].to_numpy(int)
    c = (p["balls"] * 3 + p["strikes"]).to_numpy(int)
    rows = np.arange(len(p))
    out = p[SCORE_COLUMNS + [c for c in ("w", "w_hyper") if c in p]].copy()

    def scored(Q_sel, Q_val):
        qs, qv = _pick(Q_sel, u, bo, c), _pick(Q_val, u, bo, c)
        rec = qs.argmin(axis=1)
        return qv[rows, a] - qv[rows, rec], rec, qv

    out["D"], rec, qv = scored(sel.Q_ref, val.Q_ref)
    out["Q_actual"] = qv[rows, a]
    out["V_value"] = val.V_ref[u, bo, c]
    out["A"] = out["Q_actual"] - out["V_value"]
    out["B"] = out["D"] - out["A"]
    out["rec_action"] = units.actions[u, rec]
    out["agree"] = rec == a
    out["pi_actual"] = val.policy.probs[u, c, a]
    out["D_opt"], rec_opt, _ = scored(sel.Q_opt, val.Q_opt)
    out["agree_opt"] = rec_opt == a
    out["D_1p"], rec_1p, _ = scored(sel.Q_1p, val.Q_1p)
    out["agree_1p"] = rec_1p == a
    out["n_eligible"] = units.eligible[u].sum(axis=1)
    return out


@dataclass
class PilotRun:
    scores: pd.DataFrame
    roles: dict          # (season, fold) -> models.Role
    solutions: dict      # (season, scoring fold) -> {"select": RoleSolution, "value": RoleSolution, "units": Units}
    timings: dict


def run_pilot(pitches: pd.DataFrame, pas: pd.DataFrame, support: pd.DataFrame, seasons=config.SEASONS,
              keep_solutions=False, with_re288=True) -> PilotRun:
    import time
    timings = {"fit_role": [], "solve_role": [], "score_fold": []}
    roles = {}
    for season in seasons:
        for f in range(config.N_FOLDS):
            t = time.perf_counter()
            roles[season, f] = models.fit_role(pitches, pas, season, f, with_re288)
            timings["fit_role"].append(time.perf_counter() - t)
    eligible = arsenal.eligible_sets(support)
    scores, solutions = [], {}
    for season in seasons:
        for k in range(config.N_FOLDS):
            units = models.build_units(eligible[(eligible["season"] == season) & (eligible["scoring_fold"] == k)])
            sel_fold, val_fold = arsenal.fold_roles(k)
            t = time.perf_counter()
            sel = solve_role(roles[season, sel_fold], units)
            val = solve_role(roles[season, val_fold], units)
            timings["solve_role"] += [(time.perf_counter() - t) / 2] * 2
            t = time.perf_counter()
            scores.append(score_fold(pitches, season, k, sel, val, units))
            timings["score_fold"].append(time.perf_counter() - t)
            if keep_solutions:
                solutions[season, k] = {"select": sel, "value": val, "units": units}
    return PilotRun(pd.concat(scores, ignore_index=True), roles, solutions, timings)
