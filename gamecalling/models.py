"""Pitch-outcome, contact-bin and reference-policy models for one role (project_plan.md, Phase 1).

A role is one season's fold. Each fold's models are fit on that fold only, and it
serves as the selection role for one scoring fold and the valuation role for
another (arsenal.fold_roles). Pilot state: count, batter hand and pitcher.

Pooling (empirical-Bayes Dirichlet-multinomial, decided 2026-10-27), three levels:
- base: league frequencies by count x hand (outcomes) or by hand (contact bins),
  with half a pseudo-count per category so no league probability is exactly 0;
- league: count x hand x pitch type (contact: hand x pitch type), pulled toward base;
- pitcher cell: pitcher x pitch type x count x hand (contact: pitcher x pitch type
  x hand), pulled toward the league level.
Each level has one concentration, estimated by marginal likelihood from that
role's own fold.

Reference policy: each pitcher's pitch mix by count and hand, pooled across his
catchers, over the eligible set of the scoring fold. It is pulled toward his mix
against that hand in the role's fold (Dirichlet-multinomial, one concentration).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gamecalling import payoff, pooling, solver

N_OUTCOMES = len(solver.OUTCOMES)
N_BINS = len(payoff.CONTACT_BINS)
OUTCOME_CODE = {o: i for i, o in enumerate(solver.OUTCOMES)}
BASE_PSEUDO = 0.5

OUTCOME_LEVELS = (("stand", "cidx"), ("action", "stand", "cidx"), ("pitcher", "action", "stand", "cidx"))
CONTACT_LEVELS = (("stand",), ("action", "stand"), ("pitcher", "action", "stand"))


def with_count_index(pitches: pd.DataFrame) -> pd.DataFrame:
    return pitches.assign(cidx=(pitches["balls"].astype(int) * 3 + pitches["strikes"].astype(int)))


def _counts(df: pd.DataFrame, keys, K: int) -> pd.DataFrame:
    t = df.groupby(list(keys) + ["y"]).size().unstack(fill_value=0)
    return t.reindex(columns=range(K), fill_value=0).astype(float)


def _lookup(table: pd.DataFrame, query: pd.DataFrame, keys, fill=0.0) -> np.ndarray:
    keys = list(keys)
    out = query[keys].merge(table.reset_index(), on=keys, how="left")
    return out[list(table.columns)].fillna(fill).to_numpy(dtype=float)


@dataclass
class PooledModel:
    """Base -> league -> pitcher-cell pooled categorical model."""
    levels: tuple
    base: pd.DataFrame       # probabilities, index = base keys
    league_n: pd.DataFrame   # counts, index = league keys
    cell_n: pd.DataFrame     # counts, index = cell keys
    alpha_league: float
    alpha_cell: float

    def league_mean(self, query: pd.DataFrame) -> np.ndarray:
        m0 = _lookup(self.base, query, self.levels[0], fill=np.nan)
        if np.isnan(m0).any():
            raise ValueError("query outside the base table")
        return pooling.posterior_mean(_lookup(self.league_n, query, self.levels[1]), m0, self.alpha_league)

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        m1 = self.league_mean(query)
        return pooling.posterior_mean(_lookup(self.cell_n, query, self.levels[2]), m1, self.alpha_cell)


def fit_pooled(df: pd.DataFrame, levels, K: int) -> PooledModel:
    """df has the level keys and y (category code 0..K-1)."""
    n0 = _counts(df, levels[0], K)
    base = (n0 + BASE_PSEUDO).div(n0.sum(axis=1) + K * BASE_PSEUDO, axis=0)
    n1 = _counts(df, levels[1], K)
    q1 = n1.index.to_frame(index=False)
    alpha1 = pooling.fit_alpha(n1.to_numpy(), _lookup(base, q1, levels[0]))
    n2 = _counts(df, levels[2], K)
    q2 = n2.index.to_frame(index=False)
    m1 = pooling.posterior_mean(_lookup(n1, q2, levels[1]), _lookup(base, q2, levels[0]), alpha1)
    alpha2 = pooling.fit_alpha(n2.to_numpy(), m1)
    return PooledModel(tuple(levels), base, n1, n2, alpha1, alpha2)


def outcome_training(pitches: pd.DataFrame) -> pd.DataFrame:
    use = pitches[pitches["outcome_fit_eligible"]]
    return with_count_index(use)[["pitcher", "action", "stand", "cidx"]].assign(
        y=use["outcome"].map(OUTCOME_CODE).to_numpy())


def contact_training(pitches: pd.DataFrame) -> pd.DataFrame:
    use = pitches[pitches["contact_fit_eligible"]]
    return use[["pitcher", "action", "stand"]].assign(y=use["contact_bin"].astype(int).to_numpy() - 1)


def fit_outcome_model(pitches: pd.DataFrame) -> PooledModel:
    return fit_pooled(outcome_training(pitches), OUTCOME_LEVELS, N_OUTCOMES)


def fit_contact_model(pitches: pd.DataFrame) -> PooledModel:
    return fit_pooled(contact_training(pitches), CONTACT_LEVELS, N_BINS)


# --- Units: (pitcher, batter hand) with a padded eligible action set ---

@dataclass
class Units:
    table: pd.DataFrame   # u, pitcher, stand
    slots: pd.DataFrame   # u, slot, pitcher, stand, action
    actions: np.ndarray   # (U, A) action names, None in padding
    eligible: np.ndarray  # (U, A)


def build_units(eligible_rows: pd.DataFrame) -> Units:
    """eligible_rows: (pitcher, stand, action) rows of one season's scoring fold."""
    e = eligible_rows[["pitcher", "stand", "action"]].sort_values(["pitcher", "stand", "action"])
    table = e[["pitcher", "stand"]].drop_duplicates().reset_index(drop=True)
    table["u"] = np.arange(len(table))
    e = e.merge(table, on=["pitcher", "stand"])
    e["slot"] = e.groupby("u").cumcount()
    U, A = len(table), int(e["slot"].max()) + 1
    actions = np.full((U, A), None, dtype=object)
    actions[e["u"], e["slot"]] = e["action"]
    eligible = np.zeros((U, A), bool)
    eligible[e["u"], e["slot"]] = True
    return Units(table, e[["u", "slot", "pitcher", "stand", "action"]].reset_index(drop=True), actions, eligible)


def unit_outcome_probs(model: PooledModel, units: Units) -> np.ndarray:
    """(U, 12, A, 6); padding slots are certain balls (masked by the solver)."""
    U, A = units.eligible.shape
    q = units.slots.merge(pd.DataFrame({"cidx": np.arange(solver.N_COUNTS)}), how="cross")
    out = np.zeros((U, solver.N_COUNTS, A, N_OUTCOMES))
    out[..., 0] = 1.0
    out[q["u"], q["cidx"], q["slot"]] = model.predict(q)
    return out


def unit_contact_probs(model: PooledModel, units: Units) -> np.ndarray:
    """(U, A, 8); padding slots are uniform (masked by the solver)."""
    U, A = units.eligible.shape
    out = np.full((U, A, N_BINS), 1.0 / N_BINS)
    s = units.slots
    out[s["u"], s["slot"]] = model.predict(s)
    return out


def policy_training(pitches: pd.DataFrame) -> pd.DataFrame:
    use = pitches[pitches["thrown"] & pitches["action"].notna() & ~pitches["pa_sequence_error"]]
    return with_count_index(use)[["pitcher", "stand", "action", "cidx"]]


@dataclass
class Policy:
    probs: np.ndarray        # (U, 12, A)
    alpha: float
    prior_source: pd.Series  # per unit: "hand", "both_hands" or "uniform"


def fit_policy(train: pd.DataFrame, units: Units) -> Policy:
    """Reference policy over each unit's eligible actions, from one role's fold.

    Prior mean: the pitcher's mix against that hand in the fold. If he threw no
    eligible pitch to that hand in the fold, his mix against both hands; if none
    at all, uniform over the eligible set.
    """
    U, A = units.eligible.shape
    n = np.zeros((U, solver.N_COUNTS, A))
    hit = train.merge(units.slots, on=["pitcher", "stand", "action"])
    np.add.at(n, (hit["u"].to_numpy(), hit["cidx"].to_numpy(), hit["slot"].to_numpy()), 1.0)

    hand = n.sum(axis=1)
    both_counts = train.groupby(["pitcher", "action"]).size().rename("n").reset_index()
    both = np.zeros((U, A))
    b = units.slots.merge(both_counts, on=["pitcher", "action"])
    both[b["u"], b["slot"]] = b["n"]
    uniform = units.eligible / units.eligible.sum(axis=1, keepdims=True)
    has_hand, has_both = hand.sum(1) > 0, both.sum(1) > 0
    prior = np.where(has_hand[:, None], hand / np.maximum(hand.sum(1, keepdims=True), 1),
                     np.where(has_both[:, None], both / np.maximum(both.sum(1, keepdims=True), 1), uniform))
    source = pd.Series(np.where(has_hand, "hand", np.where(has_both, "both_hands", "uniform")), name="prior_source")

    prior_c = np.broadcast_to(prior[:, None, :], n.shape)
    alpha = pooling.fit_alpha(n.reshape(-1, A), prior_c.reshape(-1, A))
    return Policy(pooling.posterior_mean(n, prior_c, alpha), alpha, source)


# --- One role's fitted pieces ---

@dataclass
class Role:
    season: int
    fold: int
    outcome: PooledModel
    contact: PooledModel
    payoffs: payoff.Payoffs
    re288: np.ndarray
    re288_n: np.ndarray
    re288_fallback: np.ndarray
    policy_train: pd.DataFrame


def fit_role(pitches: pd.DataFrame, pas: pd.DataFrame, season: int, fold: int) -> Role:
    """Fit everything one role needs from its own fold (pitches and PAs carry `fold`)."""
    p = pitches[(pitches["season"] == season) & (pitches["fold"] == fold)]
    a = pas[(pas["season"] == season) & (pas["fold"] == fold)]
    pay = payoff.role_payoffs(a, season, fold)
    re288, re288_n, re288_fb = payoff.re288_table(p, a, pay.re24)
    return Role(season, fold, fit_outcome_model(p), fit_contact_model(p), pay, re288, re288_n, re288_fb,
                policy_training(p))
