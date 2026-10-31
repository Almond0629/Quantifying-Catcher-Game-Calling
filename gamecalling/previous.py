"""M2a / M2b: previous pitch on the outcome model (research program S5).

Specification: research_notes/research_program/S5_previous_pitch.md (revision 2). The contact model stays
P0's, and M2 is built on P0 alone (not on M1c).

Both variants are ratio-shifted extensions of a parent pooled model, the construction approved for M1:
- league level: the parent's league keys plus the extra keys, pulled toward the parent's league mean;
- cell: the parent's cell keys plus the extra keys, with prior mean
      m proportional to parent_cell x L_x / L_parent      (renormalized)
  and prediction (n + alpha m) / (N + alpha).
M2a: parent P0, extra key `prev`. M2b: parent M2a (refitted in the same role), extra keys `prev` and
`prev_result`. M2b's exception: for prev in {none, non_action} it returns M2a's prediction directly, and those
rows are left out of M2b's additional counts and concentrations (they stay in M2a's).

prev: `none` (no earlier thrown pitch in the PA), the previous pitch type, or `non_action` (a thrown
non-action type). prev_result: the loader's four categories; `none` for the exception rows, whose result is
deliberately ignored.

Concentrations are fitted as in models.fit_pooled: prior means from the `w` counts, alpha by marginal
likelihood on the cells with `w_hyper` counts, from the role's own fold. `prev=False` calls only P0's own
functions.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gamecalling import models, pooling

PREV, RESULT = "prev", "prev_result"
NONE, NON_ACTION = "none", "non_action"
EXCEPTION = (NONE, NON_ACTION)
EXTRA = {"M2a": (PREV,), "M2b": (PREV, RESULT)}
SUFFIX = {(PREV,): "p", (PREV, RESULT): "pr"}


def prev_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """prev and prev_result of pitch rows (index preserved)."""
    action = rows["prev_action"].astype(object)
    prev = action.where(action.notna(), NON_ACTION).where(rows["has_prev_pitch"].astype(bool), NONE)
    result = rows["prev_result"].astype(object).where(~prev.isin(EXCEPTION), NONE)
    if result.isna().any():
        raise ValueError("a previous pitch with an action has no previous result")
    return pd.DataFrame({PREV: prev, RESULT: result}, index=rows.index)


def _with_prev(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    cols = prev_columns(rows.loc[frame.index])
    return frame.assign(**{PREV: cols[PREV].to_numpy(), RESULT: cols[RESULT].to_numpy()})


def outcome_training(pitches: pd.DataFrame) -> pd.DataFrame:
    return _with_prev(models.outcome_training(pitches), pitches)


def training(kind: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Held-out and forward observations (research.Spec.training); P0's models ignore the extra columns."""
    if kind == "outcome":
        return outcome_training(rows)
    return _with_prev(models.contact_training(rows), rows)


@dataclass
class ShiftModel:
    """A parent pooled model (models.PooledModel or ShiftModel) extended by extra keys."""
    parent: object
    extra: tuple
    league_n: pd.DataFrame   # counts, index = P0 league keys + extra
    cell_n: pd.DataFrame     # counts, index = P0 cell keys + extra
    alpha_league_x: float
    alpha_x: float
    exception: bool = False  # M2b: prev in EXCEPTION returns the parent's prediction

    @property
    def levels(self):
        return self.parent.levels

    def _keys(self, level):
        return list(self.levels[level]) + list(self.extra)

    def alphas(self) -> dict:
        own = {f"league_{SUFFIX[self.extra]}": self.alpha_league_x, SUFFIX[self.extra]: self.alpha_x}
        parent = self.parent.alphas() if hasattr(self.parent, "alphas") else \
            {"league": self.parent.alpha_league, "cell": self.parent.alpha_cell}
        return {**parent, **own}

    def _league(self, query: pd.DataFrame):
        Lp = self.parent.league_mean(query)
        return Lp, pooling.posterior_mean(models._lookup(self.league_n, query, self._keys(1)), Lp, self.alpha_league_x)

    def parts(self, query: pd.DataFrame) -> dict:
        """The parent's league mean and cell prediction, this level's league mean L_x, and the prior mean m_x."""
        Lp, Lx = self._league(query)
        Pp = self.parent.predict(query)
        m = Pp * Lx / Lp
        return {"L_parent": Lp, "P_parent": Pp, "L_x": Lx, "m_x": m / m.sum(-1, keepdims=True)}

    def _exception(self, query: pd.DataFrame) -> np.ndarray:
        return query[PREV].isin(EXCEPTION).to_numpy()[:, None] if self.exception else np.zeros((len(query), 1), bool)

    def league_mean(self, query: pd.DataFrame) -> np.ndarray:
        """League keys only (no pitcher needed)."""
        Lp, Lx = self._league(query)
        return np.where(self._exception(query), Lp, Lx)

    def prior_mean(self, query: pd.DataFrame) -> np.ndarray:
        return self.parts(query)["m_x"]

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        p = self.parts(query)
        post = pooling.posterior_mean(models._lookup(self.cell_n, query, self._keys(2)), p["m_x"], self.alpha_x)
        return np.where(self._exception(query), p["P_parent"], post)


def fit_shift(df: pd.DataFrame, parent, extra, K: int, exception: bool = False, alphas=None) -> ShiftModel:
    """Fit one extension level on a fixed, fitted parent. With `exception`, rows with prev in EXCEPTION are
    left out of this level's counts and concentrations. `alphas` (league, cell) overrides them (tests)."""
    alphas = alphas or (None, None)
    use = df[~df[PREV].isin(EXCEPTION)] if exception else df
    hw = "w_hyper" if "w_hyper" in use else "w"
    k1, k2 = list(parent.levels[1]) + list(extra), list(parent.levels[2]) + list(extra)
    n1h = models._counts(use, k1, K, hw)
    n1h = n1h[n1h.sum(axis=1) > 0]
    a1 = alphas[0] if alphas[0] is not None else \
        pooling.fit_alpha(n1h.to_numpy(), parent.league_mean(n1h.index.to_frame(index=False)))
    model = ShiftModel(parent, tuple(extra), models._counts(use, k1, K), models._counts(use, k2, K), a1, np.nan,
                       exception)
    if alphas[1] is not None:
        return replace(model, alpha_x=alphas[1])
    n2h = models._counts(use, k2, K, hw)
    n2h = n2h[n2h.sum(axis=1) > 0]
    return replace(model, alpha_x=pooling.fit_alpha(n2h.to_numpy(), model.prior_mean(n2h.index.to_frame(index=False))))


def fit_outcome(df: pd.DataFrame, variant: str) -> ShiftModel:
    """M2a or M2b outcome model on a training frame with prev and prev_result (M2b refits M2a as its parent)."""
    p0 = models.fit_pooled(df, models.OUTCOME_LEVELS, models.N_OUTCOMES)
    m2a = fit_shift(df, p0, EXTRA["M2a"], models.N_OUTCOMES)
    if variant == "M2a":
        return m2a
    return fit_shift(df, m2a, EXTRA["M2b"], models.N_OUTCOMES, exception=True)


def fit_models(train: pd.DataFrame, variant: str = "M2a", prev: bool = True):
    """Outcome and contact models of one role. prev=False: P0's own functions only. Contact is always P0's."""
    if not prev:
        return models.fit_outcome_model(train), models.fit_contact_model(train)
    return fit_outcome(outcome_training(train), variant), models.fit_contact_model(train)
