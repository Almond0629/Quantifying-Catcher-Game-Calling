"""M1: runners-on flag on the outcome and contact models (research program S4).

Specification: research_notes/research_program/S4_runners_on.md (revision 1). P0's pooled models
(models.py) are kept as they are; M1 adds one level on top of each:

- league runner level: P0's league keys x runners_on, pulled toward P0's league mean L with alpha_league_r;
- runner cell: P0's pitcher-cell keys x runners_on, pulled toward
      m_r proportional to P0_cell x L_r / L      (renormalized over categories)
  with alpha_r; the prediction is (n_r + alpha_r m_r) / (N_r + alpha_r).

runners_on is the pitch table's flag (any occupied base at the pitch). Concentrations are fitted as in
models.fit_pooled: prior means from the `w` counts, alpha by marginal likelihood on the cells with `w_hyper`
counts, from the role's own fold.

The solver uses, for base-out state b = bases * 3 + outs, the runners_on = (b >= 3) slice of the
probabilities; all 24 states are still solved at once. Every function takes an explicit switch
(`runners=False`) that calls only P0's own functions (models.fit_outcome_model, models.fit_contact_model,
scoring.solve_role).
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gamecalling import models, pooling, scoring, solver

RUNNER = "runners_on"
COMPONENTS = ("outcome", "contact")
SLICE = (np.arange(24) >= 3).astype(int)   # runners_on slice of each base-out state


@dataclass
class RunnerModel:
    """P0's pooled model plus a league runner level and runner cells."""
    p0: models.PooledModel
    league_r_n: pd.DataFrame   # counts, index = P0 league keys + runners_on
    cell_r_n: pd.DataFrame     # counts, index = P0 cell keys + runners_on
    alpha_league_r: float
    alpha_r: float
    by_runners = True

    @property
    def alpha_league(self):
        return self.p0.alpha_league

    @property
    def alpha_cell(self):
        return self.p0.alpha_cell

    def _keys(self, level):
        return list(self.p0.levels[level]) + [RUNNER]

    def parts(self, query: pd.DataFrame) -> dict:
        """L, L_r, P0_cell and m_r for query rows (P0 keys and runners_on)."""
        L = self.p0.league_mean(query)
        P = pooling.posterior_mean(models._lookup(self.p0.cell_n, query, self.p0.levels[2]), L, self.p0.alpha_cell)
        Lr = pooling.posterior_mean(models._lookup(self.league_r_n, query, self._keys(1)), L, self.alpha_league_r)
        m = P * Lr / L
        return {"L": L, "L_r": Lr, "P0_cell": P, "m_r": m / m.sum(-1, keepdims=True)}

    def prior_mean(self, query: pd.DataFrame) -> np.ndarray:
        return self.parts(query)["m_r"]

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        return pooling.posterior_mean(models._lookup(self.cell_r_n, query, self._keys(2)), self.prior_mean(query),
                                      self.alpha_r)


def fit_runner_pooled(df: pd.DataFrame, levels, K: int, p0_alphas=None, alphas=None) -> RunnerModel:
    """df: P0's training frame plus runners_on (0/1). `p0_alphas` and `alphas` (league_r, r) override the
    fitted concentrations (tests); a None entry is fitted."""
    alphas = alphas or (None, None)
    p0 = models.fit_pooled(df, levels, K, p0_alphas)
    hw = "w_hyper" if "w_hyper" in df else "w"
    k1, k2 = list(levels[1]) + [RUNNER], list(levels[2]) + [RUNNER]
    n1 = models._counts(df, k1, K)
    n1h = models._counts(df, k1, K, hw)
    n1h = n1h[n1h.sum(axis=1) > 0]
    a1 = alphas[0] if alphas[0] is not None else \
        pooling.fit_alpha(n1h.to_numpy(), p0.league_mean(n1h.index.to_frame(index=False)))
    model = RunnerModel(p0, n1, models._counts(df, k2, K), a1, np.nan)
    if alphas[1] is not None:
        return replace(model, alpha_r=alphas[1])
    n2h = models._counts(df, k2, K, hw)
    n2h = n2h[n2h.sum(axis=1) > 0]
    return replace(model, alpha_r=pooling.fit_alpha(n2h.to_numpy(), model.prior_mean(n2h.index.to_frame(index=False))))


# --- Training frames and fitting ---

def _with_runners(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(**{RUNNER: rows.loc[frame.index, RUNNER].astype(int).to_numpy()})


def outcome_training(pitches: pd.DataFrame) -> pd.DataFrame:
    return _with_runners(models.outcome_training(pitches), pitches)


def contact_training(pitches: pd.DataFrame) -> pd.DataFrame:
    return _with_runners(models.contact_training(pitches), pitches)


def training(kind: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Held-out and forward observations (research.Spec.training); P0 models ignore the extra column."""
    return outcome_training(rows) if kind == "outcome" else contact_training(rows)


def fit_models(train: pd.DataFrame, runners: bool = True, components=COMPONENTS):
    """Outcome and contact models of one role. runners=False: P0's own functions only. `components` lists the
    models that get the runner levels (component-wise adoption); the others are P0's."""
    use = set(components) if runners else set()
    outcome = fit_runner_pooled(outcome_training(train), models.OUTCOME_LEVELS, models.N_OUTCOMES) \
        if "outcome" in use else models.fit_outcome_model(train)
    contact = fit_runner_pooled(contact_training(train), models.CONTACT_LEVELS, models.N_BINS) \
        if "contact" in use else models.fit_contact_model(train)
    return outcome, contact


# --- Unit probabilities and the solve ---

def _runner_frame():
    return pd.DataFrame({RUNNER: [0, 1]})


def unit_outcome_probs(model, units: models.Units) -> np.ndarray:
    """(U, 2, 12, A, 6) by runners_on; a P0 model gives the same probabilities in both slices."""
    if not getattr(model, "by_runners", False):
        p = models.unit_outcome_probs(model, units)
        return np.stack([p, p], axis=1)
    U, A = units.eligible.shape
    q = units.slots.merge(pd.DataFrame({"cidx": np.arange(solver.N_COUNTS)}), how="cross") \
        .merge(_runner_frame(), how="cross")
    out = np.zeros((U, 2, solver.N_COUNTS, A, models.N_OUTCOMES))
    out[..., 0] = 1.0
    out[q["u"], q[RUNNER], q["cidx"], q["slot"]] = model.predict(q)
    return out


def unit_contact_probs(model, units: models.Units) -> np.ndarray:
    """(U, 2, A, 8) by runners_on."""
    if not getattr(model, "by_runners", False):
        p = models.unit_contact_probs(model, units)
        return np.stack([p, p], axis=1)
    U, A = units.eligible.shape
    out = np.full((U, 2, A, models.N_BINS), 1.0 / models.N_BINS)
    q = units.slots.merge(_runner_frame(), how="cross")
    out[q["u"], q[RUNNER], q["slot"]] = model.predict(q)
    return out


def solve_sliced(role: models.Role, units: models.Units, p: np.ndarray, pc: np.ndarray,
                 fit_policy=models.fit_policy) -> scoring.RoleSolution:
    """scoring.solve_role with runners-on probabilities: p (U, 2, 12, A, 6), pc (U, 2, A, 8). Each base-out
    state uses its own slice; given identical slices the arithmetic is P0's, element by element."""
    policy = fit_policy(role.policy_train, units)
    pay = role.payoffs
    bip_r = [np.einsum("uak,bk->uba", np.ascontiguousarray(pc[:, r]), pay.contact) for r in (0, 1)]
    on = SLICE.astype(bool)[None, :, None]
    bip = np.where(on, bip_r[1], bip_r[0])[:, :, None, :]
    elig = units.eligible[:, None, None, :]
    pb = p[:, SLICE]
    V_ref, Q_ref = solver.evaluate(pb, bip, pay.walk, pay.strikeout, policy.probs[:, None], eligible=elig)
    V_opt, Q_opt = solver.optimize(pb, bip, pay.walk, pay.strikeout, elig)
    Q_1p = np.where(elig, solver.action_values(role.re288, pb, np.where(elig, bip, 0.0), pay.walk, pay.strikeout),
                    np.inf)
    return scoring.RoleSolution(V_ref, Q_ref, V_opt, Q_opt, Q_1p, policy, p, pc, bip)


def solve_role(role: models.Role, units: models.Units, fit_policy=models.fit_policy,
               runners: bool = True) -> scoring.RoleSolution:
    if not runners:
        return scoring.solve_role(role, units, fit_policy)
    return solve_sliced(role, units, unit_outcome_probs(role.outcome, units),
                        unit_contact_probs(role.contact, units), fit_policy)
