"""Count-graph solver for one at-bat (project_plan.md, Solver).

States are the 12 counts, indexed balls * 3 + strikes. From a count, a pitch
leads to one of six outcomes, in OUTCOMES order:

- ball: one more ball, or a walk at 3 balls
- called or swinging strike: one more strike, or a strikeout at 2 strikes
- foul: one more strike below 2 strikes; the same count at 2 strikes
- hbp: a walk-valued ending
- in_play: the expected value of a ball in play (bip_value)

Values are expected runs (a cost for the pitcher). Base-out state is fixed within
a solve and enters only through the terminal values, so solving all 24 states at
once is a batch dimension: arrays share any leading batch shape and broadcast.

Two solves on the same graph:
- evaluate: V(c) = sum_a policy(a|c) Q(c, a), the reference continuation.
- optimize: V(c) = min over eligible a of Q(c, a).
Both repeat passes over all counts until no value changes by more than `tol`;
the two-strike foul loop is what needs more than one pass.
"""

import numpy as np

from gamecalling import config

OUTCOMES = ("ball", "called_strike", "swinging_strike", "foul", "hbp", "in_play")
N_COUNTS = 12
WALK, STRIKEOUT = N_COUNTS, N_COUNTS + 1  # terminal slots appended to V


def count_index(balls, strikes):
    return np.asarray(balls) * 3 + np.asarray(strikes)


def _next_states():
    balls, strikes = np.divmod(np.arange(N_COUNTS), 3)
    after_ball = np.where(balls == 3, WALK, count_index(np.minimum(balls + 1, 3), strikes))
    after_strike = np.where(strikes == 2, STRIKEOUT, count_index(balls, np.minimum(strikes + 1, 2)))
    after_foul = count_index(balls, np.minimum(strikes + 1, 2))
    return after_ball, after_strike, after_foul


AFTER_BALL, AFTER_STRIKE, AFTER_FOUL = _next_states()


def action_values(V, p_outcome, bip_value, walk_value, strikeout_value):
    """Q(c, a) given values V (..., 12) of every count.

    p_outcome (..., 12, A, 6), bip_value (..., 12, A), walk_value and strikeout_value (...).
    """
    walk_value, strikeout_value = np.asarray(walk_value, float), np.asarray(strikeout_value, float)
    V = np.asarray(V, float)
    shape = np.broadcast_shapes(V.shape[:-1], walk_value.shape, strikeout_value.shape)
    ext = np.concatenate([np.broadcast_to(V, shape + (N_COUNTS,)),
                          np.broadcast_to(walk_value, shape)[..., None],
                          np.broadcast_to(strikeout_value, shape)[..., None]], axis=-1)
    p = p_outcome
    return (
        p[..., 0] * ext[..., AFTER_BALL, None]
        + (p[..., 1] + p[..., 2]) * ext[..., AFTER_STRIKE, None]
        + p[..., 3] * ext[..., AFTER_FOUL, None]
        + p[..., 4] * walk_value[..., None, None]
        + p[..., 5] * bip_value
    )


def _check(p_outcome, eligible):
    if p_outcome.shape[-3] != N_COUNTS or p_outcome.shape[-1] != len(OUTCOMES):
        raise ValueError(f"p_outcome must end in (12, A, 6), got {p_outcome.shape}")
    if not np.all(eligible.any(axis=-1)):
        raise ValueError("a count has no eligible action")
    p = np.where(eligible[..., None], p_outcome, 0.0)
    if not np.allclose(np.where(eligible, p.sum(axis=-1), 1.0), 1.0):
        raise ValueError("outcome probabilities of an eligible action must sum to 1")
    if np.any(p < 0):
        raise ValueError("negative outcome probability")


def _iterate(step, shape, tol, max_iter):
    V = np.zeros(shape)
    for _ in range(max_iter):
        V_new, Q = step(V)
        if np.max(np.abs(V_new - V), initial=0.0) < tol:
            return V_new, Q
        V = V_new
    raise RuntimeError(f"solver did not converge to {tol} in {max_iter} passes")


def _value_shape(p_outcome, bip_value, walk_value, mask):
    return np.broadcast_shapes(p_outcome.shape[:-2], np.shape(bip_value)[:-1], mask.shape[:-1],
                               np.shape(walk_value) + (N_COUNTS,))


def evaluate(p_outcome, bip_value, walk_value, strikeout_value, policy, eligible=None,
             tol=config.SOLVER_TOL, max_iter=10_000):
    """Reference-policy evaluation. Returns V (..., 12) and Q (..., 12, A).

    policy (..., 12, A) is a probability over actions in each count. `eligible`
    (default: policy > 0) marks the actions whose Q is wanted; it must include
    every action with positive weight, and may include eligible actions the policy
    never uses. Q of an ineligible action is +inf.
    """
    policy = np.asarray(policy, float)
    eligible = policy > 0 if eligible is None else np.asarray(eligible, bool)
    if not np.allclose(policy.sum(axis=-1), 1.0) or np.any(policy < 0):
        raise ValueError("policy rows must be probabilities summing to 1")
    if np.any((policy > 0) & ~eligible):
        raise ValueError("policy puts weight on an ineligible action")
    p_outcome = np.where(eligible[..., None], p_outcome, 0.0)
    bip_value = np.where(eligible, bip_value, 0.0)
    _check(p_outcome, eligible)

    def step(V):
        Q = action_values(V, p_outcome, bip_value, walk_value, strikeout_value)
        return (policy * Q).sum(axis=-1), np.where(eligible, Q, np.inf)

    return _iterate(step, _value_shape(p_outcome, bip_value, walk_value, eligible), tol, max_iter)


def optimize(p_outcome, bip_value, walk_value, strikeout_value, eligible,
             tol=config.SOLVER_TOL, max_iter=10_000):
    """Optimal continuation: V(c) = min over eligible a of Q(c, a). Returns V and Q.

    Q of an ineligible action is +inf.
    """
    eligible = np.asarray(eligible, bool)
    p_outcome = np.where(eligible[..., None], p_outcome, 0.0)
    bip_value = np.where(eligible, bip_value, 0.0)
    _check(p_outcome, eligible)

    def step(V):
        Q = np.where(eligible, action_values(V, p_outcome, bip_value, walk_value, strikeout_value), np.inf)
        return Q.min(axis=-1), Q

    return _iterate(step, _value_shape(p_outcome, bip_value, walk_value, eligible), tol, max_iter)
