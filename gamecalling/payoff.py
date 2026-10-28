"""Run-valued payoffs: the RE24 table and the terminal values of an at-bat.

Each role (selection or valuation) estimates its payoffs from its own fold only,
within the season (project_plan.md, Payoff; decided 2026-10-27). Every value is
expected runs for the batting team from that point to the end of the half-inning,
so lower is better for the pitcher.

- RE24: mean runs_rest_of_inning by PA-start base-out state, from re24_eligible PAs.
- Strikeout: RE24 of the same runners with one more out (0 when that is the third out).
- Walk or HBP: forced runs + RE24 of the forced base state, same outs.
- Ball in play: for each final_base_out x contact bin, the mean of runs on the
  final pitch + RE24 of the next PA's start state (0 after three outs), from PAs
  that are payoff_eligible and ended in play.

Base-out index: bases bitmask (1st=1, 2nd=2, 3rd=4) * 3 + outs, so 0..23.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gamecalling import config, weighting

N_BASE_OUT = 24
CONTACT_BINS = tuple(range(1, 9))  # launch_speed_angle 1-6, bunt (7), untracked (8)


@dataclass(frozen=True)
class Payoffs:
    """Terminal values for one role's fold in one season, each indexed by base-out state."""
    re24: np.ndarray            # (24,)
    strikeout: np.ndarray       # (24,)
    walk: np.ndarray            # (24,) also used for HBP
    contact: np.ndarray         # (24, 8) value of each contact bin
    contact_n: np.ndarray       # (24, 8) PAs behind each empirical cell
    contact_fallback: np.ndarray  # (24, 8) True where the fallback replaced the empirical mean


def base_out_index(bases, outs):
    return np.asarray(bases) * 3 + np.asarray(outs)


def re24_table(pas: pd.DataFrame) -> np.ndarray:
    """Mean runs from the PA start to the end of the half-inning, by start base-out state."""
    use = pas[pas["re24_eligible"]]
    w = pd.Series(weighting.w(use), index=use.index)
    num = (use["runs_rest_of_inning"] * w).groupby(use["start_base_out"]).sum()
    table = (num / w.groupby(use["start_base_out"]).sum()).reindex(range(N_BASE_OUT))
    if table.isna().any():
        raise ValueError(f"RE24 states with no PAs: {table.index[table.isna()].tolist()}")
    return table.to_numpy(dtype="float64")


def re24_after(re24: np.ndarray, bases, outs) -> np.ndarray:
    """RE24 of a state, or 0 once the inning is over."""
    outs = np.asarray(outs)
    idx = base_out_index(bases, np.minimum(outs, 2))
    return np.where(outs >= 3, 0.0, re24[idx])


def strikeout_values(re24: np.ndarray) -> np.ndarray:
    bases, outs = np.divmod(np.arange(N_BASE_OUT), 3)
    return re24_after(re24, bases, outs + 1)


def walk_advance(bases):
    """Base state and runs after a walk or HBP: only forced runners move."""
    bases = np.asarray(bases)
    new = bases | 1 | np.where(bases & 1, 2, 0) | np.where((bases & 3) == 3, 4, 0)
    runs = (bases == 7).astype(int)
    return new, runs


def walk_values(re24: np.ndarray) -> np.ndarray:
    bases, outs = np.divmod(np.arange(N_BASE_OUT), 3)
    new, runs = walk_advance(bases)
    return runs + re24_after(re24, new, outs)


def contact_table(pas: pd.DataFrame, re24: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean value and count of each final_base_out x contact bin (NaN where empty)."""
    use = pas[pas["payoff_eligible"] & pas["pa_end"].eq("in_play")]
    if use["final_contact_bin"].isna().any():
        raise ValueError("ball-in-play PA without a contact bin")
    after = np.where(
        use["pa_end_type"].eq("continues"),
        re24_after(re24, use["next_bases"].fillna(0).astype(int), use["next_outs"].fillna(3).astype(int)),
        0.0,
    )
    value = use["runs_on_final_pitch"].to_numpy() + after
    df = pd.DataFrame({"bo": use["final_base_out"].to_numpy(), "bin": use["final_contact_bin"].astype(int).to_numpy(),
                       "value": value, "w": weighting.w(use)})
    grid = pd.MultiIndex.from_product([range(N_BASE_OUT), CONTACT_BINS])
    g = _wmean_size(df, ["bo", "bin"], "value").reindex(grid)
    shape = (N_BASE_OUT, len(CONTACT_BINS))
    return g["mean"].to_numpy().reshape(shape), g["size"].fillna(0).to_numpy(dtype=int).reshape(shape)


def _wmean_size(df: pd.DataFrame, keys, col: str) -> pd.DataFrame:
    """Weighted mean of `col` and total weight ("size") by keys; df has a `w` column."""
    g = df.assign(_wx=df[col] * df["w"]).groupby(keys)[["_wx", "w"]].sum()
    return pd.DataFrame({"mean": g["_wx"] / g["w"], "size": g["w"]})


def shift_fallback(mean: np.ndarray, n: np.ndarray, re24: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill (base-out x column) cells with fewer than min_n PAs: RE24 of the state + the column's average shift.

    The column's shift is the weighted mean of (cell mean - RE24 of the state) over
    that column's cells that pass, weighted by their counts, all from the same
    role's fold. Columns are contact bins for ball-in-play payoffs and counts for RE288.
    """
    ok = n >= min_n
    delta = np.where(ok, mean - re24[:, None], 0.0)
    w = np.where(ok, n, 0)
    if (w.sum(axis=0) == 0).any():
        raise ValueError("a column has no usable cells")
    col_delta = (delta * w).sum(axis=0) / w.sum(axis=0)
    filled = np.where(ok, mean, re24[:, None] + col_delta[None, :])
    return filled, ~ok


def re288_table(pitches: pd.DataFrame, pas: pd.DataFrame, re24: np.ndarray,
                min_n: int = config.MIN_CONTACT_CELL_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean runs from a pitch to the end of the half-inning, by base-out state x count (24 x 12).

    From thrown pitches of re24_eligible PAs without sequence errors. `pitches` must
    hold whole half-innings (one fold's games). Empty cells use shift_fallback.
    Returns the filled table, the counts and the fallback mask.
    """
    half = ["game_pk", "inning", "inning_topbot"]
    end = pitches.groupby(half)["post_bat_score"].transform("max")
    ok = pitches["pa_id"].map(pas.set_index("pa_id")["re24_eligible"]).fillna(False).astype(bool)
    use = pitches["thrown"] & ~pitches["pa_sequence_error"] & ok
    df = pd.DataFrame({"bo": pitches.loc[use, "base_out"].astype(int),
                       "c": (pitches.loc[use, "balls"] * 3 + pitches.loc[use, "strikes"]).astype(int),
                       "runs": (end - pitches["bat_score"])[use], "w": weighting.w(pitches[use])})
    grid = pd.MultiIndex.from_product([range(N_BASE_OUT), range(12)])
    g = _wmean_size(df, ["bo", "c"], "runs").reindex(grid)
    mean = g["mean"].to_numpy().reshape(N_BASE_OUT, 12)
    n = g["size"].fillna(0).to_numpy(dtype=int).reshape(N_BASE_OUT, 12)
    filled, fallback = shift_fallback(mean, n, re24, min_n)
    return filled, n, fallback


def role_payoffs(pas: pd.DataFrame, season: int, fold: int | None,
                 min_contact_n: int = config.MIN_CONTACT_CELL_N) -> Payoffs:
    """Payoffs from one season's fold (all folds if fold is None). `pas` needs the `fold` column."""
    own = pas[(pas["season"] == season) & ((pas["fold"] == fold) if fold is not None else True)]
    re24 = re24_table(own)
    mean, n = contact_table(own, re24)
    contact, fallback = shift_fallback(mean, n, re24, min_contact_n)
    return Payoffs(re24=re24, strikeout=strikeout_values(re24), walk=walk_values(re24),
                   contact=contact, contact_n=n, contact_fallback=fallback)
