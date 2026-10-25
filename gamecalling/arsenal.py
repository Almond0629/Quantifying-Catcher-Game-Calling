"""Game folds, the arsenal support rule and the eligible action set.

Within each season, games are split into three folds by original game_pk. When
fold k is scored, fold k+1 selects the recommended pitch and fold k+2 values
both pitches; together they are the training folds for fold k.

An action (a pitch type, or a family in the sensitivity check) is eligible for
a pitcher against a batter hand when, in the two training folds combined, it
reaches the role's share and pitch-count cutoffs against that hand. The pitcher
must have pitched in both training folds. The scoring fold is never used.
"""

import numpy as np
import pandas as pd

from gamecalling import config

SUPPORT_KEY = ["season", "pitcher", "stand", "action", "scoring_fold"]


def assign_folds(pitches: pd.DataFrame, n_folds=config.N_FOLDS, seed=config.FOLD_SEED) -> pd.Series:
    """Fold for each game_pk, balanced within season and fixed by original game ID."""
    folds = {}
    for season, games in pitches.groupby("season")["game_pk"]:
        ids = np.sort(games.unique())
        rng = np.random.default_rng([seed, int(season)])
        order = rng.permutation(len(ids))
        folds.update(zip(ids[order], np.arange(len(ids)) % n_folds))
    return pd.Series(folds, name="fold").rename_axis("game_pk").astype("int8")


def fold_roles(scoring_fold: int, n_folds=config.N_FOLDS) -> tuple[int, int]:
    """(selection fold, valuation fold) for a scoring fold."""
    return (scoring_fold + 1) % n_folds, (scoring_fold + 2) % n_folds


def pitcher_roles(pitches: pd.DataFrame) -> pd.DataFrame:
    """Games and starts for each (season, pitcher, fold).

    A start is an appearance where the pitcher threw his team's first pitch of the game.
    """
    starters = pitches.groupby(["game_pk", "fielding_team"]).head(1)[["game_pk", "pitcher"]]
    apps = pitches[["season", "fold", "game_pk", "pitcher"]].drop_duplicates()
    apps = apps.merge(starters.assign(start=True), on=["game_pk", "pitcher"], how="left")
    return apps.groupby(["season", "pitcher", "fold"]).agg(
        games=("game_pk", "size"), starts=("start", "count")).reset_index()


def training_roles(
    roles: pd.DataFrame, n_folds=config.N_FOLDS, sp_min_start_share=config.SP_MIN_START_SHARE
) -> pd.DataFrame:
    """SP or RP for each (season, pitcher, scoring_fold), from the two training folds combined.

    Only pitchers who appear in both training folds get a row.
    """
    rows = []
    for k in range(n_folds):
        sel, val = fold_roles(k, n_folds)
        both = roles[roles["fold"] == sel].merge(
            roles[roles["fold"] == val], on=["season", "pitcher"], suffixes=("_s", "_v"))
        rows.append(pd.DataFrame({
            "season": both["season"], "pitcher": both["pitcher"], "scoring_fold": k,
            "games": both["games_s"] + both["games_v"], "starts": both["starts_s"] + both["starts_v"],
        }))
    out = pd.concat(rows, ignore_index=True)
    out["role"] = np.where(out["starts"] / out["games"] >= sp_min_start_share, "SP", "RP")
    return out


def support_table(
    pitches: pd.DataFrame, rules=None, roles: pd.DataFrame | None = None, n_folds=config.N_FOLDS
) -> pd.DataFrame:
    """Usage of each action against each batter hand in the training folds of each scoring fold.

    Usage counts thrown pitches with an action. The share is out of the
    pitcher's thrown pitches with an action against that hand, in the two training
    folds combined. `roles` is a pitcher_roles table; a pitcher missing from either
    training fold gets no rows.
    """
    rules = rules or config.ARSENAL_RULES
    if roles is None:
        roles = pitcher_roles(pitches)
    usable = pitches[pitches["thrown"] & pitches["action"].notna()]
    counts = usable.groupby(["season", "pitcher", "fold", "stand", "action"]).size().rename("n").reset_index()
    rows = []
    for k in range(n_folds):
        train = counts[counts["fold"].isin(fold_roles(k, n_folds))]
        rows.append(
            train.groupby(["season", "pitcher", "stand", "action"])["n"].sum().reset_index().assign(scoring_fold=k)
        )
    sup = pd.concat(rows, ignore_index=True)
    sup["share"] = sup["n"] / sup.groupby(["season", "pitcher", "stand", "scoring_fold"])["n"].transform("sum")
    sup = sup.merge(training_roles(roles, n_folds)[["season", "pitcher", "scoring_fold", "role"]],
                    on=["season", "pitcher", "scoring_fold"])
    min_share = sup["role"].map({r: c["min_share"] for r, c in rules.items()})
    min_count = sup["role"].map({r: c["min_count"] for r, c in rules.items()})
    sup["passes"] = (sup["share"] >= min_share) & (sup["n"] >= min_count)
    return sup[SUPPORT_KEY + ["role", "n", "share", "passes"]]


def eligible_sets(support: pd.DataFrame) -> pd.DataFrame:
    """Eligible (season, pitcher, stand, action, scoring_fold) rows."""
    return support.loc[support["passes"], SUPPORT_KEY].reset_index(drop=True)


def mark_arsenal(pitches: pd.DataFrame, eligible: pd.DataFrame, n_folds=config.N_FOLDS) -> pd.DataFrame:
    """Add in_eligible_set, arsenal_reason and scorable to the pitch table.

    arsenal_reason separates a pitcher with no pitches in a training fold from an
    action that failed support against that batter hand (including a hand he
    never faced in the training folds).
    """
    pitches = pitches.drop(columns=["in_eligible_set", "arsenal_reason", "scorable"], errors="ignore")
    elig = eligible.rename(columns={"scoring_fold": "fold"}).assign(in_eligible_set=True)
    out = pitches.merge(elig, on=["season", "pitcher", "stand", "action", "fold"], how="left")
    out["in_eligible_set"] = out["in_eligible_set"].eq(True)

    present = pitches[pitches["thrown"] & pitches["action"].notna()][["season", "pitcher", "fold"]]
    present = present.drop_duplicates()
    absent = pd.Series(False, index=out.index)
    for offset in (1, 2):
        train_fold = ((out["fold"] + offset) % n_folds).rename("fold")
        probe = pd.concat([out[["season", "pitcher"]], train_fold], axis=1)
        hit = probe.merge(present.assign(_p=True), on=["season", "pitcher", "fold"], how="left")["_p"]
        absent |= hit.isna().to_numpy()

    candidate = out["exclusion_reason"].isna()
    reason = pd.Series(pd.NA, index=out.index, dtype="object")
    reason[candidate & ~out["in_eligible_set"] & absent] = "pitcher_missing_from_role_fold"
    reason[candidate & ~out["in_eligible_set"] & ~absent] = "below_support"
    out["arsenal_reason"] = reason
    out["scorable"] = candidate & out["in_eligible_set"]
    return out
