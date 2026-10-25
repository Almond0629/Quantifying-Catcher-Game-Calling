"""Turn Statcast rows into pitch and plate-appearance tables.

One row per regular-season Statcast pitch row, in at-bat order, with count,
base-out state, batter, handedness, pitcher, catcher, teams, pitch type and
family, the scored action (`action`, set by config.ACTION_UNIT), previous thrown
pitch, pitch outcome, contact bin and fold-independent exclusion reasons.
Out-of-arsenal exclusions depend on folds and are added in arsenal.py.
"""

import numpy as np
import pandas as pd

from gamecalling import config

RAW_COLUMNS = [
    "game_pk", "game_date", "game_type", "game_year", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "home_team", "away_team",
    "batter", "stand", "pitcher", "p_throws", "fielder_2",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "pitch_type", "description", "events", "des", "bb_type", "launch_speed_angle",
    "zone", "bat_score", "post_bat_score", "fld_score", "post_fld_score",
    "estimated_woba_using_speedangle",
]

# Pitches that were never thrown: the count changes but there is no pitch to score.
AUTOMATIC = {"automatic_ball", "automatic_strike"}

# Result of a thrown pitch, used for the previous-pitch state (four categories).
PREV_RESULT = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike", "swinging_strike_blocked": "swinging_strike",
    "foul_tip": "swinging_strike", "missed_bunt": "swinging_strike",
    "bunt_foul_tip": "swinging_strike",
    "foul": "foul", "foul_bunt": "foul",
}

# Outcome of a thrown pitch for the outcome model. A two-strike foul bunt is a
# strikeout, so it is mapped with swinging strikes (foul tips already are).
OUTCOME = {**PREV_RESULT, "hit_by_pitch": "hbp", "hit_into_play": "in_play"}

# Count changes implied by each description, for the sequence check.
BALL_DESCRIPTIONS = {"ball", "blocked_ball", "automatic_ball", "pitchout"}
STRIKE_DESCRIPTIONS = {
    "called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "missed_bunt", "bunt_foul_tip", "automatic_strike", "foul_bunt",
}

# Contact bins: Statcast launch_speed_angle 1-6, then bunt and untracked contact.
BUNT_BIN = 7
UNTRACKED_BIN = 8

PA_END = {
    "strikeout": "strikeout", "strikeout_double_play": "strikeout",
    "walk": "walk", "intent_walk": "intent_walk", "hit_by_pitch": "hbp",
    "catcher_interf": "catcher_interf", "truncated_pa": "truncated",
}

# Fold-independent exclusion reasons, in priority order (first match wins).
EXCLUSION_ORDER = [
    "position_player_pitching", "pitchout", "intentional_walk", "automatic_call", "missing_pitch_type",
    "not_an_action", "sequence_error",
]


def read_raw(path=config.RAW_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path, columns=RAW_COLUMNS)


def build_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["game_type"] == "R"].copy()
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable")
    df = df.reset_index(drop=True)

    out = pd.DataFrame({
        "season": df["game_year"].astype("int16"),
        "game_pk": df["game_pk"].astype("int64"),
        "game_date": df["game_date"],
        "inning": df["inning"].astype("int16"),
        "inning_topbot": df["inning_topbot"],
        "at_bat_number": df["at_bat_number"].astype("int16"),
        "pitch_number": df["pitch_number"].astype("int16"),
    })
    out["pa_id"] = out["game_pk"] * 1000 + out["at_bat_number"]
    top = df["inning_topbot"].eq("Top")
    out["fielding_team"] = np.where(top, df["home_team"], df["away_team"])
    out["batting_team"] = np.where(top, df["away_team"], df["home_team"])
    out["batter"] = df["batter"].astype("int64")
    out["stand"] = df["stand"]
    out["pitcher"] = df["pitcher"].astype("int64")
    out["p_throws"] = df["p_throws"]
    out["catcher"] = df["fielder_2"].astype("int64")

    out["balls"] = df["balls"].astype("int8")
    out["strikes"] = df["strikes"].astype("int8")
    out["count"] = out["balls"].astype(str) + "-" + out["strikes"].astype(str)
    out["outs"] = df["outs_when_up"].astype("int8")
    on1, on2, on3 = (df[c].notna() for c in ("on_1b", "on_2b", "on_3b"))
    out["bases"] = (on1 * 1 + on2 * 2 + on3 * 4).astype("int8")  # bitmask: 1st=1, 2nd=2, 3rd=4
    out["base_out"] = (out["bases"] * 3 + out["outs"]).astype("int8")  # 0..23
    out["runners_on"] = out["bases"] > 0

    out["pitch_type"] = df["pitch_type"]
    out["family"] = df["pitch_type"].map(config.PITCH_FAMILIES)
    out["description"] = df["description"]
    out["events"] = df["events"]
    out["zone"] = df["zone"]
    out["is_pitchout"] = df["description"].eq("pitchout") | df["pitch_type"].eq("PO")
    out["is_ibb"] = df["events"].eq("intent_walk") & df["description"].eq("automatic_ball")
    out["is_automatic"] = df["description"].isin(AUTOMATIC) & ~out["is_ibb"]
    out["thrown"] = ~df["description"].isin(AUTOMATIC)
    # Pitches whose recorded result doesn't follow the ordinary outcome transitions.
    # Kept for sequencing (and scoring, which never uses the realized outcome), but
    # not for fitting outcome or contact models.
    out["exceptional_event"] = pd.Series(pd.NA, index=out.index, dtype="object").mask(
        df["events"].eq("catcher_interf"), "catcher_interference")

    outcome = df["description"].map(OUTCOME)
    two_strike_foul_bunt = df["description"].eq("foul_bunt") & out["strikes"].eq(2)
    outcome = outcome.mask(two_strike_foul_bunt, "swinging_strike")
    out["outcome"] = outcome.where(out["thrown"])

    in_play = df["description"].eq("hit_into_play")
    bunt = in_play & df["des"].str.contains(r"\bbunt", case=False, na=False)
    contact = df["launch_speed_angle"].where(in_play)
    contact = contact.mask(bunt, BUNT_BIN)
    contact = contact.mask(in_play & contact.isna(), UNTRACKED_BIN)
    out["contact_bin"] = contact.astype("Int8")
    out["xwoba"] = df["estimated_woba_using_speedangle"]

    out["bat_score"] = df["bat_score"].astype("int16")
    out["post_bat_score"] = df["post_bat_score"].astype("int16")
    out["fld_score"] = df["fld_score"].astype("int16")
    out["post_fld_score"] = df["post_fld_score"].astype("int16")

    _add_previous_pitch(out)
    _add_runner_events(out)
    _add_sequence_check(out)
    _add_dugout_flag(out)
    _add_position_player_flag(out)
    set_action_unit(out, config.ACTION_UNIT)
    return out


def set_action_unit(out: pd.DataFrame, unit: str) -> None:
    """Set `action` and `prev_action` to pitch types or families, and the exclusion reasons.

    Pitch types in config.NON_ACTION_PITCH_TYPES, and types without a family when
    the unit is families, are excluded as `not_an_action`.
    """
    if unit == "pitch_type":
        out["action"] = out["pitch_type"].mask(out["pitch_type"].isin(config.NON_ACTION_PITCH_TYPES))
        out["prev_action"] = out["prev_pitch_type"].mask(out["prev_pitch_type"].isin(config.NON_ACTION_PITCH_TYPES))
    elif unit == "family":
        out["action"] = out["family"]
        out["prev_action"] = out["prev_family"]
    else:
        raise ValueError(f"unknown action unit {unit!r}")
    _add_exclusion_reason(out)
    _add_fit_eligibility(out)


def _add_fit_eligibility(out: pd.DataFrame) -> None:
    """Which pitches may train the pitch-outcome and contact models (before fold roles).

    Eligible: a thrown pitch with an outcome and an action, not a pitchout, not by a
    position player, not in a PA with a sequence error, and with no exceptional event.
    The contact model also needs a ball in play. Fold roles decide which eligible
    pitches each model sees.
    """
    out["outcome_fit_eligible"] = (
        out["thrown"] & out["outcome"].notna() & out["action"].notna() & ~out["is_pitchout"]
        & ~out["position_player_pitching"] & ~out["pa_sequence_error"] & out["exceptional_event"].isna()
    )
    out["contact_fit_eligible"] = out["outcome_fit_eligible"] & out["outcome"].eq("in_play")


def _add_previous_pitch(out: pd.DataFrame) -> None:
    """Previous pitch = the last pitch actually thrown earlier in the same PA."""
    pos = pd.Series(np.arange(len(out), dtype="float64"), index=out.index)
    thrown_pos = pos.where(out["thrown"])
    first_in_pa = out["pa_id"].ne(out["pa_id"].shift())
    prev_pos = thrown_pos.shift(1).mask(first_in_pa)
    prev_pos = prev_pos.groupby(out["pa_id"]).ffill()
    has_prev = prev_pos.notna()
    idx = prev_pos[has_prev].astype("int64").to_numpy()

    for col, src in (("prev_pitch_type", "pitch_type"), ("prev_family", "family")):
        vals = pd.Series(pd.NA, index=out.index, dtype="object")
        vals[has_prev] = out[src].to_numpy()[idx]
        out[col] = vals
    prev_desc = pd.Series(pd.NA, index=out.index, dtype="object")
    prev_desc[has_prev] = out["description"].to_numpy()[idx]
    out["prev_result"] = prev_desc.map(PREV_RESULT)
    out["has_prev_pitch"] = has_prev


def _add_runner_events(out: pd.DataFrame) -> None:
    """Flag rows whose base-out state or score changed since the previous row of the PA."""
    same_pa = out["pa_id"].eq(out["pa_id"].shift())
    out["outs_added_before"] = (out["outs"] - out["outs"].shift()).where(same_pa, 0).astype("int8")
    out["bases_changed_before"] = same_pa & out["bases"].ne(out["bases"].shift())
    out["runs_before"] = (out["bat_score"] - out["bat_score"].shift()).where(same_pa, 0).astype("int16")
    out["runner_event_before"] = (
        out["bases_changed_before"] | out["outs_added_before"].ne(0) | out["runs_before"].ne(0)
    )


def _add_sequence_check(out: pd.DataFrame) -> None:
    """Flag every row of a PA whose recorded counts don't follow from its descriptions."""
    same_pa = out["pa_id"].eq(out["pa_id"].shift())
    prev_desc = out["description"].shift()
    prev_b, prev_s = out["balls"].shift(), out["strikes"].shift()
    exp_b = np.where(prev_desc.isin(BALL_DESCRIPTIONS), prev_b + 1, prev_b)
    exp_s = np.where(
        prev_desc.isin(STRIKE_DESCRIPTIONS), prev_s + 1,
        np.where(prev_desc.eq("foul"), np.minimum(prev_s + 1, 2), prev_s),
    )
    bad = same_pa & (out["balls"].ne(exp_b) | out["strikes"].ne(exp_s))
    bad |= ~same_pa & (out["balls"].ne(0) | out["strikes"].ne(0))
    out["pa_sequence_error"] = bad.groupby(out["pa_id"]).transform("any")


def _add_dugout_flag(out: pd.DataFrame) -> None:
    out["dugout_called"] = (
        out["season"].eq(2025)
        & out["fielding_team"].eq(config.DUGOUT_CALLED_TEAM)
        & (out["game_date"] >= config.DUGOUT_CALLED_START)
    )


def _add_position_player_flag(out: pd.DataFrame, min_bat_pa=config.POSITION_PLAYER_MIN_BAT_PA) -> None:
    """Flag pitches thrown by position players (see config.POSITION_PLAYER_MIN_BAT_PA)."""
    bat_pa = out.drop_duplicates("pa_id")["batter"].value_counts()
    starters = out.groupby(["game_pk", "fielding_team"]).head(1)
    started = set(zip(starters["season"], starters["pitcher"]))
    batted = out["pitcher"].map(bat_pa).fillna(0) >= min_bat_pa
    no_start = ~pd.Series([k in started for k in zip(out["season"], out["pitcher"])], index=out.index)
    out["position_player_pitching"] = batted & no_start


def _add_exclusion_reason(out: pd.DataFrame) -> None:
    conditions = {
        "position_player_pitching": out["position_player_pitching"],
        "pitchout": out["is_pitchout"],
        "intentional_walk": out["is_ibb"],
        "automatic_call": out["is_automatic"],
        "missing_pitch_type": out["pitch_type"].isna(),
        "not_an_action": out["action"].isna(),
        "sequence_error": out["pa_sequence_error"],
    }
    reason = pd.Series(pd.NA, index=out.index, dtype="object")
    for name in reversed(EXCLUSION_ORDER):
        reason = reason.mask(conditions[name], name)
    out["exclusion_reason"] = reason


def build_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per PA, with explicitly named start, final-pitch and next states.

    - `start_*`: the state at the PA's first pitch (for a PA-start RE24 table).
    - `final_*`: the state immediately before the final pitch (for contact-bin and
      other terminal payoffs, e.g. after a steal or a pitching change mid-PA).
    - `next_*`: the start state of the next PA in the same half-inning, when one exists.
    - `pa_end_type`: "continues" (another PA follows in the half-inning),
      "three_outs", "walk_off" (game ends in the bottom half before three outs),
      or "game_end_other" (a game called before three outs, before the 9th).
      The last two are `censored`: the next state is unknown, not "inning over".
    """
    g = pitches.groupby("pa_id", sort=False)
    first, last = g.head(1).set_index("pa_id"), g.tail(1).set_index("pa_id")
    pa = first[[
        "season", "game_pk", "game_date", "inning", "inning_topbot", "at_bat_number",
        "fielding_team", "batting_team", "batter", "stand",
    ]].copy()
    for col in ("pitcher", "catcher", "bases", "outs", "base_out", "bat_score"):
        pa[f"start_{col}"] = first[col]
    for col in ("pitcher", "catcher", "bases", "outs", "base_out", "balls", "strikes",
                "bat_score", "fld_score"):
        pa[f"final_{col}"] = last[col]
    pa["n_rows"] = g.size()
    pa["n_thrown"] = g["thrown"].sum()
    pa["events"] = last["events"]
    pa["final_description"] = last["description"]
    pa["final_contact_bin"] = last["contact_bin"]
    end = last["events"].map(PA_END)
    end = end.mask(end.isna() & last["description"].eq("hit_into_play"), "in_play")
    pa["pa_end"] = end.fillna("no_event")
    pa["runs_on_final_pitch"] = (last["post_bat_score"] - last["bat_score"]).astype("int16")
    pa["final_post_bat_score"] = last["post_bat_score"]
    pa["final_post_fld_score"] = last["post_fld_score"]
    pa["catcher_changed"] = g["catcher"].nunique() > 1
    pa["pitcher_changed"] = g["pitcher"].nunique() > 1
    pa["pa_sequence_error"] = first["pa_sequence_error"]

    pa = pa.reset_index()
    half = ["game_pk", "inning", "inning_topbot"]
    nxt = pa.groupby(half, sort=False)[["start_bases", "start_outs"]].shift(-1)
    pa["next_bases"] = nxt["start_bases"].astype("Int8")
    pa["next_outs"] = nxt["start_outs"].astype("Int8")
    half_end_score = (
        pitches.groupby(half, sort=False)["post_bat_score"].last().rename("half_end_score")
    )
    pa = pa.join(half_end_score, on=half)
    pa["runs_rest_of_inning"] = (pa["half_end_score"] - pa["start_bat_score"]).astype("int16")
    last_half = pa.groupby("game_pk")[["inning", "inning_topbot"]].transform("last")
    pa["final_half_inning"] = (pa["inning"] == last_half["inning"]) & (
        pa["inning_topbot"] == last_half["inning_topbot"]
    )
    _add_pa_end_type(pa, has_next=nxt["start_outs"].notna())
    return pa.drop(columns="half_end_score")


# Outs recorded by the batter's final play. Runner outs on hits aren't recorded,
# so this undercounts; it is used only for final half-innings before the 9th.
OUTS_ON_PLAY = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "fielders_choice_out": 1, "truncated_pa": 1, "grounded_into_double_play": 2,
    "double_play": 2, "strikeout_double_play": 2, "sac_fly_double_play": 2,
    "sac_bunt_double_play": 2, "triple_play": 3,
}


def _add_pa_end_type(pa: pd.DataFrame, has_next: pd.Series) -> None:
    """Classify how each PA's half-inning continues; see build_plate_appearances.

    Outside the game's final half-inning, a PA with no next PA ended the half-inning
    with three outs. In the final half-inning, a bottom half that ends with the home
    team ahead is a walk-off. Otherwise, from the 9th on, a regulation game can only
    end on the third out; before the 9th (a called game), the outs on the play decide.
    """
    end_type = pd.Series("continues", index=pa.index, dtype="object")
    last_in_half = ~has_next.to_numpy()
    walk_off = (pa["inning_topbot"].eq("Bot") & (pa["final_post_bat_score"] > pa["final_post_fld_score"]))
    outs_after = pa["final_outs"] + pa["events"].map(OUTS_ON_PLAY).fillna(0)
    final = pa["final_half_inning"]
    end_type[last_in_half & ~final] = "three_outs"
    end_type[last_in_half & final & walk_off] = "walk_off"
    three = (pa["inning"] >= 9) | (outs_after >= 3)
    end_type[last_in_half & final & ~walk_off & three] = "three_outs"
    end_type[last_in_half & final & ~walk_off & ~three] = "game_end_other"
    pa["pa_end_type"] = end_type
    pa["censored"] = end_type.isin(["walk_off", "game_end_other"])
    pa["inning_over_after"] = end_type.eq("three_outs")
    half = ["game_pk", "inning", "inning_topbot"]
    half_end = pa.groupby(half, sort=False)["pa_end_type"].transform("last")
    pa["half_inning_complete"] = half_end.eq("three_outs")
    # Uses of these fields are set out in project_plan.md (Payoff).
    pa["re24_eligible"] = pa["half_inning_complete"] & (pa["inning"] <= 8)
    pa["payoff_eligible"] = ~pa["censored"] & ~pa["pa_sequence_error"]


def load(build: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not build and config.PITCHES_PARQUET.exists() and config.PAS_PARQUET.exists():
        return pd.read_parquet(config.PITCHES_PARQUET), pd.read_parquet(config.PAS_PARQUET)
    pitches = build_pitches(read_raw())
    return pitches, build_plate_appearances(pitches)
