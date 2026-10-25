import numpy as np
import pandas as pd
import pytest

from gamecalling import arsenal, config, loader


def raw_pa(rows, game_pk=1, ab=1, **pa_fields):
    """Raw Statcast-like rows for one PA. Each row: (balls, strikes, description, pitch_type, extras)."""
    base = dict(
        game_pk=game_pk, game_date="2024-05-01", game_type="R", game_year=2024, inning=1,
        inning_topbot="Top", at_bat_number=ab, home_team="SEA", away_team="TEX",
        batter=10, stand="R", pitcher=20, p_throws="R", fielder_2=30, outs_when_up=0,
        on_1b=np.nan, on_2b=np.nan, on_3b=np.nan, events=None, des=None, bb_type=None,
        launch_speed_angle=np.nan, zone=5.0, bat_score=0, post_bat_score=0, fld_score=0, post_fld_score=0,
        estimated_woba_using_speedangle=np.nan,
    )
    base.update(pa_fields)
    out = []
    for i, (b, s, desc, pt, *extra) in enumerate(rows, start=1):
        row = dict(base, pitch_number=i, balls=b, strikes=s, description=desc, pitch_type=pt)
        if extra:
            row.update(extra[0])
        out.append(row)
    return pd.DataFrame(out)[loader.RAW_COLUMNS]


def test_previous_pitch_skips_automatic_calls():
    raw = raw_pa([
        (0, 0, "called_strike", "FF"),
        (0, 1, "automatic_ball", None),
        (1, 1, "foul", "SL"),
        (1, 2, "swinging_strike", "CH", {"events": "strikeout"}),
    ])
    p = loader.build_pitches(raw)
    assert pd.isna(p.loc[0, "prev_pitch_type"]) and not p.loc[0, "has_prev_pitch"]
    assert p.loc[1, "prev_pitch_type"] == "FF"
    assert p.loc[2, "prev_pitch_type"] == "FF" and p.loc[2, "prev_result"] == "called_strike"
    assert p.loc[3, "prev_pitch_type"] == "SL" and p.loc[3, "prev_result"] == "foul"
    assert p.loc[1, "exclusion_reason"] == "automatic_call"
    assert pd.isna(p.loc[1, "outcome"])
    assert not p["pa_sequence_error"].any()


def test_outcomes_and_two_strike_foul_bunt():
    raw = raw_pa([
        (0, 0, "foul_bunt", "FF"),
        (0, 1, "foul_tip", "FF"),
        (0, 2, "foul_bunt", "FF", {"events": "strikeout"}),
    ])
    p = loader.build_pitches(raw)
    assert p["outcome"].tolist() == ["foul", "swinging_strike", "swinging_strike"]
    assert p["prev_result"].tolist()[1:] == ["foul", "swinging_strike"]


def test_contact_bins():
    raw = pd.concat([
        raw_pa([(0, 0, "hit_into_play", "FF", {"launch_speed_angle": 6.0, "events": "home_run"})], ab=1),
        raw_pa([(0, 0, "hit_into_play", "FF", {"launch_speed_angle": 2.0, "events": "sac_bunt",
                                               "des": "Player out on a sacrifice bunt."})], ab=2),
        raw_pa([(0, 0, "hit_into_play", "FF", {"events": "field_out"})], ab=3),
        raw_pa([(0, 0, "ball", "FF"), (1, 0, "hit_by_pitch", "FF", {"events": "hit_by_pitch"})], ab=4),
    ])
    p = loader.build_pitches(raw)
    assert p["contact_bin"].tolist()[:3] == [6, loader.BUNT_BIN, loader.UNTRACKED_BIN]
    assert p["contact_bin"].isna().tolist()[3:] == [True, True]


def test_base_out_and_runner_events():
    raw = raw_pa([
        (0, 0, "ball", "FF", {"on_1b": 5.0, "outs_when_up": 1}),
        (1, 0, "ball", "FF", {"on_2b": 5.0, "outs_when_up": 1}),  # stolen base
        (2, 0, "ball", "FF", {"outs_when_up": 2}),  # caught stealing third
    ])
    p = loader.build_pitches(raw)
    assert p["base_out"].tolist() == [1 * 3 + 1, 2 * 3 + 1, 0 * 3 + 2]
    assert p["runner_event_before"].tolist() == [False, True, True]
    assert p["outs_added_before"].tolist() == [0, 0, 1]


def test_sequence_error_flags_whole_pa():
    raw = raw_pa([(0, 0, "ball", "FF"), (2, 0, "ball", "FF")])
    p = loader.build_pitches(raw)
    assert p["pa_sequence_error"].all()
    assert (p["exclusion_reason"] == "sequence_error").all()


def test_pitchout_ibb_and_action_units():
    raw = pd.concat([
        raw_pa([(0, 0, "pitchout", "PO"), (1, 0, "ball", "KN"), (2, 0, "ball", "FA"), (3, 0, "ball", "SI")], ab=1),
        raw_pa([(0, 0, "automatic_ball", None, {"events": "intent_walk"})], ab=2),
    ])
    p = loader.build_pitches(raw)
    assert config.ACTION_UNIT == "pitch_type"
    # Pitch types: a knuckleball is an action, FA ("other") is not.
    assert p.loc[1, "action"] == "KN" and pd.isna(p.loc[2, "action"]) and p.loc[3, "action"] == "SI"
    assert p["exclusion_reason"].fillna("ok").tolist() == ["pitchout", "ok", "not_an_action", "ok", "intentional_walk"]
    assert pd.isna(p.loc[1, "prev_action"]) and p.loc[2, "prev_action"] == "KN"
    assert p.loc[1, "prev_result"] == "ball"
    # Families: KN has no family; the sinker is a fastball.
    loader.set_action_unit(p, "family")
    assert p["exclusion_reason"].fillna("ok").tolist() == [
        "pitchout", "not_an_action", "not_an_action", "ok", "intentional_walk"]
    assert p.loc[3, "action"] == "fastball"


def test_position_players_excluded_two_way_kept():
    # Players 99 and 98 each bat 5 times. 99 pitches in relief (position player);
    # 98 starts game 2 (two-way player, kept). Pitcher 20 starts game 1 and never bats.
    pas = [raw_pa([(0, 0, "ball", "FF")], game_pk=1, ab=1, pitcher=20)]
    pas += [raw_pa([(0, 0, "ball", "FF")], game_pk=1, ab=ab, batter=b, pitcher=20)
            for ab, b in zip(range(2, 12), [99] * 5 + [98] * 5)]
    pas += [raw_pa([(0, 0, "ball", "EP")], game_pk=1, ab=12, pitcher=99)]
    pas += [raw_pa([(0, 0, "ball", "FF")], game_pk=2, ab=1, pitcher=98)]
    p = loader.build_pitches(pd.concat(pas))
    flagged = p.groupby("pitcher")["position_player_pitching"].any().to_dict()
    assert flagged == {20: False, 98: False, 99: True}
    assert (p.loc[p.pitcher == 99, "exclusion_reason"] == "position_player_pitching").all()


def pa_table(*pas):
    return loader.build_plate_appearances(loader.build_pitches(pd.concat(pas))).set_index("at_bat_number")


def test_steal_before_contact_uses_final_state():
    # Runner on first steals second, then the batter singles; the next PA starts with runners on 1st and 3rd.
    pa = pa_table(
        raw_pa([(0, 0, "ball", "FF", {"on_1b": 5.0}),
                (1, 0, "hit_into_play", "FF", {"on_2b": 5.0, "events": "single", "launch_speed_angle": 4.0})], ab=1),
        raw_pa([(0, 0, "ball", "FF", {"on_1b": 6.0, "on_3b": 5.0})], ab=2),
    )
    assert pa.loc[1, "start_base_out"] == 1 * 3 and pa.loc[1, "final_base_out"] == 2 * 3
    assert (pa.loc[1, "next_bases"], pa.loc[1, "next_outs"]) == (5, 0)
    assert pa.loc[1, "final_balls"] == 1 and pa.loc[1, "pa_end_type"] == "continues"


def test_mid_pa_pitching_change_uses_final_pitcher():
    pa = pa_table(raw_pa([
        (0, 0, "ball", "FF", {"pitcher": 20}),
        (1, 0, "called_strike", "SL", {"pitcher": 21}),
        (1, 1, "swinging_strike", "SL", {"pitcher": 21}),
        (1, 2, "swinging_strike", "SL", {"pitcher": 21, "events": "strikeout"}),
    ]))
    assert (pa.loc[1, "start_pitcher"], pa.loc[1, "final_pitcher"]) == (20, 21)
    assert pa.loc[1, "pitcher_changed"]


def test_walk_off_is_censored_not_inning_over():
    top = dict(inning=9, inning_topbot="Top", outs_when_up=2, bat_score=3, post_bat_score=3, fld_score=3, post_fld_score=3)
    bot = dict(inning=9, inning_topbot="Bot", outs_when_up=1, on_2b=5.0, bat_score=3, fld_score=3)
    pa = pa_table(
        raw_pa([(0, 0, "swinging_strike", "FF"), (0, 1, "swinging_strike", "FF"),
                (0, 2, "swinging_strike", "FF", {"events": "strikeout"})], ab=1, **top),
        raw_pa([(0, 0, "hit_into_play", "FF", {"events": "single", "post_bat_score": 4})], ab=2, **bot),
    )
    assert pa.loc[1, "pa_end_type"] == "three_outs" and pa.loc[1, "inning_over_after"]
    assert pa.loc[2, "pa_end_type"] == "walk_off" and pa.loc[2, "censored"]
    assert not pa.loc[2, "inning_over_after"] and pd.isna(pa.loc[2, "next_outs"])
    assert not pa.loc[2, "half_inning_complete"] and not pa.loc[2, "payoff_eligible"]
    assert pa.loc[2, "runs_on_final_pitch"] == 1


def test_game_final_half_inning_endings():
    # Game 1 ends on a third out in the top of the 9th (home team ahead): three outs, not censored.
    # Game 2 stops after a 0-out single in the top of the 6th (called game): censored.
    # Game 3 ends on a 2-out single in the top of the 9th (runner thrown out): three outs.
    lead = dict(inning=9, outs_when_up=2, bat_score=2, post_bat_score=2, fld_score=5, post_fld_score=5)
    pa = loader.build_plate_appearances(loader.build_pitches(pd.concat([
        raw_pa([(0, 0, "hit_into_play", "FF", {"events": "field_out"})], game_pk=1, ab=70, **lead),
        raw_pa([(0, 0, "hit_into_play", "FF", {"events": "single"})], game_pk=2, ab=40, inning=6),
        raw_pa([(0, 0, "hit_into_play", "FF", {"events": "single"})], game_pk=3, ab=70, **lead),
    ]))).set_index("game_pk")
    assert pa.loc[1, "pa_end_type"] == "three_outs" and not pa.loc[1, "censored"]
    assert pa.loc[2, "pa_end_type"] == "game_end_other" and pa.loc[2, "censored"]
    assert pa.loc[3, "pa_end_type"] == "three_outs"


def test_catcher_interference_kept_but_not_fit():
    p = loader.build_pitches(raw_pa([
        (0, 0, "ball", "FF"),
        (1, 0, "foul", "SL", {"events": "catcher_interf"}),
    ]))
    assert p["exceptional_event"].isna().tolist() == [True, False]
    assert p.loc[1, "exceptional_event"] == "catcher_interference"
    assert p["outcome_fit_eligible"].tolist() == [True, False]
    assert p["exclusion_reason"].isna().all()  # the decision can still be scored
    pa = loader.build_plate_appearances(p)
    assert pa.loc[0, "pa_end"] == "catcher_interf"


def test_fold_roles_and_assignment():
    assert [arsenal.fold_roles(k) for k in range(3)] == [(1, 2), (2, 0), (0, 1)]
    games = pd.DataFrame({"season": [2024] * 9 + [2025] * 6, "game_pk": range(15)})
    a, b = arsenal.assign_folds(games), arsenal.assign_folds(games)
    assert a.equals(b)
    assert a.groupby(games.set_index("game_pk")["season"]).value_counts().max() == 3


RULES = {"SP": {"min_share": 0.02, "min_count": 10}, "RP": {"min_share": 0.05, "min_count": 5}}


def usage(counts, season=2024, pitcher=1):
    """Thrown pitches from {(fold, stand): {action: n}}."""
    rows = []
    for (fold, stand), fams in counts.items():
        for fam, n in fams.items():
            rows += [dict(season=season, pitcher=pitcher, fold=fold, stand=stand, action=fam,
                          thrown=True, exclusion_reason=pd.NA)] * n
    return pd.DataFrame(rows)


def starter_roles(folds=(0, 1, 2), pitcher=1, starts=5):
    return pd.DataFrame({"season": 2024, "pitcher": pitcher, "fold": list(folds), "games": 5, "starts": starts})


def eligible(p, roles, rules=RULES):
    elig = arsenal.eligible_sets(arsenal.support_table(p, rules, roles))
    return {key: set(g["action"]) for key, g in elig.groupby(["scoring_fold", "stand"])}


def test_support_uses_two_training_folds_combined():
    # Slider dips to 1 of 300 in fold 1, but every pair of training folds clears 2% and 10 pitches.
    counts = {(0, "R"): {"fastball": 270, "slider": 30}, (1, "R"): {"fastball": 299, "slider": 1},
              (2, "R"): {"fastball": 270, "slider": 30}}
    got = eligible(usage(counts), starter_roles())
    assert all(got[(k, "R")] == {"fastball", "slider"} for k in range(3))


def test_scoring_fold_is_never_used():
    # Curveballs only in fold 0: eligible when fold 0 is a training fold, not when it is scored.
    counts = {(0, "R"): {"fastball": 200, "curveball": 100}, (1, "R"): {"fastball": 300},
              (2, "R"): {"fastball": 300}}
    got = eligible(usage(counts), starter_roles())
    assert got[(0, "R")] == {"fastball"}
    assert got[(1, "R")] == got[(2, "R")] == {"fastball", "curveball"}


def test_support_by_hand():
    # Sinker thrown only to righties: eligible vs R, not vs L, in every scoring fold.
    counts = {}
    for fold in range(3):
        counts[(fold, "R")] = {"fastball": 80, "sinker": 20}
        counts[(fold, "L")] = {"fastball": 100}
    got = eligible(usage(counts), starter_roles())
    for k in range(3):
        assert got[(k, "R")] == {"fastball", "sinker"} and got[(k, "L")] == {"fastball"}


def test_role_specific_cutoffs():
    # Curveball: 6 of 200 per fold, so 12 of 400 (3%) on two training folds combined.
    # SP (2%, 10) passes; RP needs 5% and fails. With 4 per fold (8 of 400, 2%) SP fails the count.
    for per_fold, role_starts, expected in ((6, 5, True), (6, 0, False), (4, 5, False)):
        counts = {(f, "R"): {"fastball": 200 - per_fold, "curveball": per_fold} for f in range(3)}
        got = eligible(usage(counts), starter_roles(starts=role_starts))
        assert ("curveball" in got[(0, "R")]) == expected


def test_training_roles_combine_two_folds():
    # Pitcher 1 starts both fold-0 games and relieves in fold 1 (2 of 3 starts): SP when fold 2 is scored.
    # Pitcher 3 relieves in fold 0 and starts in fold 1 (1 of 2, at the 50% cutoff): SP.
    rows = []
    for game, fold, order in ((1, 0, [1, 2]), (2, 0, [1, 3]), (3, 1, [3, 1])):
        for pitcher in order:
            rows.append(dict(season=2024, fold=fold, game_pk=game, fielding_team="SEA", pitcher=pitcher))
    tr = arsenal.training_roles(arsenal.pitcher_roles(pd.DataFrame(rows)))
    assert set(tr["scoring_fold"]) == {2}  # nobody pitched in fold 2, so only fold 2 has both training folds
    roles = tr.set_index("pitcher")["role"]
    assert roles[1] == "SP" and roles[3] == "SP"
    assert 2 not in roles.index  # pitcher 2 pitched only in fold 0


def test_pitcher_missing_from_training_fold():
    # Games only in folds 0 and 1: every pitch lacks one training fold.
    counts = {(0, "R"): {"fastball": 100}, (1, "R"): {"fastball": 100}}
    p = usage(counts)
    marked = arsenal.mark_arsenal(p, arsenal.eligible_sets(arsenal.support_table(p, RULES, starter_roles((0, 1)))))
    assert not marked["scorable"].any()
    assert (marked["arsenal_reason"] == "pitcher_missing_from_role_fold").all()


data_missing = not config.PITCHES_PARQUET.exists()


@pytest.mark.skipif(data_missing, reason="run python3.12 -m gamecalling.phase0 first")
def test_real_data_accounting():
    p = pd.read_parquet(config.PITCHES_PARQUET)
    assert len(p) == 1_423_795
    assert p.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum() == 0
    reason = p["exclusion_reason"].fillna(p["arsenal_reason"])
    assert (reason.isna() == p["scorable"]).all()
    assert p.loc[p["dugout_called"], "game_pk"].nunique() == config.DUGOUT_CALLED_EXPECTED_GAMES
    assert p["base_out"].between(0, 23).all()
    assert p.groupby("game_pk")["fold"].nunique().eq(1).all()
