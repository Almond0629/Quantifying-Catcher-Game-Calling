"""Coverage accounting: every regular-season pitch is either scorable or excluded for a reason."""

import pandas as pd

from gamecalling import arsenal, config, loader
from gamecalling.loader import EXCLUSION_ORDER

ARSENAL_REASONS = ["below_support", "pitcher_missing_from_role_fold"]


def final_reason(pitches: pd.DataFrame) -> pd.Series:
    return pitches["exclusion_reason"].fillna(pitches["arsenal_reason"])


def exclusion_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """Pitches by reason and season, with the all-pitches denominator."""
    reason = final_reason(pitches).fillna("scorable")
    table = pd.crosstab(reason, pitches["season"], margins=True, margins_name="all")
    order = ["scorable"] + EXCLUSION_ORDER + ARSENAL_REASONS + ["all"]
    table = table.reindex([r for r in order if r in table.index])
    share = table / table.loc["all"]
    return table, share


def out_of_arsenal_by_pitcher(pitches: pd.DataFrame) -> pd.DataFrame:
    base = pitches[pitches["exclusion_reason"].isna()]
    g = base.groupby(["season", "pitcher"])
    df = pd.DataFrame({
        "candidate_pitches": g.size(),
        "out_of_arsenal": g["arsenal_reason"].count(),
        "below_support": g["arsenal_reason"].apply(lambda s: (s == "below_support").sum()),
    })
    df["share_out_of_arsenal"] = df["out_of_arsenal"] / df["candidate_pitches"]
    return df.sort_values(["season", "out_of_arsenal"], ascending=[True, False]).reset_index()


SENSITIVITY_CUTOFFS = {
    "SP": [(0.01, 0), (0.01, 10), (0.02, 0), (0.02, 10), (0.02, 15), (0.03, 0), (0.03, 10)],
    "RP": [(0.03, 5), (0.05, 0), (0.05, 5), (0.03, 10), (0.05, 10)],
}


def pitch_roles(pitches: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.Series:
    """Each pitch's role: its pitcher's training-fold role for the pitch's own (scoring) fold.

    Pitchers missing from a training fold have no role ("none").
    """
    if roles is None:
        roles = arsenal.pitcher_roles(pitches)
    tr = arsenal.training_roles(roles).rename(columns={"scoring_fold": "fold"})
    role = pitches[["season", "pitcher", "fold"]].merge(tr, how="left")["role"]
    return pd.Series(role.fillna("none").to_numpy(), index=pitches.index, name="role")


def arsenal_rule_sensitivity(pitches: pd.DataFrame) -> pd.DataFrame:
    """Vary one role's cutoffs with the other role held at its configured rule.

    `role pitches scorable` is out of that role's pitches.
    """
    roles = arsenal.pitcher_roles(pitches)
    role_of = pitch_roles(pitches, roles).to_numpy()
    rows = []
    for role, grid in SENSITIVITY_CUTOFFS.items():
        for min_share, min_count in grid:
            rules = {**config.ARSENAL_RULES, role: {"min_share": min_share, "min_count": min_count}}
            sup = arsenal.support_table(pitches, rules, roles)
            marked = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(sup))
            mine = role_of == role
            rows.append({
                "role": role,
                "rule": f"share >= {min_share:.0%} and n >= {min_count}",
                "configured": rules[role] == config.ARSENAL_RULES[role],
                "options passing": sup.loc[sup["role"] == role, "passes"].mean(),
                "role pitches scorable": marked.loc[mine, "scorable"].mean(),
                **{f"all scorable {s}": marked.loc[marked.season == s, "scorable"].mean() for s in config.SEASONS},
            })
    return pd.DataFrame(rows)


def role_summary(pitches: pd.DataFrame) -> pd.DataFrame:
    role = pitch_roles(pitches)
    by = pitches.groupby(role)
    return pd.DataFrame({
        "pitches": by.size(),
        "scorable": by["scorable"].mean(),
        "below_support": by["arsenal_reason"].apply(lambda s: (s == "below_support").mean()),
        "pitcher_missing_from_role_fold": by["arsenal_reason"].apply(
            lambda s: (s == "pitcher_missing_from_role_fold").mean()),
    })


def eligible_options(pitches: pd.DataFrame) -> pd.DataFrame:
    """Mean eligible actions per (pitcher, batter hand, scoring fold), by role and hand."""
    roles = arsenal.pitcher_roles(pitches)
    el = arsenal.eligible_sets(arsenal.support_table(pitches, roles=roles))
    el = el.merge(arsenal.training_roles(roles)[["season", "pitcher", "scoring_fold", "role"]])
    sizes = el.groupby(["season", "pitcher", "stand", "scoring_fold", "role"]).size().rename("options")
    return sizes.reset_index().pivot_table(index="role", columns="stand", values="options", aggfunc="mean")


def unit_sensitivity(pitches: pd.DataFrame) -> pd.DataFrame:
    """Coverage and option counts with pitch types vs. families as the action unit."""
    rows = {}
    for unit in ("pitch_type", "family"):
        p = pitches.copy()
        loader.set_action_unit(p, unit)
        p = arsenal.mark_arsenal(p, arsenal.eligible_sets(arsenal.support_table(p)))
        opts = eligible_options(p)
        rows[unit + (" (main)" if unit == config.ACTION_UNIT else "")] = {
            **{f"scorable {s}": p.loc[p.season == s, "scorable"].mean() for s in config.SEASONS},
            "below_support": p["arsenal_reason"].eq("below_support").mean(),
            "not_an_action": p["exclusion_reason"].eq("not_an_action").mean(),
            "SP options per hand": opts.loc["SP"].mean(),
            "RP options per hand": opts.loc["RP"].mean(),
        }
    return pd.DataFrame(rows).T


def runner_event_summary(pitches: pd.DataFrame) -> pd.DataFrame:
    within = pitches["pa_id"].eq(pitches["pa_id"].shift())
    rows = []
    for season, p in pitches.groupby("season"):
        w = within[p.index]
        ev = p["runner_event_before"]
        rows.append({
            "season": season,
            "pitches": len(p),
            "PAs": p["pa_id"].nunique(),
            "pitches after a runner event in the PA": int(ev.sum()),
            "share of non-first pitches": ev.sum() / w.sum(),
            "PAs with a runner event": p.loc[ev, "pa_id"].nunique(),
            "out added (CS, pickoff)": int((p["outs_added_before"] > 0).sum()),
            "bases changed, no out (SB, WP, PB, balk)": int((p["bases_changed_before"] & p["outs_added_before"].eq(0)).sum()),
            "runs scored between pitches": int(p["runs_before"].clip(lower=0).sum()),
        })
    return pd.DataFrame(rows).set_index("season").T


def _md(df: pd.DataFrame, fmt=None) -> str:
    df = df.copy()
    if fmt:
        df = df.map(fmt)
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(pitches: pd.DataFrame, pas: pd.DataFrame, out_dir=config.REPORTS_DIR) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, share = exclusion_table(pitches)
    by_pitcher = out_of_arsenal_by_pitcher(pitches)
    by_pitcher.to_csv(out_dir / "out_of_arsenal_by_pitcher.csv", index=False)
    counts.to_csv(out_dir / "coverage_counts.csv")
    sens = arsenal_rule_sensitivity(pitches)
    runners = runner_event_summary(pitches)

    scorable = share.loc["scorable"]
    dug = pitches[pitches["dugout_called"]]
    dug_games = dug["game_pk"].nunique()
    thrown = pitches[pitches["thrown"]]
    pa_end = pd.crosstab(pas["pa_end"], pas["season"], margins=True, margins_name="all")
    end_type = pd.crosstab(pas["pa_end_type"], pas["season"], margins=True, margins_name="all")
    bip = pas[pas["pa_end"] == "in_play"]
    fit = pd.DataFrame({
        "outcome_fit_eligible": pitches.groupby("season")["outcome_fit_eligible"].sum(),
        "contact_fit_eligible": pitches.groupby("season")["contact_fit_eligible"].sum(),
        "catcher_interference rows": pitches.groupby("season")["exceptional_event"].count(),
        "of which scorable": (pitches["scorable"] & pitches["exceptional_event"].notna()).groupby(pitches["season"]).sum(),
    }).T
    bins = pd.crosstab(pitches["contact_bin"], pitches["season"], normalize="columns")
    fam = pd.crosstab(thrown["family"].fillna("(none)"), thrown["season"], normalize="columns")
    act = pd.crosstab(thrown["action"].fillna("(none)"), thrown["season"], normalize="columns")
    units = unit_sensitivity(pitches)
    prev = pd.crosstab(thrown["prev_result"].fillna("(first pitch of PA)"), thrown["season"], normalize="columns")
    worst = by_pitcher[by_pitcher["candidate_pitches"] >= 300].nlargest(10, "share_out_of_arsenal")
    present = thrown[thrown["action"].notna() & ~thrown["position_player_pitching"]].groupby(["season", "pitcher"]).agg(
        pitches=("game_pk", "size"), games=("game_pk", "nunique"), folds=("fold", "nunique"))
    fold_presence = present.groupby("folds").agg(
        pitcher_seasons=("pitches", "size"), pitches=("pitches", "sum"),
        median_games=("games", "median"), max_pitches=("pitches", "max"))

    pct = lambda x: f"{x:.2%}"
    num = lambda x: f"{x:,}"
    lines = [
        "# Phase 0 coverage report",
        "",
        "Generated by `python3.12 -m gamecalling.phase0`. Denominator: all regular-season Statcast pitch rows.",
        "",
        f"Action unit: `{config.ACTION_UNIT}` (families are a sensitivity check). "
        "Arsenal rule: judged per batter hand on the two training folds (selection + valuation) of "
        "each scoring fold combined; the pitcher must pitch in both. "
        + "; ".join(f"{r}: share >= {c['min_share']:.0%} and n >= {c['min_count']}"
                    for r, c in config.ARSENAL_RULES.items())
        + f", against that hand. {config.N_FOLDS} game folds per season, seed {config.FOLD_SEED}. "
        f"SP = at least {config.SP_MIN_START_SHARE:.0%} of the pitcher's appearances in the two "
        "training folds are starts.",
        "",
        "## Go/no-go criterion 1: coverage",
        "",
        "| Season | Scorable | Threshold | Result |",
        "|---|---|---|---|",
        *[f"| {s} | {pct(scorable[s])} | 95% | {'pass' if scorable[s] >= 0.95 else 'FAIL'} |" for s in share.columns],
        "",
        "## Pitches by reason",
        "",
        "Reasons are mutually exclusive; the first matching reason in this order wins.",
        "",
        _md(counts, num),
        "",
        _md(share, pct),
        "",
        "## Starters and relievers",
        "",
        "Role is the pitcher's role in the two training folds of the pitch's scoring fold; "
        "`none` = pitcher missing from a training fold.",
        "",
        _md(role_summary(pitches), lambda x: pct(x) if isinstance(x, float) and x < 1 else f"{x:,.0f}"),
        "",
        "Mean eligible actions per pitcher, batter hand and scoring fold:",
        "",
        _md(eligible_options(pitches), lambda x: f"{x:.2f}"),
        "",
        "## Action unit sensitivity: pitch types vs. families",
        "",
        "Same arsenal rule; only the action unit changes.",
        "",
        _md(units, lambda x: pct(x) if x < 1 else f"{x:.2f}"),
        "",
        "## Arsenal rule sensitivity (cutoffs still open)",
        "",
        "One role's cutoffs vary while the other role stays at its configured rule. "
        "`role pitches scorable` is out of that role's pitches.",
        "",
        _md(sens.set_index(["role", "rule"]), lambda x: x if isinstance(x, bool) else pct(x)),
        "",
        "## Pitchers missing from a fold",
        "",
        "A pitcher-season with games in fewer than three folds lacks a training fold "
        "for every game it has, so all of its pitches are `pitcher_missing_from_role_fold`.",
        "",
        _md(fold_presence, lambda x: f"{x:,.0f}"),
        "",
        "## Highest out-of-arsenal shares (pitchers with at least 300 candidate pitches)",
        "",
        "Full table: `reports/out_of_arsenal_by_pitcher.csv`.",
        "",
        _md(worst.set_index(["season", "pitcher"]).assign(
            share_out_of_arsenal=worst["share_out_of_arsenal"].map(pct).to_numpy())),
        "",
        "## Runner events during at-bats",
        "",
        "Pitches whose base-out state or score changed since the previous pitch of the same PA. "
        "All pitches are kept; each pitch carries the base-out state when it was thrown.",
        "",
        _md(runners, lambda x: pct(x) if isinstance(x, float) and x < 1 else num(int(x))),
        "",
        "## Flags (not exclusions)",
        "",
        f"- Marlins dugout-called games (MIA fielding, 2025, from {config.DUGOUT_CALLED_START}): "
        f"{dug_games} games (expected {config.DUGOUT_CALLED_EXPECTED_GAMES}), {len(dug):,} pitch rows, "
        f"catchers {sorted(dug['catcher'].unique().tolist())}.",
        f"- PAs where the catcher changed mid-PA: {int(pas['catcher_changed'].sum())}; "
        f"pitcher changed mid-PA: {int(pas['pitcher_changed'].sum())}.",
        f"- PAs with a count-sequence error (excluded as `sequence_error`): {int(pas['pa_sequence_error'].sum())}.",
        "",
        "## How plate appearances end",
        "",
        _md(pa_end, num),
        "",
        f"`no_event` PAs end without a Statcast event; {pas.loc[pas.pa_end == 'no_event', 'inning_over_after'].mean():.0%} "
        "of them end the half-inning, so most are inning-ending runner outs, like `truncated`. "
        "Their pitches stay scorable.",
        "",
        "## How half-innings continue after each PA",
        "",
        "`walk_off` and `game_end_other` are censored: the game stopped before three outs, so the "
        "next state is unknown rather than \"inning over\". RE24 uses `re24_eligible` PAs (complete "
        "half-innings, innings 1-8); terminal payoffs use `payoff_eligible` PAs, keyed by the final-pitch "
        "state (`final_base_out`).",
        "",
        _md(end_type, num),
        "",
        f"- Balls-in-play PAs whose final-pitch base-out state differs from the start: "
        f"{int((bip['final_base_out'] != bip['start_base_out']).sum()):,}; final pitcher differs: "
        f"{int((bip['final_pitcher'] != bip['start_pitcher']).sum())}; final catcher differs: "
        f"{int((bip['final_catcher'] != bip['start_catcher']).sum())}.",
        f"- PAs: {int(pas['re24_eligible'].sum()):,} RE24-eligible, {int(pas['payoff_eligible'].sum()):,} payoff-eligible.",
        "",
        "## Model-fitting eligibility (before fold roles)",
        "",
        "Catcher interference (`exceptional_event`) stays in the sequences and can be scored, but never "
        "trains the outcome or contact models.",
        "",
        _md(fit, num),
        "",
        "## Contact bins for balls in play",
        "",
        f"1-6 = Statcast `launch_speed_angle` (weak to barrel), {7} = bunt, {8} = untracked.",
        "",
        _md(bins, pct),
        "",
        "## Actions: pitch types (thrown pitches)",
        "",
        _md(act, pct),
        "",
        "## Pitch families (thrown pitches)",
        "",
        _md(fam, pct),
        "",
        "## Previous thrown pitch result (thrown pitches)",
        "",
        _md(prev, pct),
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "phase0_coverage.md").write_text(text)
    return text
