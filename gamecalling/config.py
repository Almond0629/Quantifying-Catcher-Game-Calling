"""Paths and fixed choices for the pipeline. Method details live in project_plan.md."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "statcast_all_pitches_2024_2025.csv"
DATA_DIR = ROOT / "data"
RAW_PARQUET = DATA_DIR / "statcast_2024_2025.parquet"
PITCHES_PARQUET = DATA_DIR / "pitches_2024_2025.parquet"
PAS_PARQUET = DATA_DIR / "plate_appearances_2024_2025.parquet"
REPORTS_DIR = ROOT / "reports"

SEASONS = (2024, 2025)

# Action unit: what a pitch call is scored as (user decision, 2026-10-25).
# "pitch_type" (main): Statcast pitch_type, except the codes below.
# "family" (sensitivity check): the pitch families below.
ACTION_UNIT = "pitch_type"
NON_ACTION_PITCH_TYPES = {"PO", "FA", "UN"}  # pitchout, "other", unknown

# Position players who pitch are excluded (user decision, 2026-10-25). A pitcher
# is a position player in a season if he batted at least this many times in the
# data (all seasons) and started no games as a pitcher that season. Real pitchers
# batted at most 2 times in 2024-25; position players who pitched, at least 9.
# The no-start condition keeps two-way players (Ohtani: 14 starts in 2025).
POSITION_PLAYER_MIN_BAT_PA = 5

# Pitch families (sensitivity check). Pitch types not listed here (EP, FA, KN, SC,
# UN, PO) have no family.
PITCH_FAMILIES = {
    "FF": "fastball", "SI": "fastball",
    "FC": "cutter",
    "SL": "slider", "ST": "slider", "SV": "slider",
    "CU": "curveball", "KC": "curveball", "CS": "curveball",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed",
}
FAMILY_ORDER = ("fastball", "cutter", "slider", "curveball", "offspeed")

# Arsenal support rule (user decision, 2026-10-25; not frozen). Judged per batter
# hand on the two training folds of each scoring fold combined: an action passes
# against a hand when its share of the pitcher's pitches to that hand AND its
# pitch count reach the role's cutoffs. The pitcher must pitch in both training folds.
ARSENAL_RULES = {
    "SP": {"min_share": 0.02, "min_count": 10},
    "RP": {"min_share": 0.05, "min_count": 5},
}
# A pitcher is a starter for a scoring fold if at least this share of his
# appearances in its two training folds are starts (first pitcher for his team).
SP_MIN_START_SHARE = 0.5

# Cross-fitting: three game folds per season, fixed by original game_pk.
N_FOLDS = 3
FOLD_SEED = 20261024

# Ball-in-play payoff cells (final base-out x contact bin) with fewer PAs than
# this in a role's fold use the fallback in payoff.shift_fallback (same fold).
# 1 = only empty cells (user decision, 2026-10-27); 6-11 of 192 per season-fold are empty.
MIN_CONTACT_CELL_N = 1

# Count-graph solver: repeat passes until no value changes by more than this (runs).
SOLVER_TOL = 1e-9

# Marlins called pitches from the dugout for their final nine games of 2025,
# starting Sept 19, 2025 at Texas (MLB.com). Flagged, not excluded.
DUGOUT_CALLED_TEAM = "MIA"
DUGOUT_CALLED_START = "2025-09-19"
DUGOUT_CALLED_EXPECTED_GAMES = 9
