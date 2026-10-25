from __future__ import annotations

from datetime import date
import io
import importlib
from typing import List, Tuple

import pandas as pd
import requests
from pybaseball.datasources.statcast import ROOT_URL
from tqdm import tqdm

statcast_module = importlib.import_module("pybaseball.statcast")


OUTPUT_CSV = "statcast_all_pitches_2024_2025.csv"
SEASON_WINDOWS: List[Tuple[date, date]] = [
    # Include Seoul Series (Dodgers vs Padres) on 2024-03-20 and 2024-03-21 and Tokyo Series (Dodgers vs Cubs).
    (date(2024, 3, 20), date(2024, 9, 30)),
    (date(2025, 3, 18), date(2025, 9, 30)),
]


def month_chunks(start: date, end: date) -> List[Tuple[str, str, str]]:
    """Build monthly [start, end] chunks and a YYYY-MM label."""
    chunks: List[Tuple[str, str, str]] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while cursor <= end_ts:
        month_start = cursor.replace(day=1)
        month_end = (month_start + pd.offsets.MonthEnd(0)).to_pydatetime().date()
        chunk_start = max(month_start.to_pydatetime().date(), start)
        chunk_end = min(month_end, end)
        month_label = chunk_start.strftime("%Y-%m")
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat(), month_label))
        cursor = month_start + pd.offsets.MonthBegin(1)

    return chunks


def fetch_raw_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch raw Statcast CSV for one date range."""
    try:
        query_path = statcast_module._SC_SMALL_REQUEST.format(
            start_dt=start_date,
            end_dt=end_date,
            team="",
        )
        response = requests.get(f"{ROOT_URL}{query_path}", timeout=120)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed range {start_date} -> {end_date}: {exc}")
        return pd.DataFrame()


def fetch_statcast_chunk(start_date: str, end_date: str, month_label: str) -> pd.DataFrame:
    """Fetch one monthly chunk by splitting into daily requests.

    Baseball Savant can cap large range responses around 25k rows; daily pulls avoid truncation.
    """
    print(f"Downloading Statcast data: {month_label}")

    daily_frames: List[pd.DataFrame] = []
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    for day in tqdm(days, desc=f"{month_label} days", unit="day"):
        day_str = day.date().isoformat()
        day_df = fetch_raw_range(day_str, day_str)
        if day_df.empty:
            continue
        if "error" in day_df.columns:
            print(f"[WARN] API error for {day_str}: {day_df['error'].iloc[0]}")
            continue
        daily_frames.append(day_df)

    if not daily_frames:
        print("Rows retrieved: 0")
        return pd.DataFrame()

    month_df = pd.concat(daily_frames, ignore_index=True)
    month_df.drop_duplicates(inplace=True)
    print(f"Rows retrieved: {len(month_df):,}")
    return month_df


def fetch_full_dataset() -> pd.DataFrame:
    """Fetch all requested seasons in monthly chunks and combine."""
    frames: List[pd.DataFrame] = []

    for season_start, season_end in SEASON_WINDOWS:
        for start_date, end_date, month_label in month_chunks(season_start, season_end):
            chunk_df = fetch_statcast_chunk(start_date, end_date, month_label)
            if not chunk_df.empty:
                frames.append(chunk_df)

    if not frames:
        print("[WARN] No Statcast data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(inplace=True)

    sort_cols = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    if all(col in combined.columns for col in sort_cols):
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined.sort_values(sort_cols, ascending=[True, True, True, True], inplace=True)

    return combined


def save_dataset(df: pd.DataFrame) -> None:
    """Save combined raw Statcast dataset to CSV."""
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved {len(df):,} rows to {OUTPUT_CSV}")


def print_summary(df: pd.DataFrame) -> None:
    """Print final extraction summary."""
    total_rows = len(df)
    unique_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    unique_batters = df["batter"].nunique(dropna=True) if "batter" in df.columns else 0
    unique_pitchers = df["pitcher"].nunique(dropna=True) if "pitcher" in df.columns else 0

    print("\n=== Final Summary ===")
    print(f"Total rows retrieved: {total_rows:,}")
    print(f"Unique games: {unique_games:,}")
    print(f"Unique batters: {unique_batters:,}")
    print(f"Unique pitchers: {unique_pitchers:,}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


def main() -> None:
    full_df = fetch_full_dataset()
    save_dataset(full_df)
    print_summary(full_df)


if __name__ == "__main__":
    main()
