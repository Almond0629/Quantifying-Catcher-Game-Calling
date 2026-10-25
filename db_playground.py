"""Simple PostgreSQL query playground for this project."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import pandas as pd
import psycopg2


DB_CONFIG: Dict[str, object] = {
    "host": "localhost",
    "port": 5432,
    "dbname": "baseball",
    "user": "postgres",
    "password": "postgres",
}

# Cal Raleigh caught plate appearances where pitch #1 is FF and #2 is SL.
CAL_RALEIGH_FF_SL_QUERY = """
SELECT
    game_date,
    game_pk,
    plate_appearance_id,
    pitcher_name,
    batter_id,
    catcher_id,
    pitch_count,
    first_pitch_type,
    second_pitch_type,
    plate_appearance_result
FROM cal_raleigh_plate_appearances
WHERE catcher_id = 663728
  AND first_pitch_type = 'FF'
  AND second_pitch_type = 'SL'
ORDER BY game_date, game_pk, at_bat_number;
"""


def create_connection() -> psycopg2.extensions.connection:
    """Create PostgreSQL connection for ad-hoc querying."""
    return psycopg2.connect(**DB_CONFIG)


def run_query(sql_query: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Run SQL and return results as dataframe."""
    query = sql_query.strip().rstrip(";")
    if limit is not None:
        query = f"{query} LIMIT {int(limit)}"
    with create_connection() as conn:
        return pd.read_sql_query(query, conn)


def interactive_shell() -> None:
    """Tiny SQL shell for quick experimentation."""
    print("PostgreSQL playground connected to database `baseball`.")
    print("Type SQL and press Enter. Type `exit` to quit.\n")
    while True:
        text = input("sql> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        try:
            result = run_query(text)
            print(result.head(50).to_string(index=False))
            print(f"\nRows returned: {len(result):,}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query playground for baseball PostgreSQL database.")
    parser.add_argument(
        "--query",
        choices=["cal_raleigh_ff_sl", "interactive"],
        default="cal_raleigh_ff_sl",
        help="Which built-in query mode to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for the selected query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query == "interactive":
        interactive_shell()
        return

    df = run_query(CAL_RALEIGH_FF_SL_QUERY, limit=args.limit)
    print(df.to_string(index=False))
    print(f"\nRows returned: {len(df):,}")


if __name__ == "__main__":
    main()
