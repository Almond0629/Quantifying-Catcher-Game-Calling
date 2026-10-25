"""SQL-first Statcast pipeline: load full CSV, then derive Cal Raleigh tables in PostgreSQL."""

from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


CSV_PATH = "statcast_all_pitches_2024_2025.csv"
TARGET_DB = "baseball"
MAINTENANCE_DB = "postgres"
CAL_RALEIGH_ID = 663728

USE_COPY_LOADER = True
FALLBACK_CHUNK_SIZE = 10000

DB_SERVER_CONFIG: Dict[str, object] = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
}

IMPORTANT_COLUMNS = [
    "game_date",
    "game_pk",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "batter_name",
    "player_name",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "outs_when_up",
    "events",
    "description",
    "stand",
    "p_throws",
    "fielder_2",
    "home_team",
    "away_team",
]


def create_database_if_missing() -> bool:
    """Create target database if it does not already exist."""
    created = False
    conn = psycopg2.connect(dbname=MAINTENANCE_DB, **DB_SERVER_CONFIG)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (TARGET_DB,))
            exists = cur.fetchone() is not None
            if exists:
                print(f"[INFO] Database `{TARGET_DB}` already exists.")
                return False
            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(TARGET_DB)))
            created = True
            print(f"[INFO] Database `{TARGET_DB}` created.")
    finally:
        conn.close()
    return created


def connect_to_baseball_db() -> psycopg2.extensions.connection:
    """Connect to target baseball database."""
    return psycopg2.connect(dbname=TARGET_DB, **DB_SERVER_CONFIG)


def load_csv(path: str) -> pd.DataFrame:
    """Load local Statcast CSV."""
    print(f"[INFO] Reading CSV: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {path}")
        return pd.DataFrame()
    print(f"[INFO] Raw CSV rows: {len(df):,}")
    return df


def _sanitize_column(name: str) -> str:
    value = name.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "col"
    if value[0].isdigit():
        value = f"c_{value}"
    return value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to PostgreSQL-safe column names and ensure important fields exist."""
    normalized = df.copy()
    renamed: Dict[str, str] = {}
    used = set()

    for original in normalized.columns:
        base = _sanitize_column(str(original))
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        renamed[original] = candidate

    normalized.rename(columns=renamed, inplace=True)

    for col in IMPORTANT_COLUMNS:
        if col not in normalized.columns:
            print(f"[WARN] Missing expected column `{col}` in CSV; filling with NULLs.")
            normalized[col] = pd.NA

    return normalized


def filter_target_games(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 2024/2025 regular season plus only target Seoul/Tokyo matchups."""
    if df.empty:
        return df.copy()
    if "game_date" not in df.columns:
        print("[WARN] `game_date` missing; cannot apply season filter.")
        return df.copy()

    dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    regular_2024 = (dates >= pd.to_datetime("2024-03-28").date()) & (
        dates <= pd.to_datetime("2024-09-30").date()
    )
    regular_2025 = (dates >= pd.to_datetime("2025-03-27").date()) & (
        dates <= pd.to_datetime("2025-09-28").date()
    )

    special_2024_dates = dates.isin(
        [pd.to_datetime("2024-03-20").date(), pd.to_datetime("2024-03-21").date()]
    )
    special_2025_dates = dates.isin(
        [pd.to_datetime("2025-03-18").date(), pd.to_datetime("2025-03-19").date()]
    )

    if "home_team" in df.columns and "away_team" in df.columns:
        home = df["home_team"].astype(str).str.upper().str.strip()
        away = df["away_team"].astype(str).str.upper().str.strip()

        dodgers_padres = ((home == "LAD") & (away == "SD")) | ((home == "SD") & (away == "LAD"))
        dodgers_cubs = ((home == "LAD") & (away == "CHC")) | ((home == "CHC") & (away == "LAD"))

        special_games = (special_2024_dates & dodgers_padres) | (special_2025_dates & dodgers_cubs)
    else:
        print(
            "[WARN] Missing `home_team`/`away_team`; special games are date-filtered only."
        )
        special_games = special_2024_dates | special_2025_dates

    filtered = df[regular_2024 | regular_2025 | special_games].copy()
    print(f"[INFO] Rows after regular-season + Seoul/Tokyo filter: {len(filtered):,}")
    return filtered


def infer_pg_type(column_name: str, series: pd.Series) -> str:
    """Infer practical PostgreSQL type for a dataframe column."""
    if column_name == "game_date":
        return "DATE"

    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"

    # Try integer/float coercion for object columns before defaulting to text.
    numeric_sample = pd.to_numeric(series.dropna().head(5000), errors="coerce")
    if not numeric_sample.empty and numeric_sample.notna().mean() > 0.98:
        if (numeric_sample.dropna() % 1 == 0).all():
            return "BIGINT"
        return "DOUBLE PRECISION"

    return "TEXT"


def create_raw_table(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    """Replace and recreate raw table from dataframe schema."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS raw_statcast_pitches CASCADE;")
        column_defs = []
        for col in df.columns:
            pg_type = infer_pg_type(col, df[col])
            column_defs.append(sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type)))

        create_sql = sql.SQL("CREATE TABLE raw_statcast_pitches ({});").format(
            sql.SQL(", ").join(column_defs)
        )
        cur.execute(create_sql)
    print("[INFO] Created table `raw_statcast_pitches`.")


def _prepare_dataframe_for_load(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["game_date"] = pd.to_datetime(prepared["game_date"], errors="coerce").dt.date
    return prepared


def _chunked_rows(values: Sequence[Tuple], chunk_size: int) -> Iterable[Sequence[Tuple]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def bulk_load_raw_data_copy(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    """Load full dataframe to raw table using COPY FROM STDIN."""
    load_df = _prepare_dataframe_for_load(df)
    buffer = io.StringIO()
    load_df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)

    columns_ident = sql.SQL(", ").join(sql.Identifier(c) for c in load_df.columns)
    copy_stmt = sql.SQL(
        "COPY raw_statcast_pitches ({}) FROM STDIN WITH (FORMAT CSV, NULL '\\N');"
    ).format(columns_ident)

    with conn.cursor() as cur:
        cur.copy_expert(copy_stmt.as_string(conn), buffer)
    print(f"[INFO] COPY loaded rows: {len(load_df):,}")


def bulk_load_raw_data_execute_values(
    conn: psycopg2.extensions.connection, df: pd.DataFrame, chunk_size: int = FALLBACK_CHUNK_SIZE
) -> None:
    """Fallback loader using execute_values."""
    load_df = _prepare_dataframe_for_load(df).where(pd.notna(df), None)
    rows = list(load_df.itertuples(index=False, name=None))
    insert_sql = sql.SQL("INSERT INTO raw_statcast_pitches ({}) VALUES %s").format(
        sql.SQL(", ").join(sql.Identifier(c) for c in load_df.columns)
    )

    inserted = 0
    with conn.cursor() as cur:
        for batch in _chunked_rows(rows, chunk_size):
            execute_values(cur, insert_sql.as_string(conn), batch, page_size=chunk_size)
            inserted += len(batch)
            print(f"[INFO] execute_values inserted: {inserted:,}")


def bulk_load_raw_data(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> None:
    """Bulk load raw CSV rows into raw_statcast_pitches."""
    if df.empty:
        print("[WARN] Empty dataframe; nothing to load.")
        return
    if USE_COPY_LOADER:
        bulk_load_raw_data_copy(conn, df)
    else:
        bulk_load_raw_data_execute_values(conn, df)


def create_raw_indexes(conn: psycopg2.extensions.connection) -> None:
    """Create indexes on raw table."""
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_game_date ON raw_statcast_pitches (game_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_game_pk ON raw_statcast_pitches (game_pk);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_at_bat_number ON raw_statcast_pitches (at_bat_number);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_pitch_number ON raw_statcast_pitches (pitch_number);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_batter ON raw_statcast_pitches (batter);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_pitcher ON raw_statcast_pitches (pitcher);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_fielder_2 ON raw_statcast_pitches (fielder_2);")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_game_ab_pitch
            ON raw_statcast_pitches (game_pk, at_bat_number, pitch_number);
            """
        )
    print("[INFO] Created raw table indexes.")


def create_cal_raleigh_tables(conn: psycopg2.extensions.connection) -> None:
    """Derive Cal-specific pitch and plate-appearance tables in SQL."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS cal_raleigh_caught_pitches;")
        cur.execute(
            """
            CREATE TABLE cal_raleigh_caught_pitches AS
            SELECT
                game_date,
                game_pk,
                inning,
                inning_topbot,
                at_bat_number,
                pitch_number,
                (game_pk::text || '_' || at_bat_number::text) AS plate_appearance_id,
                pitcher AS pitcher_id,
                batter AS batter_id,
                batter_name,
                fielder_2 AS catcher_id,
                player_name AS pitcher_name,
                pitch_type,
                pitch_name,
                release_speed AS velocity,
                release_spin_rate,
                pfx_x AS movement_x,
                pfx_z AS movement_z,
                plate_x,
                plate_z,
                balls,
                strikes,
                outs_when_up,
                stand,
                p_throws,
                events,
                description
            FROM raw_statcast_pitches
            WHERE fielder_2 = %s
            ORDER BY game_date, game_pk, at_bat_number, pitch_number;
            """,
            (CAL_RALEIGH_ID,),
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_pitches_pa_id
            ON cal_raleigh_caught_pitches (plate_appearance_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_pitches_game_ab_pitch
            ON cal_raleigh_caught_pitches (game_pk, at_bat_number, pitch_number);
            """
        )

        cur.execute("DROP TABLE IF EXISTS cal_raleigh_plate_appearances;")
        cur.execute(
            """
            CREATE TABLE cal_raleigh_plate_appearances AS
            WITH pa_agg AS (
                SELECT
                    plate_appearance_id,
                    MIN(game_pk) AS game_pk,
                    MIN(game_date) AS game_date,
                    MIN(at_bat_number) AS at_bat_number,
                    MIN(batter_id) AS batter_id,
                    MIN(pitcher_name) AS pitcher_name,
                    MIN(catcher_id) AS catcher_id,
                    COUNT(*)::INT AS pitch_count,
                    MAX(CASE WHEN pitch_number = 1 THEN pitch_type END) AS first_pitch_type,
                    MAX(CASE WHEN pitch_number = 2 THEN pitch_type END) AS second_pitch_type,
                    jsonb_agg(
                        jsonb_build_object(
                            'pitch_number', pitch_number,
                            'pitch_type', pitch_type,
                            'pitch_name', pitch_name,
                            'velocity', velocity,
                            'balls', balls,
                            'strikes', strikes,
                            'description', description,
                            'events', events
                        )
                        ORDER BY pitch_number
                    ) AS pitch_sequence_json
                FROM cal_raleigh_caught_pitches
                GROUP BY plate_appearance_id
            ),
            pa_last_pitch AS (
                SELECT DISTINCT ON (plate_appearance_id)
                    plate_appearance_id,
                    COALESCE(events, description) AS plate_appearance_result
                FROM cal_raleigh_caught_pitches
                ORDER BY plate_appearance_id, pitch_number DESC
            )
            SELECT
                a.plate_appearance_id,
                a.game_pk,
                a.game_date,
                a.at_bat_number,
                a.batter_id,
                a.pitcher_name,
                a.catcher_id,
                a.pitch_count,
                a.first_pitch_type,
                a.second_pitch_type,
                a.pitch_sequence_json,
                l.plate_appearance_result
            FROM pa_agg a
            LEFT JOIN pa_last_pitch l USING (plate_appearance_id)
            ORDER BY a.game_date, a.game_pk, a.at_bat_number;
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cal_pa_id
            ON cal_raleigh_plate_appearances (plate_appearance_id);
            """
        )
    print("[INFO] Created derived Cal Raleigh tables.")


def _query_one(conn: psycopg2.extensions.connection, query: str) -> int:
    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()
        return int(result[0]) if result and result[0] is not None else 0


def print_summary(conn: psycopg2.extensions.connection, db_created: bool, csv_rows: int) -> None:
    """Print run summary and previews."""
    raw_rows = _query_one(conn, "SELECT COUNT(*) FROM raw_statcast_pitches;")
    cal_rows = _query_one(conn, "SELECT COUNT(*) FROM cal_raleigh_caught_pitches;")
    cal_pa_rows = _query_one(conn, "SELECT COUNT(*) FROM cal_raleigh_plate_appearances;")
    cal_hitters = _query_one(
        conn,
        "SELECT COUNT(DISTINCT batter_id) FROM cal_raleigh_caught_pitches;",
    )

    print("\n=== Summary ===")
    print(f"Database `{TARGET_DB}` status: {'created' if db_created else 'already existed'}")
    print(f"Total CSV rows read: {csv_rows:,}")
    print(f"Total rows loaded into raw_statcast_pitches: {raw_rows:,}")
    print(f"Total rows in cal_raleigh_caught_pitches: {cal_rows:,}")
    print(f"Total distinct plate appearances in cal_raleigh_plate_appearances: {cal_pa_rows:,}")
    print(f"Total distinct hitters caught by Cal Raleigh: {cal_hitters:,}")

    cal_pitch_preview = pd.read_sql_query(
        """
        SELECT * FROM cal_raleigh_caught_pitches
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        LIMIT 10;
        """,
        conn,
    )
    cal_pa_preview = pd.read_sql_query(
        """
        SELECT * FROM cal_raleigh_plate_appearances
        ORDER BY game_date, game_pk, at_bat_number
        LIMIT 10;
        """,
        conn,
    )

    print("\nFirst 10 rows from cal_raleigh_caught_pitches:")
    print(cal_pitch_preview.to_string(index=False))
    print("\nFirst 10 rows from cal_raleigh_plate_appearances:")
    print(cal_pa_preview.to_string(index=False))


def main() -> None:
    db_created = create_database_if_missing()
    conn = None
    try:
        conn = connect_to_baseball_db()
        # raw_df = load_csv(CSV_PATH)
        # if raw_df.empty:
        #     print("[ERROR] CSV is empty or missing; aborting.")
        #     return
        # normalized_df = normalize_columns(raw_df)
        # filtered_raw_df = filter_target_games(normalized_df)
        # create_raw_table(conn, filtered_raw_df)
        # bulk_load_raw_data(conn, filtered_raw_df)
        # create_raw_indexes(conn)
        create_cal_raleigh_tables(conn)
        conn.commit()
        # print_summary(conn, db_created=db_created, csv_rows=len(raw_df))
    except Exception as exc:  # noqa: BLE001
        if conn is not None:
            conn.rollback()
        print(f"[ERROR] Pipeline failed; transaction rolled back: {exc}")
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
