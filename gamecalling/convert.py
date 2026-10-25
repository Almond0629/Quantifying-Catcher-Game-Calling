"""Convert the raw Statcast CSV to Parquet, keeping every row and column."""

import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from gamecalling import config

# Columns that pyarrow could guess wrongly from the first block of rows.
STRING_COLUMNS = [
    "pitch_type", "game_date", "player_name", "events", "description", "des",
    "game_type", "stand", "p_throws", "home_team", "away_team", "type", "bb_type",
    "inning_topbot", "pitch_name", "if_fielding_alignment", "of_fielding_alignment",
]
NULLABLE_FLOAT_COLUMNS = ["on_1b", "on_2b", "on_3b", "launch_speed_angle", "zone"]


def csv_to_parquet(src=config.RAW_CSV, dst=config.RAW_PARQUET) -> int:
    import pyarrow as pa

    column_types = {c: pa.string() for c in STRING_COLUMNS}
    column_types.update({c: pa.float64() for c in NULLABLE_FLOAT_COLUMNS})
    table = pacsv.read_csv(
        src,
        convert_options=pacsv.ConvertOptions(column_types=column_types, strings_can_be_null=True),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")
    return table.num_rows


if __name__ == "__main__":
    n = csv_to_parquet()
    print(f"Wrote {n:,} rows to {config.RAW_PARQUET}")
