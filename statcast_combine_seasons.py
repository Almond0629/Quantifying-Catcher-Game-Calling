"""Combine the season CSVs written by statcast_download_seasons.py into one Parquet file.

    python3.12 statcast_combine_seasons.py            # data/statcast_seasons/*.csv -> data/statcast_2015_2023.parquet

The result has the columns, column order and column types of data/statcast_2024_2025.parquet, so the same
reader works on both. Two kinds of column cannot take the reference type and are listed in the printed summary:
columns that are entirely empty in 2024-2025 (stored there with no type) are stored as text here, because some
hold values in earlier seasons; and a column whose earlier values do not fit the reference type keeps a wider one.
Nothing existing is modified, and an existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REF_PARQUET = ROOT / "data" / "statcast_2024_2025.parquet"
SRC = ROOT / "data" / "statcast_seasons"
DST = ROOT / "data" / "statcast_2015_2023.parquet"
YEARS = tuple(range(2015, 2024))


def target_schema(src: Path, years) -> tuple[pa.Schema, dict]:
    widened = {}
    for y in years:
        summary = src / f"summary_{y}.json"
        if summary.exists():
            widened.update({c: y for c in json.loads(summary.read_text()).get("cast_failed", {})})
    fields, notes = [], {"text_because_empty_in_reference": [], "wider_than_reference": {}}
    for f in pq.read_schema(REF_PARQUET):
        t = f.type
        if pa.types.is_null(t):
            t = pa.string()
            notes["text_because_empty_in_reference"].append(f.name)
        elif f.name in widened:
            t = pa.float64() if pa.types.is_integer(t) else pa.string()
            notes["wider_than_reference"][f.name] = str(t)
        fields.append(pa.field(f.name, t))
    return pa.schema(fields), notes


def combine(src: Path, dst: Path, years) -> dict:
    if dst.exists():
        raise FileExistsError(f"{dst} exists; move it away first")
    files = [src / f"statcast_all_pitches_{y}.csv" for y in years]
    absent = [str(f) for f in files if not f.exists()]
    if absent:
        raise FileNotFoundError(f"season files missing: {absent}")
    schema, notes = target_schema(src, years)
    convert = pacsv.ConvertOptions(column_types={f.name: f.type for f in schema}, strings_can_be_null=True)
    rows = {}
    tmp = dst.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for y, f in zip(years, files):
            table = pacsv.read_csv(f, convert_options=convert)
            if table.schema.names != schema.names:
                raise ValueError(f"{f.name}: columns differ from the reference")
            writer.write_table(table.cast(schema))
            rows[y] = table.num_rows
            print(f"{y}: {table.num_rows:,} rows", flush=True)
    tmp.rename(dst)
    return {"file": str(dst), "rows": rows, "total_rows": sum(rows.values()), **notes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--years", default=",".join(map(str, YEARS)))
    a = ap.parse_args()
    print(json.dumps(combine(Path(a.src), Path(a.dst), tuple(int(y) for y in a.years.split(","))), indent=1))
    sys.exit(0)
