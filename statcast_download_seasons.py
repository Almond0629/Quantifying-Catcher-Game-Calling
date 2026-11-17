"""Download earlier Statcast seasons in the layout of statcast_all_pitches_2024_2025.csv.

Same request as statcast_pitch_by_pitch.py (one Baseball Savant request per day), with what a long backfill
needs on top: raw daily responses are cached, failed days are retried and reported instead of skipped, and the
season file is forced to the reference file's columns, column order and column types.

    python3.12 statcast_download_seasons.py fetch 2019     # resumable: only uncached days are requested
    python3.12 statcast_download_seasons.py fetch 2019 --refetch 2019-06-01,2019-06-02   # request those days again
    python3.12 statcast_download_seasons.py build 2019     # season CSV + summary JSON

Window: every day from the season's first regular-season date to its last, found from the data (game_type "R");
rows of other game types inside the window are kept, as in the reference file. Nothing outside
data/statcast_seasons/ is written.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pybaseball.datasources.statcast import ROOT_URL

statcast_module = importlib.import_module("pybaseball.statcast")

ROOT = Path(__file__).resolve().parent
REF_CSV = ROOT / "statcast_all_pitches_2024_2025.csv"
REF_PARQUET = ROOT / "data" / "statcast_2024_2025.parquet"
OUT = ROOT / "data" / "statcast_seasons"

# Expected first and last regular-season dates. Only a starting point: the window is confirmed from the data.
EXPECTED = {
    2015: ("2015-04-05", "2015-10-04"), 2016: ("2016-04-03", "2016-10-02"), 2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"), 2019: ("2019-03-20", "2019-09-29"), 2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"), 2022: ("2022-04-07", "2022-10-05"), 2023: ("2023-03-30", "2023-10-01"),
}
PAD_BEFORE, PAD_AFTER, MAX_EXTEND = 5, 3, 20
BACKOFF = (5, 15, 45, 90)          # seconds before retries 2..5
ROW_CAP = 25_000                   # Savant truncates large responses near this size
SORT = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
KEY = ["game_pk", "at_bat_number", "pitch_number"]


def parse(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8")
    if not text.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def request_day(day: str) -> bytes:
    path = statcast_module._SC_SMALL_REQUEST.format(start_dt=day, end_dt=day, team="")
    response = requests.get(f"{ROOT_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.content


def fetch_day(day: str) -> tuple[bytes, dict]:
    """One day's raw response, retried. Raises RuntimeError after the last failed try."""
    last = None
    for attempt in range(len(BACKOFF) + 1):
        if attempt:
            time.sleep(BACKOFF[attempt - 1])
        try:
            raw = request_day(day)
            df = parse(raw)
            if "error" in df.columns:
                raise RuntimeError(f"API error: {df['error'].iloc[0]}")
            if df.empty:               # off day, or a bad response: ask once more before believing it
                time.sleep(2)
                raw2 = request_day(day)
                df2 = parse(raw2)
                if not df2.empty and "error" not in df2.columns:
                    raw, df = raw2, df2
            return raw, {"rows": len(df), "tries": attempt + 1,
                         "games": int(df["game_pk"].nunique()) if len(df) else 0,
                         "r_rows": int((df["game_type"] == "R").sum()) if len(df) else 0,
                         "capped": bool(len(df) >= ROW_CAP)}
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[WARN] {day} try {attempt + 1}: {exc}", flush=True)
    raise RuntimeError(f"{day}: {last}")


def days_between(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [(a + timedelta(n)).isoformat() for n in range((b - a).days + 1)]


def shift(day: str, n: int) -> str:
    return (date.fromisoformat(day) + timedelta(n)).isoformat()


def stage_fetch(year: int, start: str, end: str, out: Path, pad_before: int, pad_after: int, max_extend: int,
                refetch: tuple = ()) -> int:
    cache = out / "cache" / str(year)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = out / f"days_{year}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for day in refetch:                # asked for again: forget the cached answer, keep the old file until replaced
        meta.pop(day, None)
    failed = []

    def ensure(day: str) -> bool:
        path = cache / f"{day}.csv"
        if path.exists() and day in meta:
            return True
        try:
            raw, info = fetch_day(day)
        except RuntimeError as exc:
            failed.append(str(exc))
            return False
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.rename(path)
        meta[day] = info
        meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
        print(f"{day}: {info['rows']:>6,} rows, {info['games']:>2} games, tries {info['tries']}"
              f"{'  [AT ROW CAP]' if info['capped'] else ''}", flush=True)
        return True

    first, last = shift(start, -pad_before), shift(end, pad_after)
    for day in days_between(first, last):
        ensure(day)
    # A regular-season game on the first or last fetched day means the window may be wider: extend.
    for _ in range(max_extend):
        if not meta.get(first, {}).get("r_rows"):
            break
        first = shift(first, -1)
        ensure(first)
    for _ in range(max_extend):
        if not meta.get(last, {}).get("r_rows"):
            break
        last = shift(last, 1)
        ensure(last)
    (out / f"window_{year}.json").write_text(json.dumps({"first": first, "last": last}))
    if failed:
        print(f"[FAILED] {len(failed)} day(s); run fetch again:\n  " + "\n  ".join(failed), flush=True)
        return 1
    print(f"fetch {year}: {first} .. {last} complete ({len(days_between(first, last))} days)", flush=True)
    return 0


def reference_columns() -> list[str]:
    with open(REF_CSV, newline="") as f:
        return next(csv.reader(f))


def cast_to_reference(df: pd.DataFrame) -> dict:
    """Give each column the reference file's type, so integers and floats are written the same way."""
    notes = {"cast_failed": {}, "reference_empty_but_populated": []}
    for field in pq.read_schema(REF_PARQUET):
        c = field.name
        if c not in df.columns or c == "game_date":
            continue
        try:
            if pa.types.is_integer(field.type):
                df[c] = pd.to_numeric(df[c], errors="raise").astype("Int64")
            elif pa.types.is_floating(field.type):
                df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
            elif pa.types.is_null(field.type) and df[c].notna().any():
                notes["reference_empty_but_populated"].append(c)
        except (TypeError, ValueError) as exc:
            notes["cast_failed"][c] = str(exc)[:200]
    return notes


def share_null(s: pd.Series) -> float | None:
    return round(float(s.isna().mean()), 5) if len(s) else None


def stage_build(year: int, out: Path, allow_boundary: bool) -> int:
    window = json.loads((out / f"window_{year}.json").read_text())
    meta = json.loads((out / f"days_{year}.json").read_text())
    cache = out / "cache" / str(year)
    days = days_between(window["first"], window["last"])
    missing = [d for d in days if d not in meta or not (cache / f"{d}.csv").exists()]
    if missing:
        print(f"[STOP] {len(missing)} day(s) not fetched: {missing[:10]}", flush=True)
        return 1
    ref = reference_columns()
    with open(out / "build.lock", "w") as lock:      # one build at a time: a season needs a few GB of memory
        fcntl.flock(lock, fcntl.LOCK_EX)
        frames = [df for df in (parse((cache / f"{d}.csv").read_bytes()) for d in days) if not df.empty]
        df = pd.concat(frames, ignore_index=True)
        del frames
        fetched_rows = len(df)
        reg = df.loc[(df["game_type"] == "R") & (df["game_year"] == year), "game_date"].astype(str)
        first_r, last_r = reg.min(), reg.max()
        if not allow_boundary and (first_r <= window["first"] or last_r >= window["last"]):
            print(f"[STOP] regular season {first_r}..{last_r} touches the fetched window {window}: fetch more days",
                  flush=True)
            return 1
        gd = df["game_date"].astype(str)
        df = df[(gd >= first_r) & (gd <= last_r)]
        missing_cols = [c for c in ref if c not in df.columns]
        if missing_cols:
            print(f"[STOP] reference columns missing from the download: {missing_cols}", flush=True)
            return 1
        extra_cols = [c for c in df.columns if c not in ref]
        df = df[ref]
        before = len(df)
        df = df.drop_duplicates()
        notes = cast_to_reference(df)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        bad_dates = int(df["game_date"].isna().sum())
        df = df.sort_values(SORT, kind="mergesort")
        dst = out / f"statcast_all_pitches_{year}.csv"
        tmp = dst.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.rename(dst)
        with open(dst, newline="") as f:
            header_ok = next(csv.reader(f)) == ref

        in_play = df[df["type"] == "X"]
        per_game = df[df["game_type"] == "R"].groupby("game_pk").size()
        window_days = days_between(first_r, last_r)
        summary = {
            "year": year, "file": str(dst), "header_matches_reference": header_ok, "columns": len(df.columns),
            "rows": len(df), "rows_fetched_before_trim": fetched_rows, "duplicate_rows_dropped": before - len(df),
            "duplicate_pitch_keys": int(df.duplicated(KEY).sum()), "bad_game_dates": bad_dates,
            "first_regular_season_date": first_r, "last_regular_season_date": last_r,
            "expected_window": EXPECTED.get(year), "fetched_window": window,
            "rows_by_game_type": {k: int(v) for k, v in df["game_type"].value_counts().items()},
            "games_by_game_type": {k: int(v) for k, v in df.groupby("game_type")["game_pk"].nunique().items()},
            "regular_season_pitches_per_game": {"min": int(per_game.min()), "median": float(per_game.median()),
                                                "max": int(per_game.max()), "games_under_150": int((per_game < 150).sum())},
            "days_in_window": len(window_days), "days_without_rows": [d for d in window_days if not meta[d]["rows"]],
            "days_at_row_cap": [d for d in window_days if meta[d]["capped"]],
            "days_needing_retries": {d: meta[d]["tries"] for d in window_days if meta[d]["tries"] > 1},
            "max_rows_in_a_day": max(meta[d]["rows"] for d in window_days),
            "extra_columns_dropped": extra_cols, **notes,
            "null_share": {"pitch_type": share_null(df["pitch_type"]), "fielder_2": share_null(df["fielder_2"]),
                           "release_speed": share_null(df["release_speed"]), "plate_x": share_null(df["plate_x"]),
                           "in_play.launch_speed_angle": share_null(in_play["launch_speed_angle"]),
                           "in_play.estimated_woba_using_speedangle": share_null(in_play["estimated_woba_using_speedangle"]),
                           "in_play.bb_type": share_null(in_play["bb_type"])},
            "pitch_type_counts": {str(k): int(v) for k, v in df["pitch_type"].value_counts(dropna=False).items()},
            "rows_by_month": {str(k): int(v) for k, v in df["game_date"].dt.strftime("%Y-%m").value_counts().sort_index().items()},
        }
    (out / f"summary_{year}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("year", "rows", "header_matches_reference", "first_regular_season_date",
                                               "last_regular_season_date", "games_by_game_type", "duplicate_pitch_keys",
                                               "days_without_rows", "days_at_row_cap", "extra_columns_dropped",
                                               "cast_failed")}, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "build"])
    ap.add_argument("year", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pad-before", type=int, default=PAD_BEFORE)
    ap.add_argument("--pad-after", type=int, default=PAD_AFTER)
    ap.add_argument("--max-extend", type=int, default=MAX_EXTEND)
    ap.add_argument("--refetch", default="", help="comma-separated days to request again")
    ap.add_argument("--allow-boundary", action="store_true", help="testing only: build a partial window")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "fetch":
        start, end = a.start or EXPECTED[a.year][0], a.end or EXPECTED[a.year][1]
        return stage_fetch(a.year, start, end, out, a.pad_before, a.pad_after, a.max_extend,
                           tuple(d for d in a.refetch.split(",") if d))
    return stage_build(a.year, out, a.allow_boundary)


if __name__ == "__main__":
    sys.exit(main())
