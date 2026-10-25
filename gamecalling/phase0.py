"""Phase 0: build the pitch and PA tables, folds, arsenal marks and the coverage report.

Run with: python3.12 -m gamecalling.phase0
"""

from gamecalling import arsenal, config, convert, coverage, loader


def main() -> None:
    if not config.RAW_PARQUET.exists():
        convert.csv_to_parquet()
    pitches = loader.build_pitches(loader.read_raw())
    pitches = pitches.join(arsenal.assign_folds(pitches), on="game_pk")
    support = arsenal.support_table(pitches)
    pitches = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(support))
    pas = loader.build_plate_appearances(pitches)

    pitches.to_parquet(config.PITCHES_PARQUET, index=False)
    pas.to_parquet(config.PAS_PARQUET, index=False)
    support.to_parquet(config.DATA_DIR / "arsenal_support.parquet", index=False)
    coverage.write_report(pitches, pas)
    print(f"{len(pitches):,} pitches, {len(pas):,} PAs; report in {config.REPORTS_DIR / 'phase0_coverage.md'}")


if __name__ == "__main__":
    main()
