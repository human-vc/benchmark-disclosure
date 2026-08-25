"""Emit the hand-fill worklist for missing benchmark release dates."""

import pandas as pd

from .config import BENCHMARK_DATES, INTERIM

COLUMNS = ["slug", "benchmark_release_date", "source_url", "status", "note"]


def main():
    panel = pd.read_csv(INTERIM / "panel.csv")
    missing = panel[panel["benchmark_release_date"].isna()]
    impact = (
        missing.groupby("slug")
        .size()
        .sort_values(ascending=False)
        .rename("pairs_unlocked")
        .reset_index()
    )

    existing = pd.DataFrame(columns=COLUMNS)
    if BENCHMARK_DATES.exists():
        existing = pd.read_csv(BENCHMARK_DATES)

    filled = existing[existing["benchmark_release_date"].notna()]

    annotated = existing[
        existing["benchmark_release_date"].isna()
        & existing.get("note", pd.Series(dtype=object)).notna()
        & (existing.get("note", pd.Series(dtype=object)).astype(str).str.strip() != "")
    ]
    carried = pd.concat([filled, annotated], ignore_index=True) if len(existing) else existing

    todo = impact[~impact["slug"].isin(carried.get("slug", []))].copy()
    todo["benchmark_release_date"] = ""
    todo["source_url"] = ""
    todo["status"] = "todo"
    todo["note"] = ""

    frames = [carried[COLUMNS]] if len(carried) else []
    out = pd.concat(frames + [todo[COLUMNS]], ignore_index=True)
    BENCHMARK_DATES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BENCHMARK_DATES, index=False)

    total = len(panel)
    print(f"undated pairs: {len(missing)} of {total} ({len(missing) / total:.1%})")
    print(f"benchmarks needing a date: {len(impact)}")
    print(f"already filled: {len(filled)}")
    print(f"\nwrote {BENCHMARK_DATES}\n")
    print("ranked by pairs unlocked:")
    cumulative = 0
    for row in impact.itertuples():
        cumulative += row.pairs_unlocked
        mark = " " if row.slug in set(filled["slug"]) else "*"
        print(
            f" {mark} {row.slug:24} {row.pairs_unlocked:5d}  "
            f"cum {cumulative:5d}  ({cumulative / len(missing):5.1%} of gap)"
        )
    print("\n* = still needs a date. Fill benchmark_release_date and source_url,")
    print('  set status to "verified" (or "override" to beat an Epoch date).')


if __name__ == "__main__":
    main()
