"""The placebo contrast across every analytic choice that could manufacture it.

The paper's claim is that a measurement choice moves the contrast, so the
contrast has to be reported across those choices rather than at one setting.

The headline is a coordinate in this grid, not a separate code path. `BASELINE`
names it, `test_sensitivity` asserts that cell equals what `paper_numbers`
writes, and the two cannot drift apart without a test failing. The alternative,
a sweep maintained beside the estimator, is how the manuscript came to describe
an eight-benchmark minimum that no headline number ever applied.

Percentiles depend only on the window width, so they are computed once per width
and every threshold reuses the scored frame.
"""

import numpy as np
import pandas as pd

from .config import INTERIM, RELEASE_COL, WINDOW_DAYS
from .percentiles import within_benchmark_percentile
from .placebo_calibration import contrast_by_release
from .stats import randomization_test_mean

WIDTHS = (91, 182, 273, 365)
THRESHOLDS = (1, 5, 8, 10)

BASELINE = {"window_days": WINDOW_DAYS, "min_benchmarks": 1}

COLUMNS = [
    "spec_id", "window_days", "min_benchmarks",
    "mean", "median", "share_positive", "p_value",
    "n_releases", "n_clusters", "share_newer_gap",
]


def _panel():
    return pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )


def _threshold(scored, min_benchmarks):
    """Releases carrying at least this many distinct scored benchmarks."""
    if min_benchmarks <= 1:
        return scored
    counts = scored.groupby(RELEASE_COL)["slug"].transform("nunique")
    return scored[counts >= min_benchmarks]


def _share_newer_gap(scored):
    """Eligible minus placebo mean share of the window newer than the model."""
    cells = scored[scored["eligible"] | scored["placebo"]]
    if cells.empty or "share_newer" not in cells:
        return np.nan
    elig = cells.loc[cells["eligible"], "share_newer"].mean()
    plac = cells.loc[cells["placebo"], "share_newer"].mean()
    return float(plac - elig)


def estimate(scored, min_benchmarks):
    """The placebo contrast on an already-scored frame at one threshold."""
    kept = _threshold(scored, min_benchmarks)
    contrast = contrast_by_release(kept)
    if contrast.empty:
        return {"mean": np.nan, "median": np.nan, "share_positive": np.nan,
                "p_value": np.nan, "n_releases": 0, "n_clusters": 0,
                "share_newer_gap": np.nan}
    values = contrast["contrast"].to_numpy()
    clusters = contrast["organization"].to_numpy()
    test = randomization_test_mean(values, cluster=clusters)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "share_positive": float((values > 0).mean()),
        "p_value": float(test["p_value"]),
        "n_releases": int(len(values)),
        "n_clusters": int(pd.unique(clusters).size),
        "share_newer_gap": _share_newer_gap(kept),
    }


def sweep(frame, widths=WIDTHS, thresholds=THRESHOLDS):
    """Every (window width, threshold) cell, percentiles computed once per width."""
    rows = []
    for width in widths:
        scored = within_benchmark_percentile(frame.copy(), window_days=width)
        for threshold in thresholds:
            row = {"spec_id": f"w{width}_b{threshold}",
                   "window_days": width, "min_benchmarks": threshold}
            row.update(estimate(scored, threshold))
            rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


def baseline_row(table):
    """The grid cell the manuscript reports."""
    match = table
    for key, value in BASELINE.items():
        match = match[match[key] == value]
    if len(match) != 1:
        raise ValueError(f"BASELINE selects {len(match)} rows, expected exactly 1")
    return match.iloc[0]


def leave_one_org_out(scored, min_benchmarks=BASELINE["min_benchmarks"]):
    """The contrast with each organisation removed in turn."""
    contrast = contrast_by_release(_threshold(scored, min_benchmarks))
    if contrast.empty:
        return pd.DataFrame(columns=["dropped", "mean", "n_releases", "share_of_releases"])
    total = len(contrast)
    rows = [{"dropped": "(none)", "mean": float(contrast["contrast"].mean()),
             "n_releases": total, "share_of_releases": 1.0}]
    counts = contrast["organization"].value_counts()
    for org in counts.index:
        kept = contrast[contrast["organization"] != org]
        rows.append({
            "dropped": org,
            "mean": float(kept["contrast"].mean()),
            "n_releases": int(len(kept)),
            "share_of_releases": float(counts[org] / total),
        })
    return pd.DataFrame(rows)


def main():
    frame = _panel()

    table = sweep(frame)
    table.to_csv(INTERIM.parent / "sensitivity_window.csv", index=False)
    base = baseline_row(table)
    print("placebo contrast across the specification grid")
    print("  width  min_b     mean   median  positive   rel  orgs       p")
    for row in table.itertuples():
        mark = "  <- headline" if row.spec_id == base.spec_id else ""
        print(f"  {row.window_days:>5}  {row.min_benchmarks:>5}  "
              f"{row.mean:>+7.2f}  {row.median:>+7.2f}  {row.share_positive:>7.1%}  "
              f"{row.n_releases:>4}  {row.n_clusters:>4}  {row.p_value:>6.4f}{mark}")
    print(f"\n  {len(table)} specifications, "
          f"{int((table['mean'] > 0).sum())} positive, "
          f"range {table['mean'].min():+.2f} to {table['mean'].max():+.2f}")

    at_width = table[table["min_benchmarks"] == BASELINE["min_benchmarks"]]
    corr = np.corrcoef(at_width["share_newer_gap"], at_width["mean"])[0, 1]
    slope = np.polyfit(at_width["share_newer_gap"], at_width["mean"], 1)[0]
    print(f"  contrast against the share-newer gap it comes from: "
          f"r = {corr:.3f}, slope {slope:+.1f} points per unit gap")

    scored = within_benchmark_percentile(frame.copy(), window_days=WINDOW_DAYS)
    loo = leave_one_org_out(scored)
    loo.to_csv(INTERIM.parent / "sensitivity_leave_one_out.csv", index=False)
    dropped = loo[loo["dropped"] != "(none)"]
    print(f"\nleave one organisation out (full sample "
          f"{loo.loc[loo['dropped'] == '(none)', 'mean'].iloc[0]:+.2f})")
    for row in dropped.head(5).itertuples():
        print(f"  drop {row.dropped:<22} {row.mean:>+6.2f}  "
              f"({row.share_of_releases:.1%} of releases)")
    print(f"  over all {len(dropped)} drops: "
          f"{dropped['mean'].min():+.2f} to {dropped['mean'].max():+.2f}")


if __name__ == "__main__":
    main()
