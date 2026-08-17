"""Reported-vs-omitted selectivity statistics.

Requires a populated disclosure coding sheet. See protocol/coding-protocol.md
for the instrument and protocol/disclosure-template.csv for the schema.
"""

import sys

import numpy as np
import pandas as pd

from .config import INTERIM, MIN_BENCHMARKS, MODEL_COL, ROOT

CODING_SHEET = ROOT / "data" / "disclosures.csv"
HIGH_SUSPICION = {"D", "E", "G"}


def within_benchmark_percentile(panel):
    panel = panel.copy()
    panel["percentile"] = panel.groupby("slug")["score"].rank(pct=True) * 100
    return panel


def load_coding():
    if not CODING_SHEET.exists():
        return None
    coding = pd.read_csv(CODING_SHEET)
    return coding if len(coding) else None


def gap_by_model(merged):
    rows = []
    for model, group in merged.groupby(MODEL_COL):
        reported = group.loc[group["reported"] == 1, "percentile"]
        omitted = group.loc[group["reported"] == 0, "percentile"]
        if len(reported) < 2 or len(omitted) < 1:
            continue
        rows.append(
            {
                MODEL_COL: model,
                "n_reported": len(reported),
                "n_omitted": len(omitted),
                "mean_reported": reported.mean(),
                "mean_omitted": omitted.mean(),
                "gap": reported.mean() - omitted.mean(),
                "access": group["Model accessibility"].iloc[0],
                "organization": group["Organization"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def permutation_null(merged, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed = gap_by_model(merged)["gap"].mean()
    null = np.empty(draws)
    shuffled = merged.copy()
    for i in range(draws):
        shuffled["reported"] = (
            merged.groupby(MODEL_COL)["reported"]
            .transform(lambda s: rng.permutation(s.values))
        )
        null[i] = gap_by_model(shuffled)["gap"].mean()
    return observed, null


def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    panel = within_benchmark_percentile(panel)

    coding = load_coding()
    if coding is None:
        print(f"no disclosure coding at {CODING_SHEET}")
        print("the panel is built; the disclosure side is hand-coded.")
        print("copy protocol/disclosure-template.csv to data/disclosures.csv")
        print("and code against protocol/coding-protocol.md, then rerun.")
        print(f"\neligible pairs awaiting coding: {int(panel['eligible'].sum())}")
        print(f"placebo pairs (postdate release): {int(panel['placebo'].sum())}")
        sys.exit(0)

    coding["reported"] = coding["orbit_category"].isin({"A", "B", "C"}).astype(int)
    merged = panel.merge(
        coding, left_on=[MODEL_COL, "slug"],
        right_on=["model_version", "benchmark_slug"], how="inner",
    )

    for label, subset in (
        ("eligible", merged[merged["eligible"]]),
        ("placebo", merged[merged["placebo"]]),
    ):
        gaps = gap_by_model(subset)
        if gaps.empty:
            print(f"\n{label}: no model meets the minimum reported/omitted counts")
            continue
        print(f"\n{label}: n={len(gaps)} models")
        print(f"  mean gap (reported - omitted percentile): {gaps['gap'].mean():+.1f}")
        print(f"  median: {gaps['gap'].median():+.1f}")
        print(f"  share positive: {(gaps['gap'] > 0).mean():.1%}")

    eligible = merged[merged["eligible"]]
    if not gap_by_model(eligible).empty:
        observed, null = permutation_null(eligible)
        p = (np.abs(null) >= abs(observed)).mean()
        print(f"\npermutation test: observed={observed:+.1f}, p={p:.4f}")

    drops = merged[merged["orbit_category"] == "E"]
    print(f"\ndrop cases (ORBIT E): {len(drops)}")

    coverage = merged.groupby(MODEL_COL)["slug"].nunique()
    print(f"models with >={MIN_BENCHMARKS} coded benchmarks: {(coverage >= MIN_BENCHMARKS).sum()}")


if __name__ == "__main__":
    main()
