"""The four falsification tests docs/design.md commits to.

Each is written to be capable of failing. A suite that can only confirm is
decoration, so the pass condition is stated with each test and the result is
printed whatever it says.
"""

import sys

import numpy as np
import pandas as pd

from .config import CODING_SHEET, INTERIM, RELEASE_COL
from .percentiles import add_percentiles
from .selectivity import (
    gap_by_release,
    load_coding,
    merge_coding,
    omission_deficit,
    release_sets,
)
from .stats import bootstrap_mean, ols, print_ols, randomization_test_mean


def permutation_test(merged, draws=2000, seed=0):
    """Relabel which benchmarks were disclosed, at random within release.

    Holding each release's disclosed *count* fixed is what makes this a test of
    selection rather than of table size: the convention explanation in
    design.md says providers report a fixed number of benchmarks, and this null
    grants that and asks only whether *which* ones is random.

    Vectorised over draws; the previous groupby-transform implementation
    rebuilt the frame 2,000 times.
    """
    eligible = merged[merged["group"] == "eligible"].copy()
    codes, releases = pd.factorize(eligible[RELEASE_COL])
    percentile = eligible["percentile"].to_numpy(float)
    reported = eligible["reported"].to_numpy(float)
    n_releases = len(releases)

    def mean_gap(labels):
        disclosed = np.bincount(codes, weights=labels * percentile, minlength=n_releases)
        omitted = np.bincount(codes, weights=(1 - labels) * percentile, minlength=n_releases)
        n_disclosed = np.bincount(codes, weights=labels, minlength=n_releases)
        n_omitted = np.bincount(codes, weights=1 - labels, minlength=n_releases)
        usable = (n_disclosed >= 2) & (n_omitted >= 1)
        if not usable.any():
            return np.nan
        gap = disclosed[usable] / n_disclosed[usable] - omitted[usable] / n_omitted[usable]
        return gap.mean()

    observed = mean_gap(reported)

    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    grouped = reported[order]
    boundaries = np.searchsorted(codes[order], np.arange(n_releases + 1))

    null = np.empty(draws)
    for i in range(draws):
        shuffled = grouped.copy()
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            rng.shuffle(shuffled[start:stop])
        restored = np.empty_like(shuffled)
        restored[order] = shuffled
        null[i] = mean_gap(restored)

    p = float((np.abs(null) >= abs(observed)).mean())
    return observed, null, p


def excess_omission(sets):
    """Does the omitted set sit below what the disclosed set predicts?

    design.md: "If omission is strategic, omitted-benchmark percentiles should
    sit below what the model's disclosed benchmarks predict. If innocent, they
    should be unrelated to the model's level."

    Fits mean_omitted = a + b * mean_disclosed. Under innocent omission the
    omitted benchmarks are a random slice of the release's standing, so b ~ 1
    and a ~ 0. Strategic omission pushes a below zero: the same model, held at
    the same disclosed level, does worse on what it withheld.
    """
    usable = sets.dropna(subset=["mean_disclosed", "mean_omitted"])
    if len(usable) < 10:
        return None
    X = np.column_stack([np.ones(len(usable)), usable["mean_disclosed"].to_numpy(float)])
    return ols(
        usable["mean_omitted"].to_numpy(float), X,
        ["intercept (a<0 => strategic)", "slope on disclosed level"],
        cluster=usable["organization"].to_numpy(),
    )


def reverse_gap(coding):
    """Benchmarks a provider reports that the independent source does not cover.

    design.md keeps these rather than discarding them: under a pure concealment
    story that set should not be systematically favourable. A provider reaching
    for benchmarks Epoch does not run is doing something, and it is not
    concealment.
    """
    if "reverse_gap" not in coding.columns:
        return None
    flagged = coding[
        coding["reverse_gap"].notna() & (coding["reverse_gap"].astype(str).str.strip() != "")
    ]
    return flagged


def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    panel = add_percentiles(panel)

    coding = load_coding()
    if coding is None:
        print(f"no disclosure coding at {CODING_SHEET}; nothing to falsify yet.")
        print("run `python -m src.worklist` and code against the protocol.")
        sys.exit(0)

    merged = merge_coding(panel, coding)
    if "group" not in merged.columns:
        merged["group"] = "eligible"
    sets = release_sets(merged)

    print("=" * 62)
    print("1. PLACEBO -- omitted vs postdating benchmarks")
    print("   this contrast does NOT have a null of zero. Measured on the panel")
    print("   with no disclosure labels it sits at +12.2 percentile points, a")
    print("   peer-window artifact. Read any value here against that, never")
    print("   against zero. See src/placebo_calibration.py.")
    deficit = omission_deficit(sets)
    if deficit.empty:
        print("   not computable: no release has both an omitted and a placebo cell")
    else:
        values = deficit["gap"].to_numpy(float)
        providers = deficit["organization"].to_numpy()
        mean, (lo, hi) = bootstrap_mean(values, cluster=providers)
        test = randomization_test_mean(values, cluster=providers)
        if np.isfinite(lo):
            print(f"   n={len(deficit)}  mean {mean:+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
            # an interval that is absent is not an interval that spans zero
            print(f"   -> {'excludes' if hi < 0 or lo > 0 else 'includes'} zero")
        else:
            print(f"   n={len(deficit)}  mean {mean:+.1f}  "
                  f"(no interval: {test['n_clusters']} provider(s))")
        print(f"   randomization test, provider sign-flip: p = {test['p_value']:.3f}")

    print("\n" + "=" * 62)
    print("2. EXCESS OMISSION -- omitted vs what disclosed predicts")
    fit = excess_omission(sets)
    if fit is None:
        print("   not computable: fewer than 10 releases with both sets")
    else:
        print_ols(fit, "   mean_omitted ~ mean_disclosed")

    print("\n" + "=" * 62)
    print("3. PERMUTATION -- relabel which benchmarks were disclosed")
    print("   holds each release's disclosed count fixed, so the convention")
    print("   explanation is granted and only selection is tested.")
    eligible = merged[merged["group"] == "eligible"]
    if len(eligible) < 20 or eligible["reported"].nunique() < 2:
        print("   not computable: too few coded cells")
    else:
        observed, null, p = permutation_test(merged)
        print(f"   observed {observed:+.1f}   null mean {np.nanmean(null):+.2f}"
              f"   sd {np.nanstd(null):.2f}   p={p:.4f}")

    print("\n" + "=" * 62)
    print("4. REVERSE GAP -- provider reports what the independent source lacks")
    flagged = reverse_gap(coding)
    if flagged is None or flagged.empty:
        print("   no reverse_gap entries recorded yet")
    else:
        print(f"   {len(flagged)} recorded across "
              f"{flagged['release_id'].nunique()} releases")


if __name__ == "__main__":
    main()
