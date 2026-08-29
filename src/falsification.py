
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

def eligible_vs_placebo(panel, value="percentile", min_each=1):
    usable = panel[panel[value].notna() & panel["group"].isin({"eligible", "placebo"})]
    wide = (
        usable.pivot_table(index=RELEASE_COL, columns="group", values=value,
                           aggfunc="mean")
        .rename(columns={"eligible": "mean_eligible", "placebo": "mean_placebo"})
    )
    counts = (
        usable.pivot_table(index=RELEASE_COL, columns="group", values=value,
                           aggfunc="size")
        .rename(columns={"eligible": "n_eligible", "placebo": "n_placebo"})
        .fillna(0)
    )
    org = usable.groupby(RELEASE_COL)["Organization"].first()
    out = wide.join(counts).join(org).reset_index()
    for column in ("mean_eligible", "mean_placebo", "n_eligible", "n_placebo"):
        if column not in out.columns:
            out[column] = np.nan
    out = out[(out["n_eligible"] >= min_each) & (out["n_placebo"] >= min_each)
              & out["mean_eligible"].notna() & out["mean_placebo"].notna()].copy()
    out["gap"] = out["mean_eligible"] - out["mean_placebo"]
    return out

def permutation_test(merged, draws=9999, seed=0):
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

    k = int((np.abs(null) >= abs(observed)).sum())
    p = (k + 1) / (draws + 1)
    return observed, null, p

def excess_omission(sets):
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
    if "reverse_gap" not in coding.columns:
        return None
    flagged = coding[
        coding["reverse_gap"].notna() & (coding["reverse_gap"].astype(str).str.strip() != "")
    ]
    return flagged

def placebo_under_each_measure(panel):
    rows = []
    variants = [
        ("windowed percentile (under audit)", "percentile", None),
        ("rank against all models ever", "percentile_alltime", None),
        ("trim to two-sided windows", "percentile", "two_sided"),
        ("side-balanced percentile", "pct_balanced", None),
    ]
    total = len(panel[panel["group"].isin({"eligible", "placebo"})])
    for label, value, trim in variants:
        if value not in panel.columns:
            continue
        subset = panel
        if trim == "two_sided":
            subset = panel[panel["pct_old_side"].notna()
                           & panel["pct_new_side"].notna()]
        gaps = eligible_vs_placebo(subset, value)
        if gaps.empty:
            continue
        used = subset[subset["group"].isin({"eligible", "placebo"})
                      & subset[value].notna()]
        rows.append({
            "measure": label,
            "mean": float(gaps["gap"].mean()),
            "median": float(gaps["gap"].median()),
            "positive": float((gaps["gap"] > 0).mean()),
            "releases": len(gaps),
            "cell_share": len(used) / total if total else float("nan"),
        })
    return pd.DataFrame(rows)

def report_label_free_placebo(panel):
    print("\n" + "=" * 62)
    print("0. PLACEBO WITHOUT LABELS -- available vs postdating benchmarks")
    print("   uses no disclosure coding at all: both sets are defined by")
    print("   dates alone. Zero under every innocent explanation.")
    for value, label in (("percentile", "windowed percentile"),
                         ("pct_balanced", "side-balanced percentile")):
        if value not in panel.columns:
            continue
        gaps = eligible_vs_placebo(panel, value)
        if gaps.empty:
            print(f"   {label}: no release carries both kinds of cell")
            continue
        mean, (lo, hi) = bootstrap_mean(
            gaps["gap"].to_numpy(float), cluster=gaps["Organization"].to_numpy()
        )
        print(f"   {label}: n={len(gaps)} releases, "
              f"{gaps['Organization'].nunique()} providers")
        print(f"     mean {mean:+.2f}  median {gaps['gap'].median():+.2f}  "
              f"positive {(gaps['gap'] > 0).mean():.1%}  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]")

    table = placebo_under_each_measure(panel)
    if not table.empty:
        print("\n   every candidate repair, same contrast (all should be zero):")
        print(f"   {'measure':38} {'mean':>7} {'median':>7} {'pos':>6} "
              f"{'rel':>5} {'cells':>7}")
        for row in table.itertuples():
            print(f"   {row.measure:38} {row.mean:+7.2f} {row.median:+7.2f} "
                  f"{row.positive:5.1%} {row.releases:5d} {row.cell_share:6.1%}")

    shares = panel.groupby("group")["share_newer"].mean()
    if {"eligible", "placebo"} <= set(shares.index):
        print(f"   peer windows: mean share of the window newer than the focal "
              f"model is {shares['eligible']:.4f} for available cells and "
              f"{shares['placebo']:.4f} for postdating ones, against 0.5 for a "
              f"two-sided window.")

def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    panel = add_percentiles(panel)
    dated = panel.copy()
    dated["group"] = np.where(dated["eligible"], "eligible",
                              np.where(dated["placebo"], "placebo", "unknown"))
    report_label_free_placebo(dated)

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
