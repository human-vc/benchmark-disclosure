
import sys

import numpy as np
import pandas as pd

from .config import CODING_SHEET, INTERIM, MIN_BENCHMARKS, RELEASE_COL, WINDOW_DAYS
from .percentiles import add_percentiles, capability_level
from .stats import bootstrap_mean, ols, print_ols

REPORTED_CATEGORIES = {"A", "B", "C"}
HIGH_SUSPICION = {"D", "E", "G"}

def load_coding():
    if not CODING_SHEET.exists():
        return None
    coding = pd.read_csv(CODING_SHEET)
    coded = coding["orbit_category"].notna() & (coding["orbit_category"] != "")
    prefilled = coding.get("coder", pd.Series(index=coding.index)).eq("auto")
    usable = coding[coded | prefilled]
    return usable if len(usable) else None

def merge_coding(panel, coding):
    coding = coding.copy()
    is_coded = coding["orbit_category"].notna() & (coding["orbit_category"] != "")
    coding["reported"] = np.where(
        is_coded,
        coding["orbit_category"].isin(REPORTED_CATEGORIES).astype(float),
        pd.to_numeric(coding.get("reported"), errors="coerce"),
    )
    keep = [
        "release_id", "benchmark_slug", "orbit_category", "reported", "group",
        "prior_release", "family_id", "family_rank",
    ]
    keep = [c for c in keep if c in coding.columns]
    return panel.merge(
        coding[keep].dropna(subset=["reported"]),
        left_on=[RELEASE_COL, "slug"],
        right_on=["release_id", "benchmark_slug"],
        how="inner",
        validate="1:1",
    )

def release_sets(merged):
    merged = merged.copy()
    merged["set"] = np.where(
        merged["group"] == "placebo",
        "placebo",
        np.where(merged["reported"] == 1, "disclosed", "omitted"),
    )
    wide = (
        merged.pivot_table(
            index=RELEASE_COL, columns="set", values="percentile", aggfunc="mean"
        )
        .rename(columns={c: f"mean_{c}" for c in ("disclosed", "omitted", "placebo")})
    )
    counts = (
        merged.pivot_table(
            index=RELEASE_COL, columns="set", values="percentile", aggfunc="size"
        )
        .rename(columns={c: f"n_{c}" for c in ("disclosed", "omitted", "placebo")})
        .fillna(0)
    )
    meta = merged.groupby(RELEASE_COL).agg(
        organization=("Organization", "first"),
        access=("Model accessibility", "first"),
        n_scored=("slug", "nunique"),
        level=("level", "first") if "level" in merged.columns else ("percentile", "mean"),
    )
    out = meta.join(wide).join(counts).reset_index()
    for column in ("mean_disclosed", "mean_omitted", "mean_placebo",
                   "n_disclosed", "n_omitted", "n_placebo"):
        if column not in out.columns:
            out[column] = np.nan if column.startswith("mean") else 0
    return out

def gap_by_release(sets, against="omitted", min_disclosed=2, min_other=1):
    keep = (
        (sets["n_disclosed"] >= min_disclosed)
        & (sets[f"n_{against}"] >= min_other)
        & sets["mean_disclosed"].notna()
        & sets[f"mean_{against}"].notna()
    )
    gaps = sets[keep].copy()
    gaps["gap"] = gaps["mean_disclosed"] - gaps[f"mean_{against}"]
    return gaps

def omission_deficit(sets, min_omitted=1, min_placebo=1):
    keep = (
        (sets["n_omitted"] >= min_omitted)
        & (sets["n_placebo"] >= min_placebo)
        & sets["mean_omitted"].notna()
        & sets["mean_placebo"].notna()
    )
    out = sets[keep].copy()
    out["gap"] = out["mean_omitted"] - out["mean_placebo"]
    return out

def drop_estimator(merged):
    if "orbit_category" not in merged.columns:
        return pd.DataFrame()
    rows = []
    for release, group in merged.groupby(RELEASE_COL):
        dropped = group[group["orbit_category"] == "E"]
        kept = group[group["reported"] == 1]
        if dropped.empty or len(kept) < 2:
            continue
        rows.append({
            RELEASE_COL: release,
            "n_dropped": len(dropped),
            "n_kept": len(kept),
            "mean_dropped": dropped["percentile"].mean(),
            "mean_kept": kept["percentile"].mean(),
            "drop_gap": dropped["percentile"].mean() - kept["percentile"].mean(),
            "organization": group["Organization"].iloc[0],
            "benchmarks": " | ".join(sorted(dropped["slug"])),
        })
    return pd.DataFrame(rows)

def access_group(value):
    if not isinstance(value, str):
        return None
    if value.startswith("Open weights"):
        return "open"
    if value == "API access":
        return "api"
    return None

def report_gaps(gaps, label, contrast="disclosed - omitted"):
    if gaps.empty:
        print(f"\n{label}: no release meets the minimum reported/omitted counts")
        return None
    mean, (lo, hi) = bootstrap_mean(
        gaps["gap"].values, cluster=gaps["organization"].values
    )
    print(f"\n{label}: n={len(gaps)} releases, {gaps['organization'].nunique()} providers")
    print(f"  mean gap ({contrast} percentile): {mean:+.1f}  "
          f"95% CI [{lo:+.1f}, {hi:+.1f}]  (provider-clustered bootstrap)")
    print(f"  median: {gaps['gap'].median():+.1f}   share positive: {(gaps['gap'] > 0).mean():.1%}")
    return mean

def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    panel = add_percentiles(panel)

    coding = load_coding()
    if coding is None:
        print(f"no disclosure coding at {CODING_SHEET}")
        print("the panel is built; the disclosure side is hand-coded.")
        print("run `python -m src.worklist`, copy data/worklist.csv to")
        print(f"{CODING_SHEET}, and code against protocol/coding-protocol.md.")
        print(f"\neligible pairs awaiting coding: {int(panel['eligible'].sum())}")
        print(f"placebo pairs (postdate release): {int(panel['placebo'].sum())}")
        sys.exit(0)

    merged = merge_coding(panel, coding)
    merged["level"] = merged[RELEASE_COL].map(capability_level(panel))
    merged["access"] = merged["Model accessibility"].map(access_group)

    print(f"contemporaneity window: +/-{WINDOW_DAYS} days")
    print(f"coded cells merged onto the panel: {len(merged)}")

    if "group" not in merged.columns:
        merged["group"] = "eligible"
    sets = release_sets(merged)

    gaps = gap_by_release(sets, against="omitted")
    report_gaps(gaps, "descriptive: disclosed vs omitted")
    print("  descriptive only: picks up any reason non-reported benchmarks score")
    print("  low, strategic or not.")

    deficit = omission_deficit(sets)
    report_gaps(
        deficit,
        "identifying: omitted vs placebo (zero under every innocent explanation)",
        contrast="omitted-eligible - postdating",
    )
    print("  negative = the provider withheld where it stood worse. Both sets are")
    print("  non-disclosures, so neither carries the disclosed set's selection.")

    if not gaps.empty:
        gaps["access_open"] = (gaps["access"].map(access_group) == "open").astype(float)
        X = np.column_stack([
            np.ones(len(gaps)),
            gaps["n_scored"].values,
            gaps["level"].values,
            gaps["access_open"].values,
        ])
        print_ols(
            ols(gaps["gap"].values, X,
                ["const", "n independent scores", "capability level",
                 "open weights"],
                cluster=gaps["organization"].values),
            "gap conditioned on coverage, capability and access type",
        )
        print("  coverage is why raw open-vs-closed rates are not comparable;")
        print("  capability level is the conditioning design.md asks for. The")
        print("  open-weights term is the secondary result and may well be null.")

    eligible = merged[merged["group"] == "eligible"]
    drops = drop_estimator(eligible)
    if drops.empty:
        n_e = int((eligible.get("orbit_category") == "E").sum()) if len(eligible) else 0
        print(f"\ndrop estimator (ORBIT E): {n_e} E-coded cells, "
              "not enough for a release-level estimate yet")
    else:
        mean, (lo, hi) = bootstrap_mean(
            drops["drop_gap"].values, cluster=drops["organization"].values
        )
        print(f"\ndrop estimator (ORBIT E): n={len(drops)} releases, "
              f"{int(drops['n_dropped'].sum())} dropped benchmarks")
        print(f"  mean (dropped - kept percentile): {mean:+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
        print("  negative means the provider stopped reporting where it looks worse.")

    coverage = merged.groupby(RELEASE_COL)["slug"].nunique()
    print(f"\nreleases with >={MIN_BENCHMARKS} coded benchmarks: "
          f"{(coverage >= MIN_BENCHMARKS).sum()}")

if __name__ == "__main__":
    main()
