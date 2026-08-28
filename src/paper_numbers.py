"""Every number the manuscript cites, regenerated from one command."""

import json
from datetime import date

import numpy as np
import pandas as pd

from .config import INTERIM, MIN_BENCHMARKS, RELEASE_COL, WINDOW_DAYS, WORKLIST
from .helm_external import FROZEN as HELM_FROZEN
from .helm_external import summary as helm_summary
from .percentiles import side_balanced_percentile, within_benchmark_percentile
from .sensitivity import BASELINE, leave_one_org_out, sweep
from .stats import randomization_test_mean
from .placebo_calibration import (
    contrast_by_release,
    decomposition,
    drop_capacity,
    null_calibration,
    permutation_null,
    trend_recovery,
)
from .snapshot import compare, stamp


def _contrast(panel, column):
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(subset=[column])
    wide = (
        cells.assign(side=np.where(cells["eligible"], "eligible", "placebo"))
        .pivot_table(index=RELEASE_COL, columns="side", values=column, aggfunc="mean")
        .dropna()
    )
    if not {"eligible", "placebo"}.issubset(wide.columns):
        return {"mean": None, "median": None, "share_positive": None, "n_releases": 0}
    gap = wide["eligible"] - wide["placebo"]
    return {
        "mean": round(float(gap.mean()), 3),
        "median": round(float(gap.median()), 3),
        "share_positive": round(float((gap > 0).mean()), 4),
        "n_releases": int(len(gap)),
    }


def _asymmetry_slope(panel, column):
    cells = panel[panel["eligible"]].dropna(subset=[column, "share_newer"])
    demean = lambda values: (
        values - values.groupby(cells[RELEASE_COL]).transform("mean")
    ).to_numpy()
    y, x = demean(cells[column]), demean(cells["share_newer"])
    return round(float(np.cov(y, x, bias=True)[0, 1] / x.var()), 3)


def _growth_and_saturation(panel):
    """Section 3's claim and Section 4's mechanism, both checkable."""
    dated = panel.dropna(subset=["benchmark_release_date"])
    stock = (dated[["slug", "benchmark_release_date"]].drop_duplicates("slug")
             .assign(year=lambda f: f["benchmark_release_date"].dt.year)
             .groupby("year").size().sort_index().cumsum())

    rows = []
    for slug, group in panel.groupby("slug"):
        values = group["score"].dropna()
        if len(values) >= 10 and values.max() > values.min():
            share = float((((values - values.min())
                            / (values.max() - values.min())) > 0.9).mean())
            date = group["benchmark_release_date"].dropna()
            if len(date):
                rows.append((date.iloc[0], share))
    frame = pd.DataFrame(rows, columns=["date", "top_decile_share"])
    late = frame["date"] >= frame["date"].median()
    from scipy.stats import mannwhitneyu
    stat = mannwhitneyu(frame[~late]["top_decile_share"], frame[late]["top_decile_share"])
    eligible = panel[panel["eligible"]]
    per_release = (eligible.groupby([RELEASE_COL, "Release date"]).size()
                   .reset_index(name="n"))
    half = per_release["Release date"].dt.year.astype(str) + "H" + (
        (per_release["Release date"].dt.month > 6).astype(int) + 1).astype(str)
    series = per_release.groupby(half)["n"].mean().round(3)

    return {
        "dated_stock_by_year": {int(k): int(v) for k, v in stock.items()},
        "eligible_per_release_by_half": {k: float(v) for k, v in series.items()},
        "saturation": {
            "n_benchmarks": int(len(frame)),
            "early_top_decile_share": round(float(frame[~late]["top_decile_share"].mean()), 4),
            "late_top_decile_share": round(float(frame[late]["top_decile_share"].mean()), 4),
            "mannwhitney_p": round(float(stat.pvalue), 4),
        },
    }


def _worklist_reach():
    """What the coding worklist asks a human to read."""
    sheet = pd.read_csv(WORKLIST, dtype=str).fillna("")
    eligible = sheet[sheet["group"] == "eligible"]
    return {
        "cells": int(len(eligible)),
        "releases": int(eligible["release_id"].nunique()),
        "families": int(eligible["family_id"].nunique()),
    }


def _sensitivity(panel):
    """The specification grid and the leave-one-organisation-out range."""
    frame = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    table = sweep(frame)
    dropped = leave_one_org_out(panel)
    dropped = dropped[dropped["dropped"] != "(none)"]
    at_baseline = table[table["min_benchmarks"] == BASELINE["min_benchmarks"]]
    return {
        "specifications": int(len(table)),
        "positive": int((table["mean"] > 0).sum()),
        "min": round(float(table["mean"].min()), 3),
        "max": round(float(table["mean"].max()), 3),
        "share_newer_gap_correlation": round(float(
            np.corrcoef(at_baseline["share_newer_gap"], at_baseline["mean"])[0, 1]), 3),
        "leave_one_out_min": round(float(dropped["mean"].min()), 3),
        "leave_one_out_max": round(float(dropped["mean"].max()), 3),
        "organisations_dropped": int(len(dropped)),
    }


def collect():
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    panel = side_balanced_percentile(within_benchmark_percentile(panel))
    families = pd.read_csv("data/families.csv")

    eligible_or_placebo = panel[panel["eligible"] | panel["placebo"]]
    ladder = decomposition(panel).set_index("step")

    numbers = {
        "panel": {
            "benchmarks": int(panel["slug"].nunique()),
            "releases": int(panel[RELEASE_COL].nunique()),
            "pairs": int(len(panel)),
            "eligible": int(panel["eligible"].sum()),
            "placebo": int(panel["placebo"].sum()),
            "window_days": WINDOW_DAYS,
        },
        "placebo_null": {
            "windowed_percentile": _contrast(panel, "percentile"),
            "side_balanced": _contrast(panel, "pct_balanced"),
        },
        "asymmetry": {
            "share_newer_eligible": round(
                float(panel[panel["eligible"]]["share_newer"].mean()), 4),
            "share_newer_placebo": round(
                float(panel[panel["placebo"]]["share_newer"].mean()), 4),
            "slope_windowed": _asymmetry_slope(panel, "percentile"),
            "slope_balanced": _asymmetry_slope(panel, "pct_balanced"),
        },
        "decomposition": {
            step: {
                "coef": round(float(row["placebo_coef"]), 3),
                "se": round(float(row["se"]), 3),
                "t": round(float(row["t"]), 3),
                "se_provider": round(float(row["se_provider"]), 3),
                "absorbed": round(float(row["share_absorbed"]), 4),
                "n_clusters": int(row["n_clusters"]),
                "n_benchmarks": int(row["n_benchmarks"]),
            }
            for step, row in ladder.iterrows()
        },
        "balanced_coverage": {
            "defined_share_all_cells": round(
                float(panel["pct_balanced"].notna().mean()), 4),
            "defined_share_eligible_or_placebo": round(
                float(eligible_or_placebo["pct_balanced"].notna().mean()), 4),
        },
        "drop_capacity": drop_capacity(panel, families).to_dict("records"),
        "null_calibration": null_calibration(panel).to_dict("records"),
        "min_benchmarks": MIN_BENCHMARKS,
        "opportunity": _growth_and_saturation(panel),
        "worklist": _worklist_reach(),
        "trend_recovery": trend_recovery(panel),
        "sensitivity": _sensitivity(panel),
    }

    # The decomposition lands on `pct_sided`, the windowed percentile taken
    # over peers only, not on `percentile`, which also carries the focal
    # model's own midrank contribution. Under a convention that folds the model
    # into its own older side the identity holds against `percentile` exactly,
    # but only because the model is counted among its own peers; see
    # src/percentiles.py. Checking it against `pct_sided` is the arithmetic
    # check it was meant to be.
    identity = (
        (1 - panel["share_newer"]) * panel["pct_old_side"].fillna(0)
        + panel["share_newer"] * panel["pct_new_side"].fillna(0)
    )
    defined = panel["pct_sided"].notna()
    numbers["identity_max_abs_error"] = float(
        np.nanmax(np.abs(identity[defined] - panel["pct_sided"][defined]))
    )
    # What the focal model's own inclusion is worth, cell by cell. It is the
    # gap between the reported measure and the one the identity decomposes, and
    # it is reported rather than absorbed.
    numbers["self_inclusion_max_abs_gap"] = float(
        np.nanmax(np.abs(panel["percentile"][defined] - panel["pct_sided"][defined]))
    )
    return panel, numbers


def main(draws=300):
    panel, numbers = collect()

    contrast = contrast_by_release(panel)
    headline = randomization_test_mean(contrast["contrast"].to_numpy(),
                                       cluster=contrast["organization"].to_numpy(),
                                       draws=9999)
    numbers["placebo_null"]["randomization"] = {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in headline.items()}

    result = permutation_null(panel, draws=draws)
    numbers["permutation_null"] = {
        key: (round(value, 4) if isinstance(value, float) else value)
        for key, value in result.items()
    }

    if HELM_FROZEN.exists():
        numbers["helm_lite"] = helm_summary()

    pinned = compare()
    numbers["_provenance"] = {
        "snapshot": stamp(),
        "snapshot_status": pinned["status"],
        "index_rows": pinned.get("current_index_rows"),
        "generated": date.today().isoformat(),
    }

    from .config import CODING_SHEET
    if CODING_SHEET.exists():
        numbers["coding"] = coding_block()

    out = INTERIM.parent / "paper_numbers.json"
    out.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")

    print(numbers["_provenance"]["snapshot"])
    if pinned["status"] != "match":
        print("  REFUSE TO CITE THESE NUMBERS: the data on disk is not the pinned build")
    print(f"wrote {out}")
    print(f"  placebo null, windowed  : {numbers['placebo_null']['windowed_percentile']['mean']:+.2f}")
    print(f"  placebo null, balanced  : {numbers['placebo_null']['side_balanced']['mean']:+.2f}")
    print(f"  fully conditioned       : {numbers['decomposition']['two-way FE + both']['coef']:+.2f} "
          f"({numbers['decomposition']['two-way FE + both']['se']:.2f})")
    print(f"  permutation null        : {numbers['permutation_null']['null_mean']:+.2f} "
          f"vs observed {numbers['permutation_null']['observed']:+.2f}, "
          f"{numbers['permutation_null']['sd_away']:.1f} sd, "
          f"{numbers['permutation_null']['draws_at_least_as_extreme']}/{draws} draws")
    print(f"  decomposition identity  : max abs error {numbers['identity_max_abs_error']:.2e}")
    if "helm_lite" in numbers:
        helm = numbers["helm_lite"]
        print(f"  HELM Lite frozen cells  : {helm['frozen_cells']['cells']} cells, "
              f"{helm['frozen_cells']['changed']} changed, "
              f"{helm['headline']['moved']}/{helm['headline']['models']} win rates moved, "
              f"{helm['headline']['reversals_endpoint']} pairs reordered")


def coding_block(kappa_draws=2000):
    """Every number Section 5 quotes, from the coded sheet and the pinned panel.

    Deficits are computed beside the label-free contrast on the same measure
    and window, because the section's claim is their difference, not either
    level. Reliability follows the frozen plan: derived cells, pre-adjudication,
    intervals from resampling releases rather than cells.
    """
    from .falsification import eligible_vs_placebo, permutation_test
    from .percentiles import add_percentiles
    from .reliability import (ARTIFACTS, SECOND_EVIDENCE, cohens_kappa,
                              derive_from, HIGH_SUSPICION, REPORTED)
    from .selectivity import (drop_estimator, gap_by_release, load_coding,
                              merge_coding, omission_deficit, release_sets)
    from .stats import bootstrap_mean, randomization_test_mean

    raw = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    coding = load_coding()

    def contrast_pair(window, value):
        scored = add_percentiles(raw.copy(), window_days=window)
        merged = merge_coding(scored, coding)
        merged["percentile"] = merged[value]
        deficit = omission_deficit(release_sets(merged))
        mean, (lo, hi) = bootstrap_mean(deficit["gap"].values,
                                        cluster=deficit["organization"].values)
        test = randomization_test_mean(deficit["gap"].values,
                                       cluster=deficit["organization"].values)
        dated = scored.copy()
        dated["group"] = np.where(dated["eligible"], "eligible",
                                  np.where(dated["placebo"], "placebo", "unknown"))
        null = eligible_vs_placebo(dated, value)["gap"].mean()
        return {"deficit": round(mean, 2), "low": round(lo, 2), "high": round(hi, 2),
                "n_releases": int(len(deficit)),
                "n_providers": int(deficit["organization"].nunique()),
                "sign_flip_p": round(test["p_value"], 4), "null": round(null, 2)}

    specs = {
        "windowed_182": contrast_pair(182, "percentile"),
        "side_balanced_182": contrast_pair(182, "pct_balanced"),
        "alltime": contrast_pair(182, "percentile_alltime"),
        "windowed_90": contrast_pair(90, "percentile"),
        "windowed_365": contrast_pair(365, "percentile"),
    }

    scored = add_percentiles(raw.copy())
    merged = merge_coding(scored, coding)
    eligible = merged[merged["group"] == "eligible"]
    theta_release = eligible.groupby(RELEASE_COL).agg(
        rate=("reported", lambda s: 1 - s.mean()),
        organization=("Organization", "first"))
    theta_mean, (theta_lo, theta_hi) = bootstrap_mean(
        theta_release["rate"].values, cluster=theta_release["organization"].values)

    gaps = gap_by_release(release_sets(merged), against="omitted")
    gap_mean, (gap_lo, gap_hi) = bootstrap_mean(
        gaps["gap"].values, cluster=gaps["organization"].values)

    observed, null_draws, perm_p = permutation_test(merged)
    drops = drop_estimator(eligible)
    drop_test = randomization_test_mean(drops["drop_gap"].values,
                                        cluster=drops["organization"].values)

    artifacts = pd.read_csv(ARTIFACTS, dtype=str)
    sheet = pd.read_csv(SECOND_EVIDENCE, dtype=str)
    sheet = sheet[sheet["reported_slugs"].notna()]
    done = set(sheet["release_id"])
    first = artifacts[artifacts["release_id"].isin(done)
                      & (artifacts["fetch_status"] == "ok")]
    paired = derive_from(first).merge(derive_from(sheet),
                                      on=["release_id", "benchmark_slug"],
                                      suffixes=("_1", "_2"))
    a, b = paired["orbit_category_1"], paired["orbit_category_2"]

    def three_kappas(frame):
        aa, bb = frame["orbit_category_1"], frame["orbit_category_2"]
        rep, _ = cohens_kappa(aa.isin(REPORTED).map({True: "r", False: "n"}),
                              bb.isin(REPORTED).map({True: "r", False: "n"}))
        full, _ = cohens_kappa(aa, bb)
        high, _ = cohens_kappa(aa.isin(HIGH_SUSPICION).map({True: "h", False: "l"}),
                               bb.isin(HIGH_SUSPICION).map({True: "h", False: "l"}))
        return rep, full, high

    releases = paired["release_id"].unique()
    groups = {r: g for r, g in paired.groupby("release_id")}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(kappa_draws):
        pick = rng.choice(releases, size=len(releases), replace=True)
        try:
            draws.append(three_kappas(pd.concat([groups[r] for r in pick])))
        except Exception:
            continue
    draws = np.array(draws)
    point = three_kappas(paired)
    names = ("reported", "full", "high_low")
    reliability = {}
    for i, name in enumerate(names):
        lo, hi = np.nanpercentile(draws[:, i], [2.5, 97.5])
        reliability[name] = {"kappa": round(point[i], 3),
                             "low": round(lo, 3), "high": round(hi, 3)}
    reliability["paired_cells"] = int(len(paired))
    reliability["paired_releases"] = int(paired["release_id"].nunique())
    reliability["disagreements"] = int((a != b).sum())

    counts = coding[coding["orbit_category"].notna()
                    & (coding["orbit_category"] != "")
                    & (coding.get("coder") != "auto")]
    return {
        "coverage": {
            "coded_cells": int(len(counts)),
            "categories": counts["orbit_category"].value_counts().to_dict(),
            "eligible_in_panel": int(len(eligible)),
            "excluded_off_worklist": int(len(counts)) + int((coding.get("coder") == "auto").sum()) - int(len(merged)),
        },
        "theta": {"cells": round(1 - eligible["reported"].mean(), 4),
                  "release_mean": round(theta_mean, 4),
                  "low": round(theta_lo, 4), "high": round(theta_hi, 4),
                  "n_providers": int(theta_release["organization"].nunique())},
        "disclosed_minus_omitted": {"mean": round(gap_mean, 2),
                                    "low": round(gap_lo, 2), "high": round(gap_hi, 2),
                                    "n_releases": int(len(gaps))},
        "deficit_by_spec": specs,
        "permutation": {"observed": round(float(observed), 2),
                        "null_sd": round(float(np.nanstd(null_draws)), 2),
                        "p": round(perm_p, 4)},
        "drop": {"n_releases": int(len(drops)),
                 "n_drops": int(drops["n_dropped"].sum()),
                 "mean": round(drop_test["mean"], 2),
                 "sign_flip_p": round(drop_test["p_value"], 3),
                 "p_floor": round(drop_test["p_floor"], 3),
                 "n_providers": drop_test["n_clusters"]},
        "reliability": reliability,
    }


if __name__ == "__main__":
    main()
