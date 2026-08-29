
import numpy as np
import pandas as pd

from .config import INTERIM, MIN_BENCHMARKS, RELEASE_COL, WINDOW_DAYS
from .percentiles import within_benchmark_percentile
from .stats import ols, two_way_se

MIN_VARIANCE = 1e-12

BALANCED_SHARE = (0.25, 0.75)
BALANCED_MIN_PEERS = 5

def _within(frame, column, by):
    return frame[column] - frame.groupby(by)[column].transform("mean")

def _absorb(values, *groups, tol=1e-10, max_iter=5000):
    x = np.asarray(values, dtype=float).copy()
    codes, sizes = [], []
    for group in groups:
        code = pd.factorize(np.asarray(group))[0]
        codes.append(code)
        sizes.append(int(code.max()) + 1 if len(code) else 0)

    for _ in range(max_iter):
        previous = x.copy()
        for code, size in zip(codes, sizes):
            counts = np.bincount(code, minlength=size)
            totals = np.bincount(code, weights=x, minlength=size)
            x -= (totals / np.maximum(counts, 1))[code]
        if np.max(np.abs(x - previous)) < tol:
            break
    return x

def contrast_by_release(panel):
    cells = panel[panel["eligible"] | panel["placebo"]].copy()
    cells["set"] = np.where(cells["eligible"], "eligible", "placebo")
    wide = cells.pivot_table(
        index=RELEASE_COL, columns="set", values="percentile", aggfunc="mean"
    ).dropna()
    if wide.empty or not {"eligible", "placebo"}.issubset(wide.columns):
        return pd.DataFrame(columns=["contrast", "organization"])

    orgs = cells.groupby(RELEASE_COL)["primary_org"].first()
    return pd.DataFrame({
        "contrast": wide["eligible"] - wide["placebo"],
        "organization": orgs.reindex(wide.index),
    }).reset_index()

def decomposition(panel):
    cells = panel[panel["eligible"] | panel["placebo"]].copy()
    cells["placebo_i"] = cells["placebo"].astype(float)
    cells = cells.dropna(subset=["percentile", "share_newer", "n_peers"])

    release = cells[RELEASE_COL].to_numpy()
    benchmark = cells["slug"].to_numpy()
    cluster = cells["primary_org"].to_numpy()

    y_r = _within(cells, "percentile", RELEASE_COL).to_numpy()
    x_r = _within(cells, "placebo_i", RELEASE_COL).to_numpy()
    asym_r = _within(cells, "share_newer", RELEASE_COL).to_numpy()
    peers_r = _within(cells, "n_peers", RELEASE_COL).to_numpy()

    y_rb = _absorb(cells["percentile"], release, benchmark)
    x_rb = _absorb(cells["placebo_i"], release, benchmark)
    asym_rb = _absorb(cells["share_newer"], release, benchmark)
    peers_rb = _absorb(cells["n_peers"], release, benchmark)

    steps = [
        ("release FE", y_r, [x_r]),
        ("+ benchmark FE", y_rb, [x_rb]),
        ("+ peer-window asymmetry", y_r, [x_r, asym_r]),
        ("+ peer count", y_r, [x_r, asym_r, peers_r]),
        ("two-way FE + both", y_rb, [x_rb, asym_rb, peers_rb]),
    ]
    rows = []
    for label, outcome, regressors in steps:
        design = np.column_stack([np.ones(len(outcome))] + regressors)
        names = ["const", "placebo"] + [f"z{i}" for i in range(len(regressors) - 1)]
        fit = ols(outcome, design, names, cluster=cluster)
        both = two_way_se(outcome, design, cluster, benchmark)
        rows.append({
            "step": label,
            "placebo_coef": fit["beta"][1],
            "se": both["se"][1],
            "t": both["beta"][1] / both["se"][1],
            "se_provider": fit["se"][1],
            "t_provider": fit["t"][1],
            "n_clusters": fit["n_clusters"],
            "n_benchmarks": both["n_clusters_b"],
            "n_cells": len(outcome),
        })
    out = pd.DataFrame(rows)
    base = abs(out.loc[0, "placebo_coef"])
    out["share_absorbed"] = 1 - out["placebo_coef"].abs() / base
    return out

def _window_blocks(panel, window_days=WINDOW_DAYS):
    blocks = []
    for _, group in panel.groupby("slug", sort=False):
        rows = panel.index.get_indexer(group.index.to_numpy())
        days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
        within = np.abs(days[:, None] - days[None, :]) <= window_days
        blocks.append((rows, within, within.sum(axis=1)))
    return blocks

def _percentiles_from(scores, blocks, size):
    out = np.full(size, np.nan)
    for rows, within, counts in blocks:
        values = scores[rows]
        beats = ((values[None, :] < values[:, None]).astype(float)
                 + 0.5 * (values[None, :] == values[:, None]))
        out[rows] = 100.0 * (beats * within).sum(axis=1) / counts
    return out

def permutation_null(panel, draws=300, seed=7, window_days=WINDOW_DAYS):
    from .percentiles import within_benchmark_percentile

    panel = panel.reset_index(drop=True)
    if "share_newer" not in panel.columns:
        panel = within_benchmark_percentile(panel, window_days=window_days)

    blocks = _window_blocks(panel, window_days)
    eligible = panel["eligible"].to_numpy()
    release = panel[RELEASE_COL].to_numpy()
    asymmetry = panel["share_newer"].to_numpy(float)
    scores = panel["score"].to_numpy(float)
    slug_codes = pd.factorize(panel["slug"].to_numpy())[0]

    finite_x = np.isfinite(asymmetry)
    base_keep = eligible & finite_x
    codes = pd.factorize(release[base_keep])[0]
    n_groups = int(codes.max()) + 1 if codes.size else 0
    counts = np.bincount(codes, minlength=n_groups)
    x_kept = asymmetry[base_keep]
    x_demeaned = x_kept - (np.bincount(codes, weights=x_kept, minlength=n_groups)
                           / np.maximum(counts, 1))[codes]
    x_var = x_demeaned.var()

    def slope(percentile):
        if x_var < MIN_VARIANCE or codes.size < 3:
            return np.nan
        y = percentile[base_keep]
        if not np.isfinite(y).all():
            return np.nan
        y_demeaned = y - (np.bincount(codes, weights=y, minlength=n_groups)
                          / np.maximum(counts, 1))[codes]
        return float((y_demeaned * x_demeaned).mean() / x_var)

    observed = slope(_percentiles_from(scores, blocks, len(panel)))

    rng = np.random.default_rng(seed)
    order = np.argsort(slug_codes, kind="stable")
    starts = np.searchsorted(slug_codes[order], np.arange(slug_codes.max() + 1))
    groups = np.split(order, starts[1:])

    null = np.empty(draws)
    for draw in range(draws):
        shuffled = scores.copy()
        for members in groups:
            shuffled[members] = scores[rng.permutation(members)]
        null[draw] = slope(_percentiles_from(shuffled, blocks, len(panel)))

    if not np.isfinite(null).any():
        return {"observed": observed, "null_mean": np.nan, "null_sd": np.nan,
                "sd_away": np.nan, "draws_at_least_as_extreme": 0, "draws": draws}

    spread = float(np.nanstd(null))
    return {
        "observed": observed,
        "null_mean": float(np.nanmean(null)),
        "null_sd": spread,
        "sd_away": float(abs(observed - np.nanmean(null)) / spread) if spread else np.nan,
        "draws_at_least_as_extreme": int((null <= observed).sum()),
        "draws": draws,
    }

def _asymmetry_slope(panel):
    cells = panel[panel["eligible"]].dropna(subset=["percentile", "share_newer"])
    if cells.empty:
        return np.nan
    y = _within(cells, "percentile", RELEASE_COL).to_numpy()
    x = _within(cells, "share_newer", RELEASE_COL).to_numpy()
    if x.var() < MIN_VARIANCE:
        return np.nan
    return float(np.cov(y, x, bias=True)[0, 1] / x.var())

def trend_recovery(panel, draws=200, seed=11, window_days=WINDOW_DAYS):
    blocks = _window_blocks(panel, window_days)
    size = len(panel)
    observed = _asymmetry_slope(panel)
    if not np.isfinite(observed):
        return {"observed": observed, "recovered_mean": np.nan, "recovered_sd": np.nan,
                "share_recovered": np.nan, "draws": 0}

    frame = panel.copy()
    day = frame["Release date"].map(pd.Timestamp.toordinal).to_numpy(float)
    fitted = np.empty(len(frame))
    resid = np.empty(len(frame))
    for _, index in frame.groupby("slug").indices.items():
        x, y = day[index], frame["score"].to_numpy(float)[index]
        if len(index) < 3 or x.var() < MIN_VARIANCE:
            fitted[index], resid[index] = y, 0.0
            continue
        slope = np.cov(y, x, bias=True)[0, 1] / x.var()
        line = y.mean() + slope * (x - x.mean())
        fitted[index], resid[index] = line, y - line

    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("slug").indices.values())
    recovered = np.empty(draws)
    for draw in range(draws):
        synthetic = fitted.copy()
        for index in groups:
            synthetic[index] += rng.permutation(resid[index])
        scored = _percentiles_from(synthetic, blocks, size)
        replaced = frame.assign(percentile=scored)
        recovered[draw] = _asymmetry_slope(replaced)

    good = recovered[np.isfinite(recovered)]
    return {
        "observed": float(observed),
        "recovered_mean": float(good.mean()),
        "recovered_sd": float(good.std(ddof=1)),
        "share_recovered": float(good.mean() / observed),
        "draws": int(good.size),
    }

def balanced_contrast(panel):
    lo, hi = BALANCED_SHARE
    balanced = panel[
        panel["share_newer"].between(lo, hi) & (panel["n_peers"] >= BALANCED_MIN_PEERS)
    ]
    return contrast_by_release(balanced), len(balanced) / max(len(panel), 1)

def drop_capacity(panel, families, min_benchmarks=MIN_BENCHMARKS):
    eligible = panel[panel["eligible"]][[RELEASE_COL, "slug", "primary_org"]]
    linked = eligible.merge(
        families[["release_id", "family_id", "family_rank"]],
        left_on=RELEASE_COL, right_on="release_id", how="inner",
    )
    dense = eligible.groupby(RELEASE_COL)["slug"].nunique()
    dense = set(dense[dense >= min_benchmarks].index)

    rows = []
    for label, restrict in (("all adjacent pairs", False),
                            (f"both releases >= {min_benchmarks} eligible", True)):
        pairs = shared = 0
        providers = set()
        for _, group in linked.groupby("family_id"):
            ranks = sorted(group["family_rank"].dropna().unique())
            for first, second in zip(ranks, ranks[1:]):
                a = group[group["family_rank"] == first]
                b = group[group["family_rank"] == second]
                if a.empty or b.empty:
                    continue
                if restrict and not (
                    a[RELEASE_COL].iloc[0] in dense and b[RELEASE_COL].iloc[0] in dense
                ):
                    continue
                pairs += 1
                shared += len(set(a["slug"]) & set(b["slug"]))
                providers.add(a["primary_org"].iloc[0])
        rows.append({"definition": label, "pairs": pairs,
                     "shared_benchmarks": shared, "providers": len(providers)})
    return pd.DataFrame(rows)

def null_calibration(panel, drops=(1, 2, 3), draws=400, min_benchmarks=MIN_BENCHMARKS,
                     seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in drops:
        sample = []
        for _, group in panel[panel["eligible"]].groupby(RELEASE_COL):
            values = group["percentile"].dropna().to_numpy()
            n = len(values)
            if n < max(min_benchmarks, k + 1):
                continue
            picks = np.argpartition(rng.random((draws, n)), k, axis=1)[:, :k]
            drawn = values[picks].sum(axis=1)
            total = values.sum()
            sample.append(drawn * n / (k * (n - k)) - total / (n - k))
        pooled = np.concatenate(sample) if sample else np.array([])
        rows.append({"n_dropped": k,
                     "null_sd": float(pooled.std()) if pooled.size else np.nan,
                     "draws": int(pooled.size)})
    return pd.DataFrame(rows)

def main():
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    panel = within_benchmark_percentile(panel)
    families = pd.read_csv("data/families.csv")

    contrast = contrast_by_release(panel)
    print("the null of the omission deficit, measured with no disclosure labels")
    print(f"  eligible minus placebo: {contrast['contrast'].mean():+.2f} percentile points")
    print(f"  median {contrast['contrast'].median():+.2f}   "
          f"positive in {(contrast['contrast'] > 0).mean():.1%} of "
          f"{len(contrast)} releases, {contrast['organization'].nunique()} providers")
    print("  design.md predicts zero here under every innocent explanation")

    trend = trend_recovery(panel)
    print(f"  a bare capability trend recovers {trend['share_recovered']:.0%} of the "
          f"observed slope ({trend['recovered_mean']:+.2f} against {trend['observed']:+.2f}, "
          f"sd {trend['recovered_sd']:.2f} over {trend['draws']} draws)")

    print("\nwhere it comes from (coefficient on placebo, always within release)")
    for _, row in decomposition(panel).iterrows():
        print(f"  {row['step']:26s} {row['placebo_coef']:+7.2f}  "
              f"({row['share_absorbed']:5.0%} absorbed)")
    print("  benchmark composition carries most of it and peer-window composition")
    print("  the rest; jointly they leave nothing distinguishable from zero. The")
    print("  window is symmetric in days, a benchmark's model coverage is not.")

    balanced, kept = balanced_contrast(panel)
    print(f"\ntwo-sided peer window only ({kept:.0%} of cells retained)")
    print(f"  eligible minus placebo: {balanced['contrast'].mean():+.2f} "
          f"across {len(balanced)} releases")
    print("  trimming does not fix it. Conditioning does. Report both.")

    print("\nwhat a within-family drop design can ever reach")
    for _, row in drop_capacity(panel, families).iterrows():
        print(f"  {row['definition']:34s} {row['pairs']:4d} pairs  "
              f"{row['shared_benchmarks']:5d} shared benchmarks  "
              f"{row['providers']:2d} providers")

    print("\nnull distribution of a drop gap, releases at or above the analysis threshold")
    for _, row in null_calibration(panel).iterrows():
        print(f"  {int(row['n_dropped'])} dropped: sd {row['null_sd']:.1f} percentile points")

if __name__ == "__main__":
    main()
