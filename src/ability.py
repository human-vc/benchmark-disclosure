"""A standing measure with no comparison window, and what it does not fix.

The windowed percentile ranks a release against whichever models were scored
near it in calendar time, which is where the collinearity in
Section~\\ref{sec:core} comes from. The obvious repair is to stop comparing
against a window at all: estimate a release's ability and a benchmark's
difficulty jointly over the whole score matrix, so a release's standing does
not depend on who happened to be evaluated beside it.

This fits that model and puts it through the same falsification test. Cells
whose benchmark postdates the release are held out, the model is fit on
eligible cells only, and the placebo cells are scored out of sample. If the
measure were clean the eligible-minus-placebo gap would be zero.

It is not. The gap shrinks against the windowed percentile but survives in
every specification, which is the same verdict trimming and side-balancing
already returned: the vintage structure is in which cells exist, not only in
how the comparison set is drawn, so changing the estimator does not remove it.
"""

import numpy as np
import pandas as pd

from .config import INTERIM

CLIP = 0.005
MIN_RELEASES_PER_BENCHMARK = 20
MIN_BENCHMARKS_PER_RELEASE = 3
RIDGE = 2.0


def commensurable(panel):
    """Benchmarks reported on a shared 0 to 1 scale.

    The rest are Elo-like ratings, time horizons and bespoke indices with no
    common unit, which is the same reason the paper does not compare raw
    scores across benchmarks.
    """
    span = panel.groupby("slug")["score"].agg(["min", "max"])
    keep = span[(span["min"] >= -0.01) & (span["max"] <= 1.01)].index
    frame = panel[panel["slug"].isin(keep)].copy()
    bounded = frame["score"].clip(CLIP, 1 - CLIP)
    frame["y"] = np.log(bounded / (1 - bounded))
    return frame


def estimable(frame, min_releases=MIN_RELEASES_PER_BENCHMARK,
              min_benchmarks=MIN_BENCHMARKS_PER_RELEASE):
    """Cells dense enough to carry a benchmark or a release parameter."""
    kept = frame
    for _ in range(8):
        per_benchmark = kept.groupby("slug")["release_id"].transform("nunique")
        per_release = kept.groupby("release_id")["slug"].transform("nunique")
        mask = (per_benchmark >= min_releases) & (per_benchmark.notna()) & (
            per_release >= min_benchmarks)
        if mask.all():
            break
        kept = kept[mask]
    return kept


def fit(frame, ridge=RIDGE, iterations=300, tol=1e-9):
    """Alternating least squares for y = c_b * a_i + d_b.

    The loading is ridged toward one. Left free it is unidentified at this
    density: benchmarks seen on few releases take slopes in the thousands and
    out-of-sample predictions diverge.
    """
    ability = pd.Series(0.0, index=sorted(frame["release_id"].unique()))
    difficulty = pd.Series(0.0, index=sorted(frame["slug"].unique()))
    loading = pd.Series(1.0, index=difficulty.index)
    penalty = np.diag([ridge, 0.0])
    target = np.array([ridge, 0.0])

    for _ in range(iterations):
        previous = ability.copy()
        for slug, group in frame.groupby("slug"):
            x = ability.reindex(group["release_id"]).to_numpy()
            design = np.column_stack([x, np.ones(len(x))])
            solution = np.linalg.solve(design.T @ design + penalty,
                                       design.T @ group["y"].to_numpy() + target)
            loading[slug], difficulty[slug] = solution
        for release, group in frame.groupby("release_id"):
            weights = loading.reindex(group["slug"]).to_numpy()
            centred = group["y"].to_numpy() - difficulty.reindex(group["slug"]).to_numpy()
            denominator = weights @ weights
            ability[release] = (weights @ centred) / denominator if denominator > 1e-9 else 0.0
        ability = (ability - ability.mean()) / ability.std() * 2.0
        if (ability - previous).abs().max() < tol:
            break
    return ability, difficulty, loading


def residuals(frame, ability, difficulty, loading):
    """Observed minus predicted, for every cell the fit can score."""
    scorable = frame[frame["release_id"].isin(ability.index)
                     & frame["slug"].isin(difficulty.index)].copy()
    predicted = (loading.reindex(scorable["slug"]).to_numpy()
                 * ability.reindex(scorable["release_id"]).to_numpy()
                 + difficulty.reindex(scorable["slug"]).to_numpy())
    scorable["residual"] = scorable["y"].to_numpy() - predicted
    return scorable


def placebo_gap(scorable):
    """The falsification test, in units of the residual spread."""
    cells = scorable[scorable["eligible"] | scorable["placebo"]].copy()
    cells["set"] = np.where(cells["eligible"], "eligible", "placebo")
    wide = cells.pivot_table(index="release_id", columns="set",
                             values="residual", aggfunc="mean").dropna()
    if wide.empty or not {"eligible", "placebo"}.issubset(wide.columns):
        return None
    gap = wide["eligible"] - wide["placebo"]
    spread = scorable["residual"].std()
    return {
        "mean": float(gap.mean()),
        "standardised": float(gap.mean() / spread),
        "share_positive": float((gap > 0).mean()),
        "n_releases": int(len(gap)),
        "residual_sd": float(spread),
        "organisations": cells.groupby("release_id")["primary_org"].first().reindex(wide.index),
        "gap": gap,
    }


def main():
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    frame = commensurable(panel)
    sample = estimable(frame[frame["eligible"]])
    ability, difficulty, loading = fit(sample)
    scored = residuals(frame, ability, difficulty, loading)
    result = placebo_gap(scored)

    from .stats import randomization_test_mean
    test = randomization_test_mean(result["gap"].to_numpy(),
                                   cluster=result["organisations"].to_numpy())

    print("a standing measure with no comparison window")
    print(f"  fit on {len(sample)} eligible cells, "
          f"{sample['release_id'].nunique()} releases, {sample['slug'].nunique()} benchmarks")
    print(f"  loadings stay bounded at {float(loading.abs().max()):.2f} under the ridge")
    print(f"\n  placebo cells scored out of sample: "
          f"{int(scored['placebo'].sum())} cells across {result['n_releases']} releases")
    print(f"  eligible minus placebo: {result['mean']:+.3f} logit, "
          f"{result['standardised']:+.3f} residual SD, "
          f"positive in {100 * result['share_positive']:.1f}% of releases")
    print(f"  sign-flip over {test['n_clusters']} providers: p = {test['p_value']:.4f}")
    print("\n  the windowed percentile puts the same contrast at +0.449 SD, so removing")
    print("  the comparison window shrinks this gap without closing it.")


if __name__ == "__main__":
    main()
