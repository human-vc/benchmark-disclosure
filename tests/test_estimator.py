"""Does the estimator recover an effect that is really there, and stay at zero
when it is not?

An estimator that has never been shown to recover a planted signal is not
evidence about the world. These build panels with a known answer: one where
omission is deliberately concentrated on the release's worst benchmarks, and
one where omission is random. The first must come back positive, the second
must come back at zero, and the placebo must come back at zero in both.
"""
import numpy as np
import pandas as pd

from src.config import RELEASE_COL
from src.percentiles import within_benchmark_percentile
from src.selectivity import (
    drop_estimator,
    gap_by_release,
    omission_deficit,
    release_sets,
)


def synthetic(n_releases=60, n_bench=10, n_placebo=3, strategic=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_releases):
        release = f"Org{r % 6} | Model {r} | 2025-06-01"
        date = pd.Timestamp("2025-06-01")
        for b in range(n_bench):
            rows.append((release, f"Org{r % 6}", date, f"b{b}", 0, rng.normal()))
        for b in range(n_placebo):
            rows.append((release, f"Org{r % 6}", date, f"p{b}", 1, rng.normal()))
    panel = pd.DataFrame(
        rows,
        columns=[RELEASE_COL, "Organization", "Release date", "slug", "is_placebo", "score"],
    )
    panel["Model accessibility"] = "API access"
    panel = within_benchmark_percentile(panel, window_days=182)
    panel["group"] = np.where(panel["is_placebo"] == 1, "placebo", "eligible")

    # Disclosure rule: strategic providers omit their three worst eligible
    # benchmarks; the null provider omits three at random.
    panel["reported"] = 1.0
    for release, group in panel[panel["group"] == "eligible"].groupby(RELEASE_COL):
        if strategic:
            worst = group.nsmallest(3, "percentile").index
        else:
            worst = rng.choice(group.index, size=3, replace=False)
        panel.loc[worst, "reported"] = 0.0
    panel.loc[panel["group"] == "placebo", "reported"] = 0.0
    panel["orbit_category"] = np.where(panel["reported"] == 1, "A", "E")
    return panel


def test_recovers_planted_selectivity():
    panel = synthetic(strategic=True)
    gaps = gap_by_release(release_sets(panel), against="omitted")
    assert len(gaps) == 60
    assert gaps["gap"].mean() > 25, gaps["gap"].mean()
    assert (gaps["gap"] > 0).mean() == 1.0


def test_null_panel_shows_no_gap():
    panel = synthetic(strategic=False)
    gaps = gap_by_release(release_sets(panel), against="omitted")
    assert abs(gaps["gap"].mean()) < 5, gaps["gap"].mean()


def test_placebo_group_cannot_support_the_headline_statistic():
    """Every benchmark postdating a release is non-disclosed, so the placebo
    group contains no disclosed set to compare against. Running the headline
    statistic 'within the placebo group' is not defined -- which is why the
    falsification is omission_deficit(), not this."""
    panel = synthetic(strategic=True)
    placebo_only = panel[panel["group"] == "placebo"]
    assert (placebo_only["reported"] == 0).all()
    assert gap_by_release(release_sets(placebo_only), against="omitted").empty


def test_headline_gap_tracks_disclosed_selection():
    """Sanity: with strategic omission the disclosed set really does sit above
    the unselected placebo set, and with random omission it does not."""
    strategic = gap_by_release(
        release_sets(synthetic(strategic=True)), against="placebo"
    )
    null = gap_by_release(release_sets(synthetic(strategic=False)), against="placebo")
    assert strategic["gap"].mean() > 10
    assert abs(null["gap"].mean()) < 5


def test_omission_deficit_is_zero_when_omission_is_random():
    """The identifying contrast, under the null."""
    panel = synthetic(strategic=False)
    deficit = omission_deficit(release_sets(panel))
    assert abs(deficit["gap"].mean()) < 5, deficit["gap"].mean()


def test_omission_deficit_is_negative_when_omission_is_strategic():
    """The same contrast, under the alternative. Both sets are non-disclosures,
    so the only difference is whether the benchmark could have been reported."""
    panel = synthetic(strategic=True)
    deficit = omission_deficit(release_sets(panel))
    assert deficit["gap"].mean() < -25, deficit["gap"].mean()


def test_drop_estimator_is_negative_when_drops_are_the_worst_benchmarks():
    panel = synthetic(strategic=True)
    panel["Model accessibility"] = "API access"
    drops = drop_estimator(panel[panel["group"] == "eligible"])
    assert len(drops) == 60
    assert drops["drop_gap"].mean() < -25, drops["drop_gap"].mean()


def test_gap_requires_minimum_counts():
    panel = synthetic(strategic=True)
    thin = panel[panel[RELEASE_COL] == panel[RELEASE_COL].iloc[0]].copy()
    thin.loc[thin["reported"] == 1, "reported"] = 0.0  # nothing disclosed
    assert gap_by_release(release_sets(thin), against="omitted").empty
