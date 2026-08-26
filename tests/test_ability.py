"""The joint scale has to be identified before its verdict means anything."""

import numpy as np
import pandas as pd
import pytest

from src import ability

from .conftest import make_panel


@pytest.fixture
def panel():
    """A capability trend, benchmarks of differing difficulty, both cell kinds."""
    rng = np.random.default_rng(0)
    benchmarks = {f"b{i}": pd.Timestamp("2022-06-01") + pd.Timedelta(days=140 * i)
                  for i in range(8)}
    releases = [(f"R{j}", f"Org{j % 4}", pd.Timestamp("2023-06-01") + pd.Timedelta(days=40 * j))
                for j in range(30)]
    rows = []
    for name, org, date in releases:
        strength = (date.toordinal() - 738700) / 400.0
        for slug, built in benchmarks.items():
            hardness = (built.toordinal() - 738300) / 500.0
            latent = strength - hardness + rng.normal(0, 0.25)
            score = float(1 / (1 + np.exp(-latent)))
            rows.append((name, org, date.strftime("%Y-%m-%d"), slug,
                         built.strftime("%Y-%m-%d"), score))
    return make_panel(rows)


def test_commensurable_keeps_only_zero_to_one_benchmarks(panel):
    frame = panel.copy()
    frame.loc[frame["slug"] == "b0", "score"] = 1500.0
    kept = ability.commensurable(frame)
    assert "b0" not in set(kept["slug"])
    assert kept["y"].notna().all()
    assert np.isfinite(kept["y"]).all()


def test_logit_is_finite_at_the_bounds():
    frame = pd.DataFrame({"slug": ["b", "b"], "score": [0.0, 1.0]})
    out = ability.commensurable(frame)
    assert np.isfinite(out["y"]).all()


def test_estimable_enforces_both_density_floors(panel):
    frame = ability.commensurable(panel)
    kept = ability.estimable(frame, min_releases=5, min_benchmarks=2)
    assert kept.groupby("slug")["release_id"].nunique().min() >= 5
    assert kept.groupby("release_id")["slug"].nunique().min() >= 2


def test_ridge_keeps_loadings_bounded(panel):
    """Left free the loading diverges at this density; that is why it is ridged."""
    frame = ability.commensurable(panel)
    sample = ability.estimable(frame[frame["eligible"]], min_releases=5, min_benchmarks=2)
    _, _, loading = ability.fit(sample, ridge=ability.RIDGE)
    assert float(loading.abs().max()) < 10.0


def test_ability_recovers_the_planted_ordering(panel):
    """Later releases are stronger by construction, so ability must track date."""
    frame = ability.commensurable(panel)
    sample = ability.estimable(frame[frame["eligible"]], min_releases=5, min_benchmarks=2)
    est, _, _ = ability.fit(sample)
    dates = (sample.drop_duplicates("release_id")
             .set_index("release_id")["Release date"].map(pd.Timestamp.toordinal))
    common = est.index.intersection(dates.index)
    assert np.corrcoef(est.reindex(common), dates.reindex(common))[0, 1] > 0.8


def test_placebo_gap_returns_none_without_both_kinds(panel):
    frame = ability.commensurable(panel)
    only_eligible = frame[frame["eligible"]].copy()
    sample = ability.estimable(only_eligible, min_releases=5, min_benchmarks=2)
    est, diff, load = ability.fit(sample)
    scored = ability.residuals(only_eligible, est, diff, load)
    assert ability.placebo_gap(scored) is None


def test_placebo_cells_are_scored_out_of_sample(panel):
    """The fit must never see a placebo cell, or the gap is zero by construction."""
    frame = ability.commensurable(panel)
    sample = ability.estimable(frame[frame["eligible"]], min_releases=5, min_benchmarks=2)
    assert not sample["placebo"].any()
    est, diff, load = ability.fit(sample)
    scored = ability.residuals(frame, est, diff, load)
    assert scored["placebo"].sum() > 0
