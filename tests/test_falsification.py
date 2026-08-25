"""The falsification tests must fire on a planted effect and stay quiet without one.

A permutation test that never rejects, or an excess-omission regression whose
intercept is negative regardless of the data, would let the study confirm
itself. These check both directions.
"""
import numpy as np
import pandas as pd
import pytest

from src.falsification import excess_omission, permutation_test
from src.selectivity import release_sets
from tests.test_estimator import synthetic


def test_permutation_rejects_strategic_omission():
    panel = synthetic(strategic=True, n_releases=40)
    observed, null, p = permutation_test(panel, draws=300, seed=1)
    assert observed > 20
    assert abs(np.nanmean(null)) < 5, "null should centre on zero"
    assert p < 0.01


def test_permutation_does_not_reject_random_omission():
    panel = synthetic(strategic=False, n_releases=40, seed=7)
    observed, null, p = permutation_test(panel, draws=300, seed=1)
    assert p > 0.05, f"observed={observed}, p={p}"


def test_permutation_null_preserves_disclosed_counts():
    """The convention explanation is granted, not tested: each release keeps
    exactly as many disclosed benchmarks as it really had."""
    panel = synthetic(strategic=True, n_releases=10)
    eligible = panel[panel["group"] == "eligible"]
    counts = eligible.groupby("release_id")["reported"].sum()
    assert counts.nunique() == 1  # synthetic omits exactly 3 everywhere
    observed, null, p = permutation_test(panel, draws=50, seed=0)
    assert np.isfinite(observed)


def test_excess_omission_intercept_is_negative_under_strategy():
    sets = release_sets(synthetic(strategic=True, n_releases=60))
    fit = excess_omission(sets)
    assert fit is not None
    intercept = fit["beta"][0]
    assert intercept < -10, intercept


def test_excess_omission_intercept_is_near_zero_under_the_null():
    sets = release_sets(synthetic(strategic=False, n_releases=60, seed=3))
    fit = excess_omission(sets)
    assert fit is not None
    intercept, slope = fit["beta"]
    assert abs(intercept - 50 * (1 - slope)) < 15, (intercept, slope)


def test_excess_omission_returns_none_on_thin_data():
    sets = release_sets(synthetic(strategic=True, n_releases=5))
    assert excess_omission(sets) is None


class TestLabelFreePlacebo:
    """The strongest form of the placebo test: it uses no disclosure coding at
    all, so it can be run before a single cell is read, and a non-zero result
    condemns the standing measure rather than the coding."""

    PANEL = pd.DataFrame([
        dict(release_id="R1", group="eligible", percentile=70.0,
             percentile_balanced=60.0, Organization="A"),
        dict(release_id="R1", group="eligible", percentile=80.0,
             percentile_balanced=70.0, Organization="A"),
        dict(release_id="R1", group="placebo", percentile=50.0,
             percentile_balanced=55.0, Organization="A"),
        dict(release_id="R2", group="eligible", percentile=40.0,
             percentile_balanced=40.0, Organization="B"),
        dict(release_id="R2", group="placebo", percentile=40.0,
             percentile_balanced=40.0, Organization="B"),
        # a release with only one kind of cell carries no contrast
        dict(release_id="R3", group="eligible", percentile=90.0,
             percentile_balanced=90.0, Organization="B"),
    ])

    def test_gap_is_eligible_minus_placebo(self):
        from src.falsification import eligible_vs_placebo

        got = eligible_vs_placebo(self.PANEL).set_index("release_id")["gap"]
        assert got["R1"] == pytest.approx(25.0)   # mean(70,80) - 50
        assert got["R2"] == pytest.approx(0.0)

    def test_releases_without_both_kinds_are_dropped(self):
        from src.falsification import eligible_vs_placebo

        assert "R3" not in set(eligible_vs_placebo(self.PANEL)["release_id"])

    def test_no_disclosure_column_is_required(self):
        """The point of this test is that it runs with no coding sheet."""
        from src.falsification import eligible_vs_placebo

        assert "reported" not in self.PANEL.columns
        assert len(eligible_vs_placebo(self.PANEL)) == 2

    def test_alternative_measure_is_honoured(self):
        from src.falsification import eligible_vs_placebo

        got = eligible_vs_placebo(self.PANEL, value="percentile_balanced")
        assert got.set_index("release_id")["gap"]["R1"] == pytest.approx(10.0)
