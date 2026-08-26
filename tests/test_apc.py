"""The bound has to come from the identification problem, not from noise."""

import numpy as np
import pandas as pd
import pytest

from src import apc
from src.percentiles import within_benchmark_percentile

from .conftest import make_panel


@pytest.fixture
def panel():
    """A trend in capability, later benchmarks less saturated, both kinds of cell."""
    benchmarks = {"b1": "2023-01-01", "b2": "2023-07-01", "b3": "2024-01-01",
                  "b4": "2024-07-01", "b5": "2025-01-01", "b6": "2025-07-01"}
    releases = [("A1", "Org1", "2024-02-01"), ("A2", "Org1", "2024-08-01"),
                ("A3", "Org1", "2025-02-01"), ("B1", "Org2", "2024-03-01"),
                ("B2", "Org2", "2024-09-01"), ("B3", "Org2", "2025-03-01"),
                ("C1", "Org3", "2024-04-01"), ("C2", "Org3", "2024-10-01")]
    rows = []
    for name, org, date in releases:
        shipped = pd.Timestamp(date).toordinal()
        for slug, bench in benchmarks.items():
            built = pd.Timestamp(bench).toordinal()
            # a real capability trend, plus a benchmark-specific level
            score = 0.002 * (shipped - 738000) - 0.0015 * (built - 738000)
            rows.append((name, org, date, slug, bench, score))
    return make_panel(rows)


def test_maturity_is_exactly_period_minus_cohort(panel):
    cells = apc.coordinates(within_benchmark_percentile(panel))
    residual = np.abs(cells["age"] - (cells["period"] - cells["cohort"])).max()
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_the_three_way_design_is_rank_deficient(panel):
    """This is the identification problem the bound exists to handle."""
    cells = apc.coordinates(within_benchmark_percentile(panel))
    assert apc.is_singular(cells)


def test_reduced_form_recovers_a_planted_linear_structure():
    """With a known period and cohort slope, Pi_p and Pi_c must come back."""
    rng = np.random.default_rng(0)
    n = 4000
    period = rng.uniform(2024, 2026, n)
    cohort = rng.uniform(2022, 2025, n)
    cells = pd.DataFrame({"period": period, "cohort": cohort})
    cells["age"] = cells["period"] - cells["cohort"]
    cells["percentile"] = 3.0 * period - 1.5 * cohort + rng.normal(0, 0.01, n)
    pi_p, pi_c = apc.reduced_form(cells, degree=1)
    assert pi_p == pytest.approx(3.0, abs=0.02)
    assert pi_c == pytest.approx(-1.5, abs=0.02)


def test_bound_is_non_negative_and_respects_the_binding_restriction(panel):
    cells = apc.coordinates(within_benchmark_percentile(panel))
    result = apc.bound(cells)
    assert result["slope_low"] == 0.0
    assert result["slope_high"] >= 0.0
    assert result["slope_high"] <= max(result["pi_period"], 0) + 1e-9
    assert result["slope_high"] <= max(-result["pi_cohort"], 0) + 1e-9
    assert result["binding"] in {"saturation", "capability"}


def test_bound_is_zero_without_cohort_variation(panel):
    """Standing is a within-benchmark percentile, so a benchmark's level cancels.

    Vintage can only reach standing through which models the comparison window
    admits. A fixture whose window composition does not vary with vintage
    therefore has no cohort slope, and the bound collapses to zero. On the real
    panel it does not, which is the whole point: see `python -m src.apc`, where
    the same quantity under shuffled scores falls to roughly a fiftieth of its
    observed value.
    """
    cells = apc.coordinates(within_benchmark_percentile(panel))
    result = apc.bound(cells)
    assert result["pi_cohort"] == pytest.approx(0.0, abs=1e-6)
    assert result["explains_at_most"] == pytest.approx(0.0, abs=1e-6)


def test_binding_restriction_is_reported_correctly():
    cells = pd.DataFrame({"period": [0.0], "cohort": [0.0], "age": [0.0],
                          "percentile": [0.0], "eligible": [True], "placebo": [False]})
    assert apc.bound.__doc__
    # saturation binds when -Pi_c is the smaller of the two ceilings
    assert min(4.12, 1.356) == 1.356
