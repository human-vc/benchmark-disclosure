
import pandas as pd
import pytest

from src import sensitivity
from src.percentiles import within_benchmark_percentile
from src.placebo_calibration import contrast_by_release

from .conftest import make_panel

@pytest.fixture
def panel():
    benchmarks = {
        "b1": "2023-01-01", "b2": "2023-06-01", "b3": "2024-01-01",
        "b4": "2024-06-01", "b5": "2025-01-01", "b6": "2025-06-01",
    }
    releases = [
        ("A1", "Org1", "2024-03-01"), ("A2", "Org1", "2024-09-01"),
        ("A3", "Org1", "2025-03-01"), ("B1", "Org2", "2024-04-01"),
        ("B2", "Org2", "2024-10-01"), ("B3", "Org2", "2025-04-01"),
        ("C1", "Org3", "2024-05-01"), ("C2", "Org3", "2024-11-01"),
        ("D1", "Org4", "2024-06-01"), ("D2", "Org4", "2024-12-01"),
    ]
    rows, tick = [], 0
    for name, org, date in releases:
        for slug, bench_date in benchmarks.items():
            tick += 1
            rows.append((name, org, date, slug, bench_date, (tick % 17) / 17))
    return make_panel(rows)

def test_baseline_is_a_grid_coordinate():
    assert sensitivity.BASELINE["min_benchmarks"] in sensitivity.THRESHOLDS
    assert sensitivity.BASELINE["window_days"] in sensitivity.WIDTHS

def test_threshold_of_one_is_a_no_op(panel):
    scored = within_benchmark_percentile(panel.copy())
    assert len(sensitivity._threshold(scored, 1)) == len(scored)

def test_baseline_cell_equals_the_headline(panel):
    scored = within_benchmark_percentile(panel.copy(),
                                         window_days=sensitivity.BASELINE["window_days"])
    headline = contrast_by_release(scored)

    table = sensitivity.sweep(panel.copy(),
                              widths=(sensitivity.BASELINE["window_days"],),
                              thresholds=(sensitivity.BASELINE["min_benchmarks"],))
    row = sensitivity.baseline_row(table)

    assert row["n_releases"] == len(headline)
    assert row["mean"] == pytest.approx(headline["contrast"].mean(), abs=1e-9)
    assert row["median"] == pytest.approx(headline["contrast"].median(), abs=1e-9)
    assert row["share_positive"] == pytest.approx((headline["contrast"] > 0).mean(), abs=1e-9)

def test_baseline_row_rejects_an_ambiguous_grid():
    table = pd.DataFrame({
        "window_days": [182, 182], "min_benchmarks": [1, 1],
        "mean": [1.0, 2.0], "spec_id": ["a", "b"],
    })
    with pytest.raises(ValueError):
        sensitivity.baseline_row(table)

def test_leave_one_out_drops_each_organisation_once(panel):
    scored = within_benchmark_percentile(panel.copy())
    loo = sensitivity.leave_one_org_out(scored)
    dropped = loo[loo["dropped"] != "(none)"]
    assert dropped["dropped"].is_unique
    assert len(dropped) == contrast_by_release(scored)["organization"].nunique()
    full = loo.loc[loo["dropped"] == "(none)", "mean"].iloc[0]
    assert full == pytest.approx(contrast_by_release(scored)["contrast"].mean(), abs=1e-9)

def test_share_newer_gap_is_placebo_minus_eligible(panel):
    scored = within_benchmark_percentile(panel.copy())
    cells = scored[scored["eligible"] | scored["placebo"]]
    expected = (cells.loc[cells["placebo"], "share_newer"].mean()
                - cells.loc[cells["eligible"], "share_newer"].mean())
    assert sensitivity._share_newer_gap(scored) == pytest.approx(expected, abs=1e-12)
