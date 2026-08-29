import numpy as np
import pandas as pd

from src.paper_numbers import _asymmetry_slope, _contrast
from src.percentiles import side_balanced_percentile, within_benchmark_percentile
from tests.conftest import make_panel

def panel_with_boundary():
    rows = []
    for i in range(24):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i)
        score = 0.4 + 0.02 * i
        rows.append((f"R{i}", f"Org{i % 3}", date, "old", "2023-01-01", score))
        if i >= 8:
            rows.append((f"R{i}", f"Org{i % 3}", date, "late", "2025-01-01", score))
    return side_balanced_percentile(
        within_benchmark_percentile(make_panel(rows), window_days=182), window_days=182
    )

def test_contrast_reports_a_consistent_release_count():
    panel = panel_with_boundary()
    result = _contrast(panel, "percentile")
    assert result["n_releases"] > 0
    assert 0.0 <= result["share_positive"] <= 1.0
    assert np.isfinite(result["mean"]) and np.isfinite(result["median"])

def test_balanced_measure_is_defined_on_no_more_cells_than_the_raw_one():
    panel = panel_with_boundary()
    assert panel["pct_balanced"].notna().sum() <= panel["percentile"].notna().sum()

def test_balanced_measure_reduces_the_reported_asymmetry_slope():
    panel = panel_with_boundary()
    assert abs(_asymmetry_slope(panel, "pct_balanced")) < abs(
        _asymmetry_slope(panel, "percentile")
    )

def test_contrast_ignores_releases_missing_either_side():
    rows = [("R0", "Org", "2025-06-01", "b", "2024-01-01", 0.5),
            ("R1", "Org", "2025-07-01", "b", "2024-01-01", 0.7)]
    panel = side_balanced_percentile(
        within_benchmark_percentile(make_panel(rows), window_days=182), window_days=182
    )
    assert _contrast(panel, "percentile")["n_releases"] == 0
