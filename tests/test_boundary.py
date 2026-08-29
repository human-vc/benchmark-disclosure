
import numpy as np
import pandas as pd
import pytest

from src import boundary

from .conftest import make_panel

def synthetic(jump=0.0, n_releases=90, seed=0, slope=0.00002):
    rng = np.random.default_rng(seed)
    benchmarks = {f"b{i}": pd.Timestamp("2023-01-01") + pd.Timedelta(days=37 * i)
                  for i in range(14)}
    rows = []
    for j in range(n_releases):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=11 * j)
        for slug, built in benchmarks.items():
            maturity = (date - built).days
            level = 0.5 + slope * maturity + (jump if maturity > 0 else 0.0)
            rows.append((f"R{j}", f"Org{j % 12}", date.strftime("%Y-%m-%d"), slug,
                         built.strftime("%Y-%m-%d"),
                         float(np.clip(level + rng.normal(0, 0.02), 0, 1))))
    return make_panel(rows)

def test_running_variable_is_the_date_difference():
    frame = boundary.cells(synthetic())
    expected = (frame["Release date"] - frame["benchmark_release_date"]).dt.days
    assert (frame["maturity"] == expected).all()

def test_treatment_is_the_threshold_on_maturity():
    frame = boundary.cells(synthetic())
    assert (frame["treat"] == (frame["maturity"] > 0).astype(int)).all()
    assert frame.loc[frame["eligible"], "maturity"].min() > 0
    assert frame.loc[frame["placebo"], "maturity"].max() <= 0

def test_local_linear_finds_a_planted_jump():
    frame = boundary.cells(synthetic(jump=0.25))
    frame["outcome"] = frame["score"]
    result = boundary.local_linear(frame, bandwidth=200, outcome="outcome")
    assert result["jump"] > 0.1
    assert result["low"] > 0

def test_local_linear_finds_nothing_when_there_is_nothing():
    frame = boundary.cells(synthetic(jump=0.0))
    frame["outcome"] = frame["score"]
    result = boundary.local_linear(frame, bandwidth=200, outcome="outcome")
    assert result["low"] <= 0 <= result["high"]

def test_randomization_p_is_large_under_no_effect():
    frame = boundary.cells(synthetic(jump=0.0, slope=0.0))
    frame["outcome"] = frame["score"]
    band = frame[np.abs(frame["maturity"]) <= 120]
    assert boundary.randomization_p(band, "outcome", draws=499) > 0.10

def test_randomization_picks_up_a_slope_inside_the_window():
    frame = boundary.cells(synthetic(jump=0.0, slope=0.0004))
    frame["outcome"] = frame["score"]
    band = frame[np.abs(frame["maturity"]) <= 120]
    assert boundary.randomization_p(band, "outcome", draws=499) < 0.05

def test_randomization_p_is_small_under_a_large_effect():
    frame = boundary.cells(synthetic(jump=0.4))
    frame["outcome"] = frame["score"]
    band = frame[np.abs(frame["maturity"]) <= 120]
    assert boundary.randomization_p(band, "outcome", draws=499) < 0.05

def test_window_selection_respects_the_minimum_per_side():
    frame = boundary.cells(synthetic())
    width, table = boundary.select_window(frame, widths=range(20, 200, 20), draws=199)
    assert (table["left"] >= boundary.MIN_PER_SIDE).all()
    assert (table["right"] >= boundary.MIN_PER_SIDE).all()
    if width is not None:
        assert table.loc[table["width"] <= width, "min_p"].min() > boundary.BALANCE_ALPHA

def test_window_selection_stops_at_the_first_imbalance():
    frame = boundary.cells(synthetic())
    width, table = boundary.select_window(frame, widths=range(20, 400, 20), draws=199)
    failed = table[table["min_p"] <= boundary.BALANCE_ALPHA]
    if len(failed):
        assert table["width"].max() == failed["width"].iloc[0]

def test_density_ratio_is_one_on_a_uniform_design():
    frame = boundary.cells(synthetic())
    assert 0.5 < boundary.density_ratio(frame, width=60) < 2.0
