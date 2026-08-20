import numpy as np
import pandas as pd

from src.config import RELEASE_COL
from src.percentiles import within_benchmark_percentile
from tests.conftest import make_panel


def test_window_excludes_distant_models(tiny_panel):
    out = within_benchmark_percentile(tiny_panel, window_days=182)
    by = out.set_index(RELEASE_COL)

    # A, B, C are all within 182 days of each other; D is years later.
    assert by.loc["A", "n_peers"] == 3
    assert by.loc["D", "n_peers"] == 1

    # Among A(0.1), B(0.5), C(0.9): midrank percentiles 1/6, 3/6, 5/6.
    assert by.loc["A", "percentile"] == 100 / 6
    assert by.loc["B", "percentile"] == 50.0
    assert by.loc["C", "percentile"] == 500 / 6

    # D is alone in its window, so it is at the midpoint of a field of one.
    assert by.loc["D", "percentile"] == 50.0


def test_alltime_ranking_would_place_d_differently(tiny_panel):
    """The whole point of windowing: D's standing changes when it is ranked
    against models from another era rather than its own."""
    out = within_benchmark_percentile(tiny_panel, window_days=182)
    alltime = tiny_panel.groupby("slug")["score"].rank(pct=True) * 100
    d_windowed = out.loc[out[RELEASE_COL] == "D", "percentile"].iloc[0]
    d_alltime = alltime[tiny_panel[RELEASE_COL] == "D"].iloc[0]
    assert d_windowed == 50.0
    assert d_alltime == 100.0  # top of 4 by raw score, though alone in its era
    assert d_windowed != d_alltime


def test_ties_take_midrank():
    panel = make_panel([
        ("A", "O", "2025-01-01", "b", "2024-01-01", 0.5),
        ("B", "O", "2025-01-02", "b", "2024-01-01", 0.5),
    ])
    out = within_benchmark_percentile(panel, window_days=182)
    assert set(out["percentile"]) == {50.0}


def test_window_width_changes_peer_count(tiny_panel):
    wide = within_benchmark_percentile(tiny_panel, window_days=10_000)
    assert wide["n_peers"].min() == 4
