import numpy as np
import pandas as pd
import pytest

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


class TestSideDecomposition:
    """The window is symmetric in calendar time; a benchmark's model coverage
    is not, because it starts when the benchmark was built. These pin the
    measurement of that asymmetry, which is what the study's central claim
    turns on."""

    def test_percentile_is_the_weighted_average_of_the_two_sides(self):
        """percentile_sided = (1-s)*P_old + s*P_new, exactly."""
        import numpy as np

        from src.config import INTERIM
        from src.percentiles import add_percentiles

        path = INTERIM / "panel.csv"
        if not path.exists():
            pytest.skip("panel not built")
        panel = add_percentiles(pd.read_csv(path, parse_dates=["Release date"]))
        ok = panel.dropna(subset=["percentile_sided", "newer_share"])
        recomposed = (
            (1 - ok["newer_share"]) * ok["percentile_older"].fillna(0)
            + ok["newer_share"] * ok["percentile_newer"].fillna(0)
        )
        assert np.abs(ok["percentile_sided"] - recomposed).max() < 1e-9

    def test_balanced_is_undefined_when_a_side_is_empty(self):
        """A release at the very start of a benchmark's coverage has no older
        peer. Inventing one would hide the case the measure exists to expose."""
        import numpy as np

        from src.percentiles import within_benchmark_percentile

        panel = pd.DataFrame({
            "slug": ["b"] * 3,
            "score": [0.1, 0.5, 0.9],
            "Release date": pd.to_datetime(["2025-01-01", "2025-02-01",
                                            "2025-03-01"]),
        })
        out = within_benchmark_percentile(panel, window_days=182)
        assert np.isnan(out.loc[0, "percentile_balanced"])   # nothing older
        assert np.isnan(out.loc[2, "percentile_balanced"])   # nothing newer
        assert not np.isnan(out.loc[1, "percentile_balanced"])

    def test_newer_share_is_one_at_the_left_boundary(self):
        from src.percentiles import within_benchmark_percentile

        panel = pd.DataFrame({
            "slug": ["b"] * 3,
            "score": [0.1, 0.5, 0.9],
            "Release date": pd.to_datetime(["2025-01-01", "2025-02-01",
                                            "2025-03-01"]),
        })
        out = within_benchmark_percentile(panel, window_days=182)
        assert out.loc[0, "newer_share"] == 1.0
        assert out.loc[2, "newer_share"] == 0.0
        assert out.loc[1, "newer_share"] == pytest.approx(0.5)
