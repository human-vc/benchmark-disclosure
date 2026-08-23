import numpy as np
import pandas as pd

from src.config import RELEASE_COL
from src.percentiles import side_balanced_percentile, within_benchmark_percentile
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


def test_side_balanced_decomposition_is_an_identity():
    """percentile = (1 - share_newer) * P_old + share_newer * P_new, cell by cell.

    This is what licenses the correction. The contamination is not in either
    side of the window, it is in the empirical weight between them, so replacing
    that weight is a targeted repair rather than a different measure.
    """
    rng = np.random.default_rng(5)
    rows = []
    for r in range(20):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=35 * r)
        for b in range(4):
            rows.append((f"R{r}", "Org", date, f"b{b}", "2023-01-01",
                         float(rng.normal())))
    panel = side_balanced_percentile(
        within_benchmark_percentile(make_panel(rows), window_days=182), window_days=182
    )
    recon = ((1 - panel["share_newer"]) * panel["pct_old_side"].fillna(0)
             + panel["share_newer"] * panel["pct_new_side"].fillna(0))
    assert np.nanmax(np.abs(recon - panel["percentile"])) < 1e-9


def test_a_cell_with_an_empty_side_has_no_balanced_value():
    # the earliest model on a benchmark has no older peer but itself, and the
    # latest has no newer peer; neither gets a filled-in one-sided number
    rows = [("R0", "Org", "2025-01-01", "b", "2024-01-01", 0.1),
            ("R1", "Org", "2025-03-01", "b", "2024-01-01", 0.5)]
    panel = side_balanced_percentile(
        within_benchmark_percentile(make_panel(rows), window_days=182), window_days=182
    )
    assert panel["pct_new_side"].isna().sum() == 1
    assert panel["pct_balanced"].isna().sum() >= 1


def test_balancing_shrinks_the_asymmetry_gradient():
    rows = []
    for i in range(24):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i)
        rows.append((f"R{i}", "Org", date, "old", "2023-01-01", 0.4 + 0.02 * i))
        if i >= 8:
            rows.append((f"R{i}", "Org", date, "late", "2025-01-01", 0.4 + 0.02 * i))
    panel = side_balanced_percentile(
        within_benchmark_percentile(make_panel(rows), window_days=182), window_days=182
    )

    def gradient(column):
        cells = panel.dropna(subset=[column, "share_newer"])
        x = cells["share_newer"].to_numpy()
        y = cells[column].to_numpy()
        return abs(np.cov(y, x, bias=True)[0, 1] / x.var())

    assert gradient("pct_balanced") < gradient("percentile")
