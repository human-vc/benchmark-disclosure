"""The placebo group's null, and why it is not zero.

design.md nominates the eligible-versus-placebo contrast as identifying on the
ground that both sets are non-disclosures. The claim these tests pin down is
that the contrast is instead a property of the percentile's peer window: a
release sitting at the left edge of a benchmark's model coverage is ranked
against peers drawn almost entirely from models newer than itself. If a future
change to the percentile makes the window symmetric, the first test here should
start failing, and that would be good news worth noticing rather than a
regression to paper over.
"""
import numpy as np
import pandas as pd

from src.config import RELEASE_COL
from src.percentiles import within_benchmark_percentile
from src.placebo_calibration import (
    _absorb,
    balanced_contrast,
    contrast_by_release,
    decomposition,
    drop_capacity,
    null_calibration,
    permutation_null,
    _asymmetry_slope,
)
from tests.conftest import make_panel


def coverage_panel(edge=True):
    """Two benchmarks. The second one's model coverage starts late.

    Every release is scored on an old benchmark, always eligible. The second
    benchmark is introduced in 2025 and, as real evaluators do, is run only on
    models from shortly before its introduction onward. The handful of models
    that predate it therefore sit at the left boundary of its coverage and carry
    the placebo cells. Scores rise with release date, which is the pattern the
    windowed percentile is meant to absorb.

    With edge=False the second benchmark is old too, so no boundary exists and
    no artifact should appear.
    """
    late_date = "2025-01-01" if edge else "2022-06-01"
    first_scored = 8 if edge else 0
    rows = []
    for i in range(24):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i)
        score = 0.4 + 0.02 * i
        rows.append((f"R{i}", f"Org{i % 4}", date, "old", "2023-01-01", score))
        if i >= first_scored:
            rows.append((f"R{i}", f"Org{i % 4}", date, "late", late_date, score))
    return make_panel(rows)


def test_share_newer_is_extreme_at_the_edges_of_coverage():
    # the focal cell is inside its own window, so the share never quite reaches
    # one or zero; what matters is that the edges are one-sided
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    late = panel[panel["slug"] == "late"].sort_values("Release date")
    assert late["share_newer"].iloc[0] > 0.8
    assert late["share_newer"].iloc[-1] < 0.2


def test_placebo_cells_sit_where_the_window_is_one_sided():
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    placebo = panel[panel["placebo"]]["share_newer"].mean()
    eligible = panel[panel["eligible"]]["share_newer"].mean()
    assert placebo > eligible
    assert placebo > 0.5


def test_contrast_is_positive_without_any_disclosure_information():
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    contrast = contrast_by_release(panel)
    assert not contrast.empty
    assert contrast["contrast"].mean() > 0, "the null of the identifying estimator"


def test_conditioning_on_the_peer_window_absorbs_the_contrast():
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    ladder = decomposition(panel)
    raw = abs(ladder.loc[ladder["step"] == "release FE", "placebo_coef"].iloc[0])
    conditioned = abs(ladder.loc[ladder["step"] == "+ peer count", "placebo_coef"].iloc[0])
    assert raw > 0, "fixture produced no contrast to absorb"
    assert conditioned < raw


def test_two_way_absorption_beats_one_sequential_pass():
    """The bug this replaces asserted the opposite and shipped for a day.

    Demeaning by release and then by benchmark removes both sets of effects only
    on a balanced panel. This panel is not balanced: the late benchmark is scored
    on a subset of releases. A single sequential pass therefore leaves most of the
    benchmark effect in place, and the module reported that benchmark composition
    explained none of the contrast when it explains most of it.
    """
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    cells = panel[panel["eligible"] | panel["placebo"]].copy()
    cells["placebo_i"] = cells["placebo"].astype(float)
    release = cells[RELEASE_COL].to_numpy()
    benchmark = cells["slug"].to_numpy()

    absorbed = _absorb(cells["percentile"], release, benchmark)
    by_release = pd.Series(absorbed).groupby(release).mean().abs().max()
    by_benchmark = pd.Series(absorbed).groupby(benchmark).mean().abs().max()
    assert by_release < 1e-6, "release effects not absorbed"
    assert by_benchmark < 1e-6, "benchmark effects not absorbed"

    one_pass = pd.Series(cells["percentile"].to_numpy())
    one_pass = one_pass - one_pass.groupby(release).transform("mean")
    one_pass = one_pass - one_pass.groupby(benchmark).transform("mean")
    leftover = one_pass.groupby(release).mean().abs().max()
    assert leftover > by_release, "sequential demeaning should leave a residual here"


def test_decomposition_reports_the_jointly_conditioned_row():
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    steps = decomposition(panel)["step"].tolist()
    assert "two-way FE + both" in steps, "the fully conditioned row must be reported"


def test_permutation_null_is_near_zero_when_scores_carry_no_trend():
    """Separates a real bias from an accounting identity.

    share_newer and percentile come from the same peer set. If their relationship
    were mechanical it would survive destroying the capability trend. It does not.
    """
    rng = np.random.default_rng(11)
    rows = []
    for r in range(14):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=40 * r)
        for b in range(6):
            # staggered coverage starts, so share_newer varies within a release
            if r < 2 * b:
                continue
            rows.append((f"R{r}", f"Org{r % 3}", date, f"b{b}", "2023-01-01",
                         float(rng.normal())))
    panel = make_panel(rows)
    result = permutation_null(panel, draws=25, seed=2)
    assert np.isfinite(result["null_mean"]), result
    assert abs(result["null_mean"]) < 12, result
    assert result["draws"] == 25


def test_balanced_contrast_keeps_a_subset_and_reports_the_share():
    panel = within_benchmark_percentile(coverage_panel(), window_days=182)
    _, kept = balanced_contrast(panel)
    assert 0.0 <= kept <= 1.0


def test_drop_capacity_is_reported_at_two_definitions_and_the_strict_one_is_smaller():
    panel = within_benchmark_percentile(coverage_panel(edge=False), window_days=182)
    families = pd.DataFrame({
        "release_id": [f"R{i}" for i in range(24)],
        "family_id": ["fam"] * 24,
        "family_rank": list(range(1, 25)),
    })
    capacity = drop_capacity(panel, families, min_benchmarks=1)
    assert len(capacity) == 2
    assert capacity["pairs"].iloc[1] <= capacity["pairs"].iloc[0]


def test_null_calibration_falls_as_more_benchmarks_are_dropped():
    rng = np.random.default_rng(3)
    rows = []
    for r in range(12):
        for b in range(12):
            rows.append((f"R{r}", "Org", "2025-06-01", f"b{b}", "2024-01-01",
                         float(rng.normal())))
    panel = within_benchmark_percentile(make_panel(rows), window_days=182)
    sds = null_calibration(panel, drops=(1, 3), draws=120, min_benchmarks=8)["null_sd"]
    assert sds.iloc[0] > sds.iloc[1]


def test_contrast_by_release_survives_a_panel_with_no_placebo_cells():
    # the pivot lacks the column entirely rather than producing an empty one,
    # so an .empty check alone does not catch this
    rows = [("R0", "Org", "2025-06-01", "b", "2024-01-01", 0.5),
            ("R1", "Org", "2025-07-01", "b", "2024-01-01", 0.7)]
    panel = within_benchmark_percentile(make_panel(rows), window_days=182)
    assert panel["placebo"].sum() == 0
    assert contrast_by_release(panel).empty


def test_absorb_matches_a_naive_reference():
    """The fast path factorizes once and uses bincount. Pin it to the obvious
    implementation, because a silent divergence here moves every coefficient."""
    rng = np.random.default_rng(4)
    n = 400
    g1 = rng.integers(0, 25, n)
    g2 = rng.integers(0, 9, n)
    values = rng.normal(size=n) + g1 * 0.3 - g2 * 0.7

    def naive(v, a, b, tol=1e-10, max_iter=5000):
        s = pd.Series(np.asarray(v, dtype=float))
        for _ in range(max_iter):
            prev = s.to_numpy().copy()
            s = s - s.groupby(a).transform("mean")
            s = s - s.groupby(b).transform("mean")
            if np.max(np.abs(s.to_numpy() - prev)) < tol:
                break
        return s.to_numpy()

    assert np.max(np.abs(_absorb(values, g1, g2) - naive(values, g1, g2))) < 1e-9


def test_absorb_removes_both_sets_of_effects_on_an_unbalanced_panel():
    rng = np.random.default_rng(6)
    g1 = np.repeat(np.arange(30), 7)[: 30 * 7]
    g2 = rng.integers(0, 11, len(g1))
    keep = rng.random(len(g1)) > 0.35          # unbalance it on purpose
    g1, g2 = g1[keep], g2[keep]
    values = rng.normal(size=len(g1)) + g1 * 0.2 + g2 * 0.5

    out = _absorb(values, g1, g2)
    assert abs(pd.Series(out).groupby(g1).mean()).max() < 1e-8
    assert abs(pd.Series(out).groupby(g2).mean()).max() < 1e-8


def test_null_calibration_affine_shortcut_matches_the_direct_gap():
    """null_calibration computes mean(drawn) - mean(rest) in closed form.
    Check the algebra against the definition on concrete numbers."""
    rng = np.random.default_rng(9)
    values = rng.normal(size=13)
    n, k = len(values), 3
    total = values.sum()
    for _ in range(50):
        picks = rng.choice(n, k, replace=False)
        direct = values[picks].mean() - np.delete(values, picks).mean()
        shortcut = values[picks].sum() * n / (k * (n - k)) - total / (n - k)
        assert abs(direct - shortcut) < 1e-10


def test_permutation_null_observed_matches_the_direct_slope():
    """The two code paths must agree on the observed value by construction.

    The benchmarks need different coverage starts, or share_newer is constant
    within a release, the release demeaning annihilates it, and the slope is
    noise over noise rather than a number.
    """
    rng = np.random.default_rng(12)
    rows = []
    for r in range(16):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=40 * r)
        for b in range(5):
            if r < 2 * b:
                continue
            rows.append((f"R{r}", f"Org{r % 4}", date, f"b{b}", "2023-01-01",
                         0.3 + 0.02 * r + float(rng.normal()) * 0.05))
    panel = within_benchmark_percentile(make_panel(rows), window_days=182)
    direct = _asymmetry_slope(panel)
    assert np.isfinite(direct), "fixture has no within-release variation to fit"
    result = permutation_null(panel, draws=5, seed=1)
    assert abs(result["observed"] - direct) < 1e-6


def test_a_degenerate_regressor_returns_nan_rather_than_noise():
    """Constant share_newer within release must not produce a number.

    Every benchmark here shares a release date, so demeaning by release leaves
    floating-point dust. A truthiness check on the variance lets that through.
    """
    rng = np.random.default_rng(12)
    rows = []
    for r in range(16):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=40 * r)
        for b in range(5):
            rows.append((f"R{r}", f"Org{r % 4}", date, f"b{b}", "2023-01-01",
                         float(rng.normal())))
    panel = within_benchmark_percentile(make_panel(rows), window_days=182)
    assert np.isnan(_asymmetry_slope(panel))
    assert np.isnan(permutation_null(panel, draws=3, seed=1)["observed"])
