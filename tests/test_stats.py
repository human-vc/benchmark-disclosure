"""Inference that refuses rather than guesses."""
import io
import contextlib

import numpy as np
import pytest

from src.stats import (
    MIN_CLUSTERS,
    bootstrap_mean,
    ols,
    print_ols,
    randomization_test_mean,
    wild_cluster_bootstrap,
)


def design(n, rng, effect=0.0, clusters=None):
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    g = rng.integers(0, clusters, n) if clusters else None
    noise = rng.normal(size=n)
    if g is not None:
        noise = noise + rng.normal(size=clusters)[g]
    return X, effect * x + noise, g


def test_one_cluster_withholds_instead_of_reporting_certainty():
    rng = np.random.default_rng(0)
    X, y, g = design(2, rng, clusters=1)
    fit = ols(y, X, ["const", "x"], cluster=np.zeros(2))
    assert np.isnan(fit["se"]).all()
    assert np.isnan(fit["t"]).all()
    assert fit["inference"].startswith("withheld")
    assert np.isfinite(fit["beta"]).all(), "the point estimate is still reported"


def test_withheld_output_prints_no_stars():
    rng = np.random.default_rng(1)
    X, y, _ = design(2, rng)
    fit = ols(y, X, ["const", "x"], cluster=np.zeros(2))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_ols(fit, "title")
    printed = buffer.getvalue()
    assert "*" not in printed
    assert "se not reported" in printed
    assert "withheld" in printed


def test_enough_clusters_reports_normally():
    rng = np.random.default_rng(2)
    X, y, g = design(600, rng, effect=1.0, clusters=MIN_CLUSTERS + 20)
    fit = ols(y, X, ["const", "x"], cluster=g)
    assert fit["inference"] == "cluster-robust"
    assert np.isfinite(fit["se"]).all()
    assert abs(fit["beta"][1] - 1.0) < 0.25


def test_unclustered_fit_still_works():
    rng = np.random.default_rng(3)
    X, y, _ = design(300, rng, effect=2.0)
    fit = ols(y, X, ["const", "x"])
    assert fit["inference"] == "HC1"
    assert abs(fit["beta"][1] - 2.0) < 0.2


def test_underdetermined_fit_withholds():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2, 4))
    fit = ols(rng.normal(size=2), X)
    assert np.isnan(fit["se"]).all()
    assert fit["inference"].startswith("withheld")


def test_bootstrap_interval_is_nan_below_the_threshold():
    rng = np.random.default_rng(5)
    values = rng.normal(size=40)
    g = rng.integers(0, 3, 40)
    mean, (lo, hi) = bootstrap_mean(values, cluster=g, draws=200)
    assert np.isfinite(mean)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_interval_exists_above_the_threshold():
    rng = np.random.default_rng(6)
    values = rng.normal(size=400) + 1.0
    g = rng.integers(0, MIN_CLUSTERS + 8, 400)
    mean, (lo, hi) = bootstrap_mean(values, cluster=g, draws=400)
    assert lo < mean < hi


def test_randomization_test_cannot_reject_with_one_cluster():
    values = np.array([8.6, 8.6])
    result = randomization_test_mean(values, cluster=np.zeros(2), draws=200)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["granular"]


def test_randomization_test_finds_a_real_effect():
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(30), 6)
    values = rng.normal(size=len(g)) * 0.4 + 2.0
    assert randomization_test_mean(values, cluster=g, draws=999)["p_value"] < 0.01


def test_randomization_test_is_calibrated_under_the_null():
    """Across repeated null datasets the p-values should not concentrate low."""
    rejected = 0
    trials = 40
    for seed in range(trials):
        rng = np.random.default_rng(100 + seed)
        g = np.repeat(np.arange(20), 4)
        values = rng.normal(size=len(g)) + rng.normal(size=20)[g]
        if randomization_test_mean(values, cluster=g, draws=299,
                                   seed=seed)["p_value"] < 0.05:
            rejected += 1
    assert rejected <= 6, f"rejected {rejected}/{trials} under the null"


def test_wild_cluster_bootstrap_rejects_a_real_effect():
    rng = np.random.default_rng(8)
    X, y, g = design(300, rng, effect=1.5, clusters=8)
    result = wild_cluster_bootstrap(y, X, g, index=1, draws=199)
    assert result["p_value"] < 0.05
    assert result["n_clusters"] == 8
    assert result["granular"]


def test_wild_cluster_bootstrap_does_not_reject_a_null_effect():
    rng = np.random.default_rng(9)
    X, y, g = design(300, rng, effect=0.0, clusters=8)
    assert wild_cluster_bootstrap(y, X, g, index=1, draws=199)["p_value"] > 0.10


def test_wild_bootstrap_reports_its_own_resolution_limit():
    rng = np.random.default_rng(10)
    X, y, g = design(120, rng, clusters=5)
    result = wild_cluster_bootstrap(y, X, g, index=1, draws=99)
    assert result["weights"] == "webb"
    assert result["distinct_draws_available"] == 6 ** 5
    rademacher = wild_cluster_bootstrap(y, X, g, index=1, draws=99,
                                        weights="rademacher")
    assert rademacher["distinct_draws_available"] == 2 ** 5
    assert rademacher["p_floor"] == 1 / (min(99, 2 ** 4) + 1)
    assert result["p_value"] >= 1 / 100
