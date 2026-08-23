"""The external exhibit, pinned.

This is the only claim in the project about numbers someone else published, so
it carries the most risk and gets the most explicit test. The three invariants
the exhibit rests on are checked separately, because a referee's first objection
is that HELM changed what it measured, and the answer has to be that they did
not.
"""
import json

import pytest

from src.helm_external import (
    CONTROL,
    FROZEN,
    SCENARIOS,
    control,
    frozen_cells,
    frozen_models,
    headline_drift,
    load,
    scenario_columns_are_stable,
    summary,
)


@pytest.fixture(scope="module")
def helm():
    if not FROZEN.exists():
        pytest.skip("frozen HELM evidence not present")
    return load()


def test_all_fourteen_releases_are_present(helm):
    _, order = helm
    assert len(order) == 14
    assert order[0] == "v1.0.0" and order[-1] == "v1.13.0"


def test_the_scenarios_never_change(helm):
    # the model column is relabelled at v1.10.0; the ten scenarios are not
    payload, order = helm
    assert scenario_columns_are_stable(payload, order)


def test_the_pool_grows(helm):
    payload, order = helm
    first = len(payload["releases"][order[0]]["rows"])
    last = len(payload["releases"][order[-1]]["rows"])
    assert last > 2 * first


def test_the_frozen_evidence_is_bit_identical(helm):
    """240 published scores, unchanged across every release. If this ever fails,
    the exhibit is void and must not be published."""
    payload, order = helm
    models = frozen_models(payload, order)
    assert len(models) == 24
    cells = frozen_cells(payload, order, models)
    assert cells["cells"] == 24 * len(SCENARIOS)
    assert cells["changed"] == 0
    assert cells["max_spread"] == 0.0


def test_every_frozen_model_moves_anyway(helm):
    payload, order = helm
    head = headline_drift(payload, order, frozen_models(payload, order))
    assert head["moved"] == head["models"] == 24


def test_the_drift_is_not_a_monotone_rescaling(helm):
    """The load-bearing claim is the reordering, not the drift.

    Everyone's win rate falling as newer models enter is arithmetic. Pairs
    changing order is not, and it is what a leaderboard exists to prevent.
    """
    payload, order = helm
    head = headline_drift(payload, order, frozen_models(payload, order))
    assert head["reversals_endpoint"] >= 1
    assert head["reversals_ever"] >= head["reversals_endpoint"]
    assert head["rose"] > 0 and head["fell"] > 0


def test_drift_loads_on_initial_standing(helm):
    payload, order = helm
    head = headline_drift(payload, order, frozen_models(payload, order))
    assert head["corr_change_with_initial"] < -0.5


def test_every_release_carries_a_source_hash(helm):
    payload, order = helm
    for version in order:
        digest = payload["releases"][version]["sha256_of_source_file"]
        assert len(digest) == 64


def test_summary_runs(helm):
    report = summary()
    assert report["frozen_cells"]["changed"] == 0


@pytest.fixture(scope="module")
def helm_control():
    if not CONTROL.exists():
        pytest.skip("frozen HELM Capabilities evidence not present")
    return control()


def test_the_control_grows_like_the_treatment(helm_control):
    # if the control pool were static, its zero drift would prove nothing
    assert helm_control["pool_size_last"] > 2 * helm_control["pool_size_first"]
    assert helm_control["releases"] >= 14


def test_the_control_evidence_is_also_frozen(helm_control):
    assert helm_control["scenario_columns_stable"]
    assert helm_control["frozen_cells"]["changed"] == 0
    assert helm_control["frozen_cells"]["cells"] > 0


def test_an_absolute_headline_does_not_drift(helm_control):
    """The prediction, stated before the test was run.

    Same evaluator, same infrastructure, same release cadence, comparable pool
    growth. The only difference is that this headline is an absolute mean rather
    than a pool-relative win rate, and nothing moves.
    """
    assert helm_control["headline_kind"] == "absolute"
    assert helm_control["moved"] == 0
    assert helm_control["reversals_endpoint"] == 0
    assert helm_control["reversals_ever"] == 0


def test_treatment_and_control_diverge(helm, helm_control):
    """The comparison is the result. Neither half carries it alone."""
    payload, order = helm
    treated = headline_drift(payload, order, frozen_models(payload, order))
    assert treated["moved"] == treated["models"]
    assert helm_control["moved"] == 0


def test_correlation_is_undefined_rather_than_zero_when_nothing_moves(helm_control):
    # a control that moves nothing has no variance to correlate; reporting 0.0
    # here would read as "no relationship" rather than "no variation"
    payload, order = load(CONTROL)
    head = headline_drift(payload, order, frozen_models(payload, order),
                          headline="Mean score")
    assert head["corr_change_with_initial"] is None
