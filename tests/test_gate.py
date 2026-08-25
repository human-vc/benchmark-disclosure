"""Temporal gate boundaries."""
import pandas as pd

from src.config import RELEASE_COL
from tests.conftest import make_panel


def test_same_day_is_not_eligible():
    panel = make_panel([("A", "O", "2025-01-01", "b", "2025-01-01", 0.5)])
    assert not panel["eligible"].iloc[0]
    assert panel["placebo"].iloc[0]


def test_day_before_is_eligible():
    panel = make_panel([("A", "O", "2025-01-02", "b", "2025-01-01", 0.5)])
    assert panel["eligible"].iloc[0]
    assert not panel["placebo"].iloc[0]


def test_unknown_vintage_is_in_neither_set():
    panel = make_panel([("A", "O", "2025-01-01", "b", None, 0.5)])
    assert not panel["eligible"].iloc[0]
    assert not panel["placebo"].iloc[0]
    assert not panel["date_known"].iloc[0]


def test_eligible_and_placebo_are_disjoint():
    panel = make_panel([
        ("A", "O", "2025-01-01", "b1", "2024-01-01", 0.5),
        ("A", "O", "2025-01-01", "b2", "2026-01-01", 0.5),
        ("A", "O", "2025-01-01", "b3", None, 0.5),
    ])
    assert not (panel["eligible"] & panel["placebo"]).any()
    assert panel["eligible"].sum() == 1
    assert panel["placebo"].sum() == 1
