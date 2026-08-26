import numpy as np
import pandas as pd
import pytest

from src.config import RELEASE_COL


def make_panel(rows):
    """rows: (release, org, release_date, slug, bench_date, score)"""
    frame = pd.DataFrame(
        rows,
        columns=[RELEASE_COL, "Organization", "Release date", "slug",
                 "benchmark_release_date", "score"],
    )
    frame["Release date"] = pd.to_datetime(frame["Release date"])
    frame["benchmark_release_date"] = pd.to_datetime(frame["benchmark_release_date"])
    frame["Model accessibility"] = "API access"
    frame["Model name"] = frame[RELEASE_COL]
    frame["primary_org"] = frame["Organization"].str.split(",").str[0].str.strip()
    frame["date_known"] = frame["benchmark_release_date"].notna()
    predates = frame["benchmark_release_date"] < frame["Release date"]
    frame["eligible"] = predates & frame["date_known"]
    frame["placebo"] = ~predates & frame["date_known"]
    return frame


@pytest.fixture
def tiny_panel():
    return make_panel([
        ("A", "Org1", "2025-01-01", "b1", "2024-01-01", 0.1),
        ("B", "Org1", "2025-02-01", "b1", "2024-01-01", 0.5),
        ("C", "Org1", "2025-03-01", "b1", "2024-01-01", 0.9),
        ("D", "Org2", "2027-01-01", "b1", "2024-01-01", 0.95),
    ])
