"""Family linkage invariants.

A drop is only defined relative to a predecessor in the same family, so a bad
linkage does not produce a noisy estimate -- it produces a wrong one.
"""
import pandas as pd
import pytest

from src.families import build_families, derive_family, load_families, predecessors
from src.config import RELEASE_COL


def test_version_markers_are_stripped_but_tiers_are_kept():
    assert derive_family("Anthropic", "Claude Opus 4.5") == "Anthropic / Claude Opus"
    assert derive_family("Anthropic", "Claude Sonnet 4.6") == "Anthropic / Claude Sonnet"
    assert derive_family("Anthropic", "Claude Opus 4.5") != derive_family(
        "Anthropic", "Claude Sonnet 4.5"
    )


def test_disambiguating_dates_collapse_but_sizes_do_not():
    """(Jun 2025) is version information; (7B) is product identity."""
    assert derive_family("G", "Gemini 2.5 Pro (Jun 2025)") == derive_family(
        "G", "Gemini 2.5 Pro (May 2025)"
    )
    assert derive_family("A", "Qwen2.5-Coder (7B)") != derive_family(
        "A", "Qwen2.5-Coder (32B)"
    )


def test_organizations_never_share_a_family():
    assert derive_family("OpenAI", "Foo 1").split(" / ")[0] == "OpenAI"
    assert derive_family("Meta AI", "Foo 1") != derive_family("OpenAI", "Foo 1")


def test_ranks_are_dense_and_ordered_by_date():
    panel = pd.DataFrame({
        RELEASE_COL: ["r1", "r2", "r3"],
        "Organization": ["O", "O", "O"],
        "Model name": ["Thing 1", "Thing 3", "Thing 2"],
        "Release date": pd.to_datetime(["2024-01-01", "2026-01-01", "2025-01-01"]),
    })
    fam = build_families(panel)
    assert list(fam.sort_values("family_rank")["model_name"]) == [
        "Thing 1", "Thing 2", "Thing 3"
    ]
    assert list(fam["family_rank"].sort_values()) == [1, 2, 3]


def test_predecessors_returns_earlier_releases_newest_first():
    fam = pd.DataFrame({
        "family_id": ["F"] * 3,
        "release_id": ["a", "b", "c"],
        "family_rank": [1, 2, 3],
        "model_name": ["A", "B", "C"],
    })
    got = predecessors(fam, "c")
    assert list(got["release_id"]) == ["b", "a"]
    assert predecessors(fam, "a").empty


def test_checked_in_families_file_validates():
    fam = load_families()
    assert not fam["release_id"].duplicated().any()
    assert (fam["family_rank"] >= 1).all()
