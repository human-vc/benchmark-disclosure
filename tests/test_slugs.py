"""Slug matching between Epoch's metadata names and its score filenames.

The two sides normalise differently and an unmapped pair is silently dropped,
which previously cost the panel eight whole benchmarks that were documented as
"no score file in the public bundle" when the files were shipping all along.
A stale alias table must fail loudly rather than shrink the sample quietly.
"""
import pandas as pd
import pytest

from src.build_matrix import KNOWN_UNJOINED, SLUG_ALIASES, collapse_meta, slugify


def test_previously_lost_benchmarks_now_map_to_their_score_files():
    expected = {
        "HellaSwag": "hella_swag",
        "Winogrande": "wino_grande",
        "TriviaQA": "trivia_qa",
        "ScienceQA": "science_qa",
        "OpenBookQA": "open_book_qa",
        "CadEval": "cad_eval",
        "Remote Labor Index": "rli",
    }
    for name, slug in expected.items():
        assert slugify(name) == slug, name


def test_osworld_versions_stay_separate():
    """OSWorld and OSWorld 2.0 are different benchmarks with different score
    files. Merging them attributed OSWorld's 2024 date to OSWorld 2.0, which
    would admit models that shipped before the benchmark existed."""
    assert slugify("OSWorld") == "os_world"
    assert slugify("OSWorld 2.0") == "osworld_2"
    assert slugify("OSWorld") != slugify("OSWorld 2.0")


def test_collapse_meta_yields_one_row_per_slug():
    meta = pd.DataFrame({
        "benchmark_name": ["V1", "V2", "Other"],
        "slug": ["x", "x", "y"],
        "benchmark_release_date": pd.to_datetime(["2024-01-01", "2025-01-01", None]),
    })
    out = collapse_meta(meta)
    assert len(out) == 2
    assert not out["slug"].duplicated().any()
    row = out[out["slug"] == "x"].iloc[0]
    # latest known date wins: attributing the older variant's date to a newer
    # benchmark is the anti-conservative direction for the gate.
    assert row["benchmark_release_date"] == pd.Timestamp("2025-01-01")
    assert row["n_collapsed"] == 2


def test_known_unjoined_is_explicit_and_small():
    """If this grows, someone has accepted data loss. Make them do it on purpose."""
    assert KNOWN_UNJOINED == {"ebr_bench"}


def test_alias_targets_are_not_themselves_aliased():
    for source, target in SLUG_ALIASES.items():
        assert target not in SLUG_ALIASES, f"{source} -> {target} chains"
