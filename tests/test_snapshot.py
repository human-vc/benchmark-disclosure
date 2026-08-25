"""Snapshot drift detection."""
import json

import pytest

from src import snapshot


@pytest.fixture
def raw_tree(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "epoch_capabilities_index.csv").write_text("model,score\na,1\nb,2\n")
    (raw / "gpqa_diamond.csv").write_text("model,score\na,0.5\n")
    nested = raw / "additional_eci_data"
    nested.mkdir()
    (nested / "slopes.csv").write_text("bench,slope\nx,1.0\n")
    return raw


def test_fingerprint_counts_rows_excluding_header(raw_tree):
    fp = snapshot.fingerprint(raw_tree)
    assert fp["index_rows"] == 2
    assert fp["csv_files"] == 3
    assert fp["files"]["additional_eci_data/slopes.csv"]["rows"] == 1


def test_missing_raw_tree_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot.fingerprint(tmp_path / "absent")


def test_absent_manifest_is_unpinned_not_match(raw_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT", tmp_path / "snapshot.json")
    assert snapshot.compare(raw_tree)["status"] == "unpinned"
    assert "UNPINNED" in snapshot.stamp(raw_tree)


def test_identical_tree_matches(raw_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT", tmp_path / "snapshot.json")
    snapshot.capture(raw_tree, captured="2026-08-17")
    assert snapshot.compare(raw_tree)["status"] == "match"


def test_edited_file_is_caught_by_content(raw_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT", tmp_path / "snapshot.json")
    snapshot.capture(raw_tree, captured="2026-08-17")

    (raw_tree / "gpqa_diamond.csv").write_text("model,score\na,0.9\n")

    result = snapshot.compare(raw_tree)
    assert result["status"] == "drift"
    assert result["changed"] == ["gpqa_diamond.csv"]
    assert "DRIFT" in snapshot.stamp(raw_tree)


def test_new_benchmark_file_is_drift(raw_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT", tmp_path / "snapshot.json")
    snapshot.capture(raw_tree, captured="2026-08-17")
    (raw_tree / "hle_external.csv").write_text("model,score\na,0.3\n")

    result = snapshot.compare(raw_tree)
    assert result["status"] == "drift"
    assert result["added"] == ["hle_external.csv"]


def test_stamp_reports_both_index_counts_on_drift(raw_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT", tmp_path / "snapshot.json")
    snapshot.capture(raw_tree, captured="2026-08-17")
    (raw_tree / "epoch_capabilities_index.csv").write_text("model,score\na,1\nb,2\nc,3\n")

    line = snapshot.stamp(raw_tree)
    assert "2 pinned" in line and "3 on disk" in line


def test_checked_in_manifest_is_wellformed():
    pinned = snapshot.load()
    assert pinned is not None, "data/snapshot.json should be checked in"
    assert pinned["index_rows"] > 0
    assert pinned["files"], "manifest lists no files"
    for record in pinned["files"].values():
        assert len(record["sha256"]) == 64
