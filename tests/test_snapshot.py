"""The snapshot pin and the numbers file.

Both exist for the same reason: a quantity that moves because its inputs moved
should say so, rather than be discovered when a figure stops matching prose.
"""

import json

import pandas as pd
import pytest

from src import snapshot


class TestManifest:
    def test_changed_content_is_detected_under_an_unchanged_name(self, tmp_path):
        """The case that would otherwise pass unremarked: same file name, new
        bytes. Epoch republishes in place."""
        root = tmp_path / "raw"
        root.mkdir()
        (root / "scores.csv").write_text("a,b\n1,2\n")
        manifest = tmp_path / "manifest.csv"
        snapshot.write(root, manifest)

        (root / "scores.csv").write_text("a,b\n1,3\n")
        added, removed, changed = snapshot.verify(root, manifest)
        assert changed == ["scores.csv"]
        assert not added and not removed

    def test_additions_and_removals_are_separated(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "one.csv").write_text("x")
        manifest = tmp_path / "manifest.csv"
        snapshot.write(root, manifest)

        (root / "one.csv").unlink()
        (root / "two.csv").write_text("y")
        added, removed, changed = snapshot.verify(root, manifest)
        assert added == ["two.csv"] and removed == ["one.csv"] and changed == []

    def test_an_unchanged_snapshot_reports_clean(self, tmp_path, capsys):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "one.csv").write_text("x")
        manifest = tmp_path / "manifest.csv"
        snapshot.write(root, manifest)
        assert snapshot.report(root, manifest) is True

    def test_missing_manifest_is_not_silently_clean(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        assert snapshot.verify(root, tmp_path / "absent.csv") is None
        assert snapshot.report(root, tmp_path / "absent.csv") is False

    def test_nested_files_are_hashed(self, tmp_path):
        root = tmp_path / "raw"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "deep.csv").write_text("z")
        got = snapshot.scan(root)
        assert list(got["path"]) == ["sub/deep.csv"]


class TestNumbers:
    def test_every_reported_quantity_is_json_serialisable(self):
        """numpy scalars are not JSON, and a file that fails to write is a file
        the write-up quietly keeps quoting from a stale copy."""
        from src.config import INTERIM
        from src.numbers import collect

        if not (INTERIM / "panel.csv").exists():
            pytest.skip("panel not built")
        json.dumps(collect())

    def test_the_snapshot_digest_travels_with_the_numbers(self):
        from src.config import INTERIM
        from src.numbers import collect

        if not (INTERIM / "panel.csv").exists():
            pytest.skip("panel not built")
        got = collect()
        assert "snapshot_manifest_digest" in got
