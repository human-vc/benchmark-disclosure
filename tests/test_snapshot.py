
import json

import pytest

from src import snapshot

class TestManifest:
    def _pin(self, tmp_path, files):
        root = tmp_path / "raw"
        root.mkdir(exist_ok=True)
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        manifest = tmp_path / "snapshot.json"
        snapshot.capture(root, captured="test", manifest=manifest)
        return root, manifest

    def test_changed_content_is_detected_under_an_unchanged_name(self, tmp_path):
        root, manifest = self._pin(tmp_path, {"scores.csv": "a,b\n1,2\n"})

        (root / "scores.csv").write_text("a,b\n1,3\n")
        result = snapshot.compare(root, manifest)
        assert result["changed"] == ["scores.csv"]
        assert not result["added"] and not result["removed"]
        assert result["status"] == "drift"

    def test_additions_and_removals_are_separated(self, tmp_path):
        root, manifest = self._pin(tmp_path, {"one.csv": "x"})

        (root / "one.csv").unlink()
        (root / "two.csv").write_text("y")
        result = snapshot.compare(root, manifest)
        assert result["added"] == ["two.csv"]
        assert result["removed"] == ["one.csv"]
        assert result["changed"] == []

    def test_an_unchanged_snapshot_reports_clean(self, tmp_path):
        root, manifest = self._pin(tmp_path, {"one.csv": "x"})
        assert snapshot.compare(root, manifest)["status"] == "match"
        assert snapshot.report(root, manifest) is True

    def test_missing_manifest_is_not_silently_clean(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        absent = tmp_path / "absent.json"
        assert snapshot.load(absent) is None
        assert snapshot.compare(root, absent)["status"] == "unpinned"
        assert snapshot.report(root, absent) is False

    def test_nested_files_are_hashed(self, tmp_path):
        root = tmp_path / "raw"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "deep.csv").write_text("z")
        assert list(snapshot.fingerprint(root)["files"]) == ["sub/deep.csv"]

    def test_row_counts_are_pinned_alongside_the_hash(self, tmp_path):
        root, manifest = self._pin(tmp_path, {"scores.csv": "a,b\n1,2\n3,4\n"})
        assert snapshot.load(manifest)["files"]["scores.csv"]["rows"] == 2

class TestNumbers:
    def test_every_reported_quantity_is_json_serialisable(self):
        from src.config import INTERIM
        from src.paper_numbers import collect

        if not (INTERIM / "panel.csv").exists():
            pytest.skip("panel not built")
        _, numbers = collect()
        json.dumps(numbers)

    def test_the_snapshot_stamp_travels_with_the_numbers(self):
        from src.config import ROOT

        emitted = ROOT / "data" / "paper_numbers.json"
        if not emitted.exists():
            pytest.skip("paper_numbers.json not built")
        provenance = json.loads(emitted.read_text())["_provenance"]
        assert provenance["snapshot"].startswith("snapshot:")
        assert provenance["snapshot_status"] in {"match", "drift", "unpinned"}
