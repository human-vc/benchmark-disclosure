"""Pin the Epoch data snapshot.

`data/raw/` is gitignored, so two people running the same code against the hub
on different days get different numbers with nothing to warn them. Epoch's
Benchmarking Hub is updated continuously: a rebuild three days later added
twenty model-versions and moved every count in the README. This module makes
that drift loud instead of silent.

The manifest is checked in; the data it describes is not.
"""

import hashlib
import json
from pathlib import Path

from .config import INDEX_FILE, RAW, SNAPSHOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def fingerprint(raw: Path = RAW) -> dict:
    """Describe the raw tree on disk: per-file hash and row count."""
    if not raw.exists():
        raise FileNotFoundError(f"no raw data at {raw}; run python -m src.download_data")

    files = {}
    for path in sorted(raw.rglob("*.csv")):
        rel = path.relative_to(raw).as_posix()
        files[rel] = {"sha256": _sha256(path), "rows": _rows(path)}

    index = files.get(INDEX_FILE, {})
    return {
        "index_rows": index.get("rows", 0),
        "csv_files": len(files),
        "files": files,
    }


def capture(raw: Path = RAW, zip_sha256: str = "", captured: str = "") -> dict:
    """Write the manifest. Deliberate act: it redefines what reproduces."""
    manifest = fingerprint(raw)
    manifest["zip_sha256"] = zip_sha256
    manifest["captured"] = captured
    SNAPSHOT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load() -> dict | None:
    if not SNAPSHOT.exists():
        return None
    return json.loads(SNAPSHOT.read_text())


def compare(raw: Path = RAW) -> dict:
    """Diff the tree on disk against the checked-in manifest."""
    pinned = load()
    if pinned is None:
        return {"status": "unpinned", "added": [], "removed": [], "changed": []}

    current = fingerprint(raw)
    have, want = current["files"], pinned["files"]

    added = sorted(set(have) - set(want))
    removed = sorted(set(want) - set(have))
    changed = sorted(
        name for name in set(have) & set(want)
        if have[name]["sha256"] != want[name]["sha256"]
    )

    return {
        "status": "match" if not (added or removed or changed) else "drift",
        "added": added,
        "removed": removed,
        "changed": changed,
        "pinned_index_rows": pinned.get("index_rows", 0),
        "current_index_rows": current["index_rows"],
        "captured": pinned.get("captured", ""),
    }


def stamp(raw: Path = RAW) -> str:
    """One line for a script header, so every printed number carries its vintage."""
    result = compare(raw)
    if result["status"] == "unpinned":
        return "snapshot: UNPINNED (no data/snapshot.json; numbers are not reproducible)"
    if result["status"] == "match":
        return f"snapshot: {result['captured']} ({result['current_index_rows']} model-versions)"

    drifted = len(result["added"]) + len(result["removed"]) + len(result["changed"])
    return (
        f"snapshot: DRIFT from {result['captured']} "
        f"({drifted} files differ; index {result['pinned_index_rows']} pinned "
        f"vs {result['current_index_rows']} on disk) "
        f"- numbers will not match the checked-in build"
    )


def main():
    result = compare()
    print(stamp())
    for label in ("added", "removed", "changed"):
        names = result.get(label, [])
        if names:
            print(f"  {label}: {len(names)}")
            for name in names[:10]:
                print(f"    {name}")
            if len(names) > 10:
                print(f"    ... and {len(names) - 10} more")


if __name__ == "__main__":
    main()
