
from __future__ import annotations

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
    if not raw.exists():
        raise FileNotFoundError(
            f"no raw data at {raw}; run python -m src.download_data")

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

def capture(raw: Path = RAW, zip_sha256: str = "", captured: str = "",
            manifest: Path = SNAPSHOT) -> dict:
    fingerprinted = fingerprint(raw)
    fingerprinted["zip_sha256"] = zip_sha256
    fingerprinted["captured"] = captured
    manifest.write_text(json.dumps(fingerprinted, indent=2, sort_keys=True) + "\n")
    return fingerprinted

def load(manifest: Path = SNAPSHOT) -> dict | None:
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text())

def compare(raw: Path = RAW, manifest: Path = SNAPSHOT) -> dict:
    pinned = load(manifest)
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

def stamp(raw: Path = RAW, manifest: Path = SNAPSHOT) -> str:
    result = compare(raw, manifest)
    if result["status"] == "unpinned":
        return f"snapshot: UNPINNED (no {manifest.name}; numbers are not reproducible)"
    if result["status"] == "match":
        return f"snapshot: {result['captured']} ({result['current_index_rows']} model-versions)"

    drifted = len(result["added"]) + len(result["removed"]) + len(result["changed"])
    return (
        f"snapshot: DRIFT from {result['captured']} "
        f"({drifted} files differ; index {result['pinned_index_rows']} pinned "
        f"vs {result['current_index_rows']} on disk) "
        f"- numbers will not match the checked-in build"
    )

def report(raw: Path = RAW, manifest: Path = SNAPSHOT) -> bool:
    result = compare(raw, manifest)
    print(stamp(raw, manifest))
    if result["status"] == "unpinned":
        print(f"  no manifest at {manifest}; run `python -m src.download_data`")
        return False
    if result["status"] == "match":
        return True

    if result["changed"]:
        print(f"  {len(result['changed'])} file(s) changed content -- the ones "
              f"that move published numbers without changing any file name:")
        for name in result["changed"][:10]:
            print(f"    {name}")
    for label in ("added", "removed"):
        names = result[label]
        if names:
            print(f"  {len(names)} file(s) {label}, e.g. {names[:5]}")
    print("  Every quantity downstream is now computed against different "
          "inputs than the pin. Re-pin deliberately, and re-derive.")
    return False

def main():
    raise SystemExit(0 if report() else 1)

if __name__ == "__main__":
    main()
