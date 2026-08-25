"""Pin the upstream data snapshot, and notice when it moves.

`download_data` fetches Epoch's bundle from a live URL, so re-running the
pipeline a month later silently analyses a different dataset. Everything
downstream still runs, every number still prints, and nothing says the numbers
changed because the inputs did. That is the failure mode this exists to close:
a drift is a finding, and it has to announce itself rather than be discovered
when a figure stops matching the text.

The manifest records a SHA-256 per raw file. `verify` reports what was added,
what was removed, and what changed content -- and changed content is the one
that matters, because a file with the same name and different bytes is the case
that would otherwise pass unremarked.
"""

import hashlib
import sys

import pandas as pd

from .config import RAW, ROOT

MANIFEST = ROOT / "data" / "snapshot_manifest.csv"


def digest(path, chunk=1 << 20):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            sha.update(block)
    return sha.hexdigest()


def scan(root=RAW):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(root)),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            })
    return pd.DataFrame(rows)


def write(root=RAW, manifest=MANIFEST):
    current = scan(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(manifest, index=False)
    return current


def verify(root=RAW, manifest=MANIFEST):
    """Return (added, removed, changed) against the pinned manifest."""
    if not manifest.exists():
        return None
    pinned = pd.read_csv(manifest).set_index("path")
    current = scan(root).set_index("path")
    added = sorted(set(current.index) - set(pinned.index))
    removed = sorted(set(pinned.index) - set(current.index))
    shared = sorted(set(pinned.index) & set(current.index))
    changed = [p for p in shared
               if pinned.loc[p, "sha256"] != current.loc[p, "sha256"]]
    return added, removed, changed


def report(root=RAW, manifest=MANIFEST):
    """Print the drift. Returns True when the snapshot matches the pin."""
    result = verify(root, manifest)
    if result is None:
        print(f"no snapshot manifest at {manifest}; run `python -m src.snapshot pin`")
        return False
    added, removed, changed = result
    if not (added or removed or changed):
        print(f"snapshot matches the pinned manifest ({len(scan(root))} files)")
        return True
    print(f"SNAPSHOT DRIFT against {manifest}:")
    if changed:
        print(f"  {len(changed)} file(s) changed content -- the ones that move "
              f"published numbers without changing any file name:")
        for path in changed[:10]:
            print(f"    {path}")
    if added:
        print(f"  {len(added)} file(s) added, e.g. {added[:5]}")
    if removed:
        print(f"  {len(removed)} file(s) removed, e.g. {removed[:5]}")
    print("  Every quantity downstream is now computed against different "
          "inputs than the pin. Re-pin deliberately, and re-derive.")
    return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pin":
        current = write()
        print(f"pinned {len(current)} files to {MANIFEST}")
        return
    sys.exit(0 if report() else 1)


if __name__ == "__main__":
    main()
