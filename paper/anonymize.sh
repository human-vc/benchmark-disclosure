#!/usr/bin/env bash
# Build the anonymous supplementary archive for a double-blind submission.
#
# The repository source is already clean: no author names appear in any tracked
# file, and the compiled PDF carries no /Author field. The identity leak is
# entirely in git metadata, where both authors appear by name and email and one
# email is a GitHub noreply address embedding a numeric user id that resolves to
# an account. Mirroring the repository would carry that history across, so this
# builds a fresh tree with no .git directory rather than trying to scrub one.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-dist/anonymous-artifact.zip}
STAGE=$(mktemp -d)
mkdir -p "$(dirname "$OUT")"

# Tracked source only, so nothing untracked or local leaks in. data/raw is
# gitignored by design; the snapshot manifest is what makes it reproducible.
git ls-files -z \
  | grep -zv '^paper/\.baseline/' \
  | while IFS= read -r -d '' f; do
      mkdir -p "$STAGE/$(dirname "$f")"
      cp "$f" "$STAGE/$f"
    done

cat > "$STAGE/README.md" <<'EOF'
# Anonymous artifact

Code and derived data for the submission. Author-identifying information has
been removed and no version-control history is included.

## Reproducing the numbers

    pip install -r requirements.txt
    python -m src.download_data        # fetch the evaluator's public bundle
    python -m src.snapshot             # confirm it matches the pinned build
    python -m src.build_matrix
    python -m src.paper_numbers        # writes data/paper_numbers.json

`data/raw/` is not redistributed here. It is fetched from the public source and
verified against `data/snapshot.json`, which records a SHA-256 and row count for
every file in the build the manuscript reports. If the upstream data has moved
on, `python -m src.snapshot` will say so rather than silently producing
different numbers.

Every number in the manuscript is written by `python -m src.paper_numbers` into
`data/paper_numbers.json`; none is typed by hand.

The external leaderboard evidence in `data/external/` is frozen at the bytes
used, with a SHA-256 per source file, because a public leaderboard is a moving
target and the argument depends on the reader seeing what we saw.

    python -m src.helm_external        # reproduces the external exhibit
    python -m pytest tests/ -q         # 114 tests
EOF

rm -f "$OUT"
( cd "$STAGE" && zip -qr "$OLDPWD/$OUT" . -x '*.DS_Store' )
rm -rf "$STAGE"

echo "wrote $OUT"
echo
echo "residual identity check:"
if unzip -p "$OUT" '*' 2>/dev/null | grep -qiE 'jacob|crainic|kevzho|kevin zhou|human-vc'; then
  echo "  FAIL: identifying string found in the archive"
  exit 1
else
  echo "  no author names, usernames or org names found in archive contents"
fi
unzip -l "$OUT" | tail -1
