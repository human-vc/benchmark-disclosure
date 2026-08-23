#!/usr/bin/env bash
# Compile and report the four things that matter after every single edit:
# hard errors, undefined references and citations, overfull boxes, and the page
# count against the venue limit. Batching edits and compiling once hides which
# edit broke what.
set -uo pipefail
cd "$(dirname "$0")"

LIMIT=${1:-9}
LOG=$(mktemp)
tectonic -X compile main.tex --outdir . --keep-logs >"$LOG" 2>&1
STATUS=$?

echo "=== build ==="
if [ $STATUS -ne 0 ]; then
  echo "FAILED"
  grep -E "^error|! " "$LOG" | head -20
  exit 1
fi

ERR=$(grep -cE "^error" "$LOG" || true)
UNDEF=$(grep -oE "(Citation|Reference) \`[^']+' (on page [0-9]+ )?undefined" main.log 2>/dev/null | sort -u || true)
OVER=$(grep -cE "Overfull \\\\[hv]box" main.log 2>/dev/null); OVER=${OVER:-0}
# tectonic reports the count for main.xdv, and pdfinfo is authoritative when
# it is installed. Fall back to the xdv line rather than guessing from the PDF.
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
else
  PAGES=$(grep -oE "Output written on main\.xdv \([0-9]+ page" main.log 2>/dev/null \
          | grep -oE "[0-9]+" | head -1)
fi
PAGES=${PAGES:-?}

# The venue counts CONTENT pages. References and the appendix are excluded, so
# the total is the wrong number to trim against and reporting it invites cutting
# material that costs nothing.
CONTENT=$(pdftotext -layout main.pdf - 2>/dev/null | awk -v RS='\f' '
  # the submission style prints line numbers, so References is not at line start
  /(^|[[:space:]])References([[:space:]]|$)/ {print NR-1; found=1; exit}
  END {if (!found) print NR}')
CONTENT=${CONTENT:-?}

echo "errors      : ${ERR:-0}"
echo "overfull    : ${OVER:-0}"
echo "pages       : ${PAGES} total, ${CONTENT} content (limit ${LIMIT})"
if [ -n "$UNDEF" ]; then
  echo "undefined   :"; echo "$UNDEF" | sed 's/^/  /'
else
  echo "undefined   : none"
fi

if [ "${CONTENT}" != "?" ] && [ "${CONTENT}" -gt "${LIMIT}" ] 2>/dev/null; then
  echo "OVER THE CONTENT LIMIT by $((CONTENT - LIMIT))"
else
  echo "within the content limit"
fi
rm -f "$LOG"
