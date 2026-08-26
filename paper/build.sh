#!/usr/bin/env bash
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
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
else
  PAGES=$(grep -oE "Output written on main\.xdv \([0-9]+ page" main.log 2>/dev/null \
          | grep -oE "[0-9]+" | head -1)
fi
PAGES=${PAGES:-?}

CONTENT=$(pdftotext -layout main.pdf - 2>/dev/null | awk -v RS='\f' '
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
