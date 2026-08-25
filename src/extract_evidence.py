"""Show a coder every place a release artifact could be reporting a benchmark.

Usage:
    python -m src.extract_evidence "<release_id>" <url> [--context N] [--all]
    python -m src.extract_evidence --release "<release_id>"     # cached url from artifacts.csv

The coding protocol asks a factual question -- what does this artifact report --
and the answer has to come from the artifact, not from memory. This surfaces the
candidate passages so the answer is made against text that a second coder can
pull up at the same URL and see for themselves.

What it deliberately does not do is decide. It prints hits and non-hits; whether
a hit is a score *for this model* rather than a mention in prose, a baseline in
a related-work table, or a competitor's number is exactly the judgment the
protocol reserves for the coder, and the A/B/C split turns on it.
"""

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd

from .benchmark_aliases import PATTERNS, hits, is_weak
from .config import ARTIFACTS, WORKLIST
from .artifact_tools import fetch


def html_to_text(raw):
    """Strip a page to readable text, keeping table cells separated.

    Provider blog posts put their benchmark table in markup; collapsing tags to
    nothing would run "GPQA" straight into "88.7" from the next cell and make
    the context unreadable, so block-level tags become newlines and inline ones
    become spaces.
    """
    raw = re.sub(r"(?is)<(script|style|svg|noscript)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/tr|/h[1-6]|/li)\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)<(/td|/th)\s*>", " | ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\xa0]+", " ", raw)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)


# A fetch can succeed at the HTTP level and still return no artifact: gated
# Hugging Face repos answer 200 with a login notice, and CDNs answer with a
# challenge page. Both parse to a short document in which nothing matches,
# which reads exactly like a provider that reported no benchmarks. That is the
# one error this study cannot absorb, so a suspicious body is refused rather
# than coded.
REFUSAL_MARKERS = (
    "is restricted. you must have access",
    "you need to agree to share your contact information",
    "please log in", "sign in to continue", "access to model",
    "enable javascript and cookies to continue", "just a moment...",
    "403 forbidden", "404 not found", "page not found",
    "attention required! | cloudflare",
)
MIN_PLAUSIBLE_CHARS = 800


def guard(text, url):
    """Raise unless the fetched body looks like a real artifact."""
    head = " ".join(text[:2000].split()).lower()
    for marker in REFUSAL_MARKERS:
        if marker in head:
            raise SystemExit(
                f"REFUSED: {url}\n  body carries {marker!r}; this is an access "
                f"wall, not an artifact. Record fetch_status=blocked and find "
                f"another official source -- do not code it as unreported."
            )
    if len(text) < MIN_PLAUSIBLE_CHARS:
        raise SystemExit(
            f"REFUSED: {url}\n  body is only {len(text)} chars, too short to be "
            f"a release artifact. Record fetch_status=blocked and find another "
            f"official source -- do not code it as unreported."
        )


def artifact_text(path):
    if path.suffix == ".pdf":
        from .artifact_tools import pdf_pages
        return "\n".join(f"\n[p{n}] {t}" for n, t in pdf_pages(path))
    return html_to_text(path.read_text(errors="replace"))


# Hugging Face cards are markdown, so the results picture is as likely to be
# ![bench](url) as it is an <img> tag. Missing the markdown form was worth a
# whole release's table.
IMG_SRC = re.compile(
    r"""(?i)<img[^>]*?\bsrc=["']?([^"'>\s]+)|!\[[^\]]*\]\(([^)\s]+)"""
)

# Chrome, not content. Listing every image on a marketing page buries the one
# that is the results table, so the obvious furniture is dropped by name.
NOT_A_TABLE = re.compile(
    r"(?i)(logo|icon|avatar|favicon|banner|header|footer|badge|arrow|spinner"
    r"|thumb|profile|social|share|author|/emoji|sprite|\.svg$)"
)


def image_tables(raw, url):
    """Image sources on the page whose name suggests a results table.

    Provider launch posts routinely render their benchmark table as a picture.
    Text extraction then returns the prose and misses every number, which reads
    as a provider that reported nothing -- the false-omission failure this
    pipeline is built to avoid. Surfacing the candidate images means the coder
    opens them and reads the table, the way a human reader of the page would.
    """
    from urllib.parse import urljoin
    seen, out = set(), []
    for match in IMG_SRC.finditer(raw):
        candidate = match.group(1) or match.group(2)
        if NOT_A_TABLE.search(candidate):
            continue
        # Page HTML escapes query separators, and an unescaped &amp; in a
        # CDN URL fetches a different object or nothing at all.
        src = urljoin(url, html.unescape(candidate))
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


TABLE = re.compile(r"(?is)<table\b.*?</table>")
ROW = re.compile(r"(?is)<tr\b.*?</tr>")
CELL = re.compile(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>")


MD_ROW = re.compile(r"^\s*\|.*\|\s*$")
MD_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def markdown_tables(raw):
    """Contiguous runs of markdown pipe rows, as compact rows.

    Hugging Face cards are markdown and most of them put the results table in
    pipe syntax, which never reaches the <table> renderer.
    """
    out, current = [], []
    for line in raw.splitlines():
        if MD_ROW.match(line):
            if not MD_RULE.match(line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                current.append(" | ".join(cells))
        elif current:
            if len(current) > 1:
                out.append(current)
            current = []
    if len(current) > 1:
        out.append(current)
    return out


def render_tables(raw, keep=None):
    """Every <table> on the page as compact pipe-separated rows.

    Provider cards increasingly ship the results table as hand-styled HTML in
    which each cell carries eighty characters of inline CSS, so the surrounding
    context a term search prints is almost all style attributes and the row is
    unreadable. Stripping to cells makes the table legible, which is what the
    coder needs in order to pick the right column.
    """
    out = []
    blocks = [None] * 0
    for index, rows in enumerate(markdown_tables(raw), 1):
        if keep and not any(keep.lower() in row.lower() for row in rows):
            continue
        out.append(f"--- markdown table {index} ({len(rows)} rows) ---\n"
                   + "\n".join(rows))
    for index, block in enumerate(TABLE.findall(raw), 1):
        rows = []
        for row in ROW.findall(block):
            cells = [" ".join(html.unescape(re.sub(r"(?s)<[^>]+>", " ", cell)).split())
                     for cell in CELL.findall(row)]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue
        if keep and not any(keep.lower() in row.lower() for row in rows):
            continue
        out.append(f"--- table {index} ({len(rows)} rows) ---\n" + "\n".join(rows))
    return out


def pdf_images(path, out_dir, min_bytes=40_000):
    """Write out every sizeable embedded image in a PDF, return the paths.

    Google's Gemini model cards put the whole results table on one page as a
    single raster. pypdf then extracts that page as a couple of hundred
    characters of prose and every benchmark on it reads as unreported -- the
    Gemini 3 Pro card scored zero alias hits across all 25 of its eligible
    benchmarks before this existed. The threshold drops logos and rules.
    """
    import pypdf

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    reader = pypdf.PdfReader(str(path))
    for number, page in enumerate(reader.pages, 1):
        try:
            images = list(page.images)
        except Exception:
            continue
        for index, image in enumerate(images):
            if len(image.data) < min_bytes:
                continue
            target = out_dir / f"{path.stem}_p{number}_{index}{Path(image.name).suffix or '.png'}"
            target.write_bytes(image.data)
            written.append(target)
    return written


def eligible_slugs(release_id):
    worklist = pd.read_csv(WORKLIST)
    rows = worklist[(worklist["release_id"] == release_id)
                    & (worklist["group"] == "eligible")]
    return sorted(rows["benchmark_slug"])


NUMBER = re.compile(r"\d+(?:\.\d+)?\s*%?")


def report(text, slugs, context=160, cap=6):
    found = hits(text, slugs)
    for slug in slugs:
        occurrences = found.get(slug, [])
        if not occurrences:
            continue
        print(f"\n### {slug}  ({len(occurrences)} hit(s))")
        seen = set()
        shown = 0
        for term, start, end in occurrences:
            left = max(0, start - context)
            snippet = " ".join(text[left:end + context].split())
            key = snippet[:80]
            if key in seen:
                continue
            seen.add(key)
            mark = " [weak alias]" if is_weak(slug, term) else ""
            print(f"  <{term}>{mark}  ...{snippet}...")
            shown += 1
            if shown >= cap:
                remaining = len(occurrences) - shown
                if remaining > 0:
                    print(f"  ... {remaining} further hit(s) not shown")
                break
    missing = [s for s in slugs if s not in found]
    print(f"\n### no alias hit ({len(missing)}): {' '.join(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_id")
    parser.add_argument("url", nargs="?")
    parser.add_argument("--context", type=int, default=160)
    parser.add_argument("--cap", type=int, default=6)
    parser.add_argument("--vega", nargs="?", const="", default=None,
                        help="read embedded chart data; optional title filter")
    parser.add_argument("--pdf-images", action="store_true",
                        help="write out embedded PDF images and list them")
    parser.add_argument("--tables", nargs="?", const="", default=None,
                        help="render HTML tables; optional substring filter")
    parser.add_argument("--all", action="store_true",
                        help="search every panel slug, not just this release's")
    args = parser.parse_args()

    url = args.url
    if url is None:
        table = pd.read_csv(ARTIFACTS)
        row = table[table["release_id"] == args.release_id]
        if row.empty or not isinstance(row.iloc[0]["source_url"], str):
            print(f"no source_url recorded for {args.release_id}")
            sys.exit(1)
        url = row.iloc[0]["source_url"]

    path = fetch(url)
    if args.vega is not None:
        from .vega_charts import charts, render

        raw = path.read_text(errors="replace")
        for title, scores in charts(raw):
            if args.vega and args.vega.lower() not in title.lower():
                continue
            print(render(title, scores))
        return

    if args.pdf_images:
        if path.suffix != ".pdf":
            print(f"{path} is not a PDF")
            return
        for image in pdf_images(path, path.parent / "pdf_images"):
            print(image)
        return

    text = artifact_text(path)
    guard(text, url)
    if path.suffix == ".html":
        pictures = image_tables(path.read_text(errors="replace"), url)
        if pictures:
            print("\n### candidate image tables -- open these before coding "
                  "anything as unreported")
            for src in pictures[:12]:
                print(f"  {src}")
    if args.tables is not None and path.suffix == ".html":
        for block in render_tables(path.read_text(errors="replace"),
                                   args.tables or None):
            print(block)
            print()
        return
    slugs = sorted(PATTERNS) if args.all else eligible_slugs(args.release_id)
    print(f"{args.release_id}\n{url}\n{path}  ({len(text):,} chars, "
          f"{len(slugs)} slug(s) in scope)")
    report(text, slugs, args.context, args.cap)


if __name__ == "__main__":
    main()
