"""Fetch and search provider artifacts so extraction is reproducible.

The coding sheet records a source_url per release. That is only meaningful if
someone else can go back to the URL and recover the same reported-benchmark
set, so the extraction runs through here rather than by hand.

Two things this exists to work around:

  - Provider release *blog posts* usually render their benchmark table as an
    image. Text extraction sees the prose and misses the table, which would
    silently code a reported benchmark as omitted -- a false drop, biased
    toward the study's own hypothesis. System cards and technical reports
    carry the same table as real text and are the right source anyway under
    the protocol's hierarchy.
  - Those PDFs run to hundreds of pages and tens of megabytes, past what a
    page fetch will return.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data" / "artifact_cache"

# Benchmark names worth locating in a long document. Deliberately broad: a term
# that hits nothing costs a line of output, a term that is missing costs a
# false omission.
SEARCH_TERMS = [
    "SWE-bench", "SWE bench", "GPQA", "AIME", "Terminal-Bench", "Terminal Bench",
    "MMLU", "MMMLU", "MMMU", "ARC-AGI", "ARC AGI", "HLE", "Humanity's Last Exam",
    "OSWorld", "tau-bench", "τ2", "TAU-bench", "BrowseComp", "Aider",
    "FrontierMath", "GDPval", "Vending", "MATH", "GSM8K", "HumanEval",
    "LiveBench", "SimpleQA", "Terminus", "CritPt", "WeirdML", "SimpleBench",
    "Cybench", "AgentHarm", "MRCR", "GraphWalks", "LiveCodeBench", "Codeforces",
    "MMLU-Pro", "DROP", "BBH", "HellaSwag", "WinoGrande", "TriviaQA", "PIQA",
    "OpenBookQA", "ScienceQA", "ANLI", "LAMBADA", "BoolQ", "SuperGLUE",
    "Chess", "OTIS", "Mock AIME", "USAMO", "IMO", "VPCT", "Balrog",
    "Fiction.Live", "GSO", "DeepResearch", "AlgoTune", "ALE-Bench",
    "Forecast", "MetR", "METR", "time horizon", "SciCode", "Video-MME",
    "EnigmaEval", "APEX", "RLI", "Remote Labor", "the Agent Company",
    "Lech Mazur", "ProofBench", "MirrorCode", "geobench", "GeoBench",
]


def fetch(url, name=None):
    """Download to a content-addressed cache. Returns the local path.

    The suffix is decided by sniffing the downloaded bytes, not by looking for
    ".pdf" in the URL. arxiv.org/pdf/2309.10305 serves a PDF from a URL with no
    ".pdf" in it, and an earlier version of this function stored it as .html and
    handed the raw binary to the HTML stripper. That does not fail loudly: it
    yields mojibake in which no benchmark name matches, so every benchmark in
    the artifact reads as unreported. Silent false omissions are the one error
    mode this study cannot tolerate, since they point the way the hypothesis
    points.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    stem = name or key
    for suffix in (".pdf", ".html"):
        existing = CACHE / f"{stem}{suffix}"
        if existing.exists():
            return existing

    staged = CACHE / f"{stem}.part"
    subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-o", str(staged), url], check=True
    )
    with open(staged, "rb") as handle:
        magic = handle.read(5)
    path = CACHE / f"{stem}{'.pdf' if magic.startswith(b'%PDF') else '.html'}"
    staged.replace(path)
    return path


def pdf_pages(path):
    import pypdf

    reader = pypdf.PdfReader(str(path))
    for index, page in enumerate(reader.pages):
        try:
            yield index + 1, page.extract_text() or ""
        except Exception:
            yield index + 1, ""


def locate(path, terms=SEARCH_TERMS):
    """Which pages mention which benchmark terms.

    Word-boundary matched. A substring search puts METR inside "symmetric",
    MATH inside "mathematics", DROP inside "dropped" and RLI inside "earlier",
    and a false hit here is worse than a miss: it would code a benchmark as
    reported when the artifact never mentions it.
    """
    patterns = {
        term: re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])",
                         re.IGNORECASE)
        for term in terms
    }
    hits = {}
    for number, text in pdf_pages(path):
        for term, pattern in patterns.items():
            if pattern.search(text):
                hits.setdefault(term, []).append(number)
    return hits


def page_text(path, numbers):
    out = []
    wanted = set(numbers)
    for number, text in pdf_pages(path):
        if number in wanted:
            out.append(f"--- page {number} ---\n{text}")
    return "\n\n".join(out)


def main():
    url = sys.argv[1]
    path = fetch(url)
    if path.suffix == ".pdf":
        if len(sys.argv) > 2:
            print(page_text(path, [int(x) for x in sys.argv[2].split(",")]))
        else:
            hits = locate(path)
            print(f"{path}  ({sum(1 for _ in pdf_pages(path))} pages)")
            for term, pages in sorted(hits.items(), key=lambda kv: -len(kv[1])):
                print(f"  {term:24} {pages[:14]}")
    else:
        print(path.read_text()[:5000])


if __name__ == "__main__":
    main()
