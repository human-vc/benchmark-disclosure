"""Fetch and search provider artifacts so extraction is reproducible."""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data" / "artifact_cache"

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
    """Download to a content-addressed cache. Returns the local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    suffix = ".pdf" if ".pdf" in url.lower() else ".html"
    path = CACHE / f"{name or key}{suffix}"
    if path.exists():
        return path
    subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-o", str(path), url], check=True
    )
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
    """Which pages mention which benchmark terms."""
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
