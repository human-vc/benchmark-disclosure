"""Write one release's evidence into the second coder's sheet.

Same reasoning as `record_artifact`: every write fills the same field the same
way, and re-coding a release overwrites in place. The difference is what this
command refuses to touch. The pinned `source_url`, `source_tier`, `source_date`
and `artifact_kind` were carried over from the first extraction so that both
coders read the same documents, and a second coder who edited them would turn
the exercise into a comparison of two different readings of two different
pages. This writes only the judgment fields.
"""

import argparse
import re

import pandas as pd

from .benchmark_aliases import ALIASES
from .config import ROOT

SECOND_EVIDENCE = ROOT / "data" / "artifacts_second_coder.csv"
WRITABLE = ("reported_slugs", "coder", "flagged_for_review", "notes")


def check(evidence):
    """Reject evidence the derivation would misparse.

    Tokens are pipe-separated because a variant name carries spaces of its own
    -- "MATH full set, 0-shot CoT, not the level-5 subset" is one token. A
    space-separated line still parses: `derive_coding` reads the whole line as
    a single token, records its first slug with a garbage value, and treats
    every other benchmark on the line as unreported. That is a false omission
    for each one, which is the error direction that manufactures evidence for
    this study's hypothesis, and nothing downstream would flag it. So it is
    refused here rather than discovered in the kappa.
    """
    problems = []
    for token in [t for t in evidence.split("|") if t.strip()]:
        token = token.strip()
        if token.startswith("+"):
            continue
        slug = re.split(r"[=~!]", token, maxsplit=1)[0].strip()
        if slug not in ALIASES:
            problems.append(f"{slug!r} is not a benchmark in the panel")
            continue
        # A variant name and a quoted reason carry spaces of their own. A bare
        # slug and a numeric value do not, so whitespace there means several
        # tokens were joined by spaces instead of pipes.
        if "~" in token or "!" in token:
            continue
        rest = token[len(slug):].lstrip("=").strip()
        if any(c.isspace() for c in rest) or any(c.isspace() for c in slug):
            joined = [t for t in token.split() if "=" in t or t in ALIASES]
            problems.append(
                f"{slug!r} carries {len(joined)} tokens run together: {token!r}\n"
                f"    separate them with | not spaces")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_id")
    parser.add_argument("--slugs", default="",
                        help="pipe-separated evidence; empty means the "
                             "artifact reports none of the eligible benchmarks")
    parser.add_argument("--coder", required=True)
    parser.add_argument("--flag", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    problems = check(args.slugs)
    if problems:
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit("refusing to write evidence the derivation would misparse")

    sheet = pd.read_csv(SECOND_EVIDENCE, dtype=str)
    match = sheet["release_id"] == args.release_id
    if not match.any():
        raise SystemExit(f"not in the second-coder draw: {args.release_id!r}")

    values = dict(reported_slugs=args.slugs, coder=args.coder,
                  flagged_for_review=args.flag, notes=args.notes)
    for field in WRITABLE:
        sheet.loc[match, field] = values[field]
    sheet.to_csv(SECOND_EVIDENCE, index=False)

    done = sheet["coder"].notna().sum()
    print(f"{args.release_id}: {args.slugs or '(nothing reported)'}")
    print(f"second sheet now {done}/{len(sheet)} coded")


if __name__ == "__main__":
    main()
