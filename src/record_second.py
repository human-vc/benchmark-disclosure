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

import pandas as pd

from .config import ROOT

SECOND_EVIDENCE = ROOT / "data" / "artifacts_second_coder.csv"
WRITABLE = ("reported_slugs", "coder", "flagged_for_review", "notes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_id")
    parser.add_argument("--slugs", default="",
                        help="evidence syntax; empty means the artifact "
                             "reports none of the eligible benchmarks")
    parser.add_argument("--coder", required=True)
    parser.add_argument("--flag", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

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
