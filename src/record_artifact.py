
import argparse

import pandas as pd

from .config import ARTIFACTS

FIELDS = ("source_tier", "source_url", "extra_source_urls", "source_date",
          "artifact_kind",
          "reported_slugs", "coder", "flagged_for_review", "fetch_status",
          "notes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_id")
    parser.add_argument("--url", required=True)
    parser.add_argument("--extra", default="",
                        help="space-separated co-released official artifacts")
    parser.add_argument("--tier", type=int, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--slugs", default="")
    parser.add_argument("--coder", default="kevin")
    parser.add_argument("--status", default="ok")
    parser.add_argument("--flag", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    table = pd.read_csv(ARTIFACTS, dtype=str)
    match = table["release_id"] == args.release_id
    if not match.any():
        raise SystemExit(f"unknown release_id: {args.release_id!r}")

    values = dict(source_tier=str(args.tier), source_url=args.url,
                  extra_source_urls=args.extra,
                  source_date=args.date, artifact_kind=args.kind,
                  reported_slugs=args.slugs, coder=args.coder,
                  flagged_for_review=args.flag, fetch_status=args.status,
                  notes=args.notes)
    for field in FIELDS:
        table.loc[match, field] = values[field]
    table.to_csv(ARTIFACTS, index=False)
    done = (table["fetch_status"] == "ok").sum()
    print(f"{args.release_id}: {args.slugs or '(nothing reported)'}")
    print(f"artifacts.csv now {done}/{len(table)} coded")

if __name__ == "__main__":
    main()
