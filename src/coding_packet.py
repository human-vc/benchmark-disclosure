"""Per-release reading packets for the second coder.

`artifacts_second_coder.csv` names the releases to re-extract and pins the URLs,
but it does not say which benchmarks are in each release's eligible choice set.
Without that the second coder would have to reconstruct the choice set from
`worklist.csv` by hand, and any release where they reconstructed it differently
would show up as coder disagreement when it is really a bookkeeping difference.
The choice set is fixed by the panel, not by judgment, so it is given.

What the packet gives and what it withholds follows the same line as the blank
sheet in `reliability.py`. It gives the artifact URLs and the eligible slugs
with their search aliases -- the shared setup both coders work from. It
withholds the independent score, the prior release, and everything in
`artifacts.csv`. The independent score is withheld deliberately: seeing "Epoch
has 59.6 here" while reading an artifact that prints 59.4 invites the coder to
resolve an ambiguity toward the number they were shown, and the A/B/C judgment
is supposed to be made from the artifact alone.
"""

import argparse

import pandas as pd

from .benchmark_aliases import ALIASES
from .config import ARTIFACTS, INTERIM, ROOT, WORKLIST

SECOND_EVIDENCE = ROOT / "data" / "artifacts_second_coder.csv"
DEFAULT_OUT = INTERIM / "second_coder_packets.md"
FULL_OUT = INTERIM / "coding_packets_all.md"

SYNTAX = """\
Evidence syntax for `reported_slugs` (space-separated):

    slug=value      a numeric score for THIS model is reported      -> A
    slug            named, but no number given                      -> B
    slug~Variant=v  a different variant reported instead            -> C
    slug!reason     absent, with a benign reason quoted in the text -> F
    +name           reported here, but not scored by Epoch (reverse gap)

Omit a slug entirely if the artifact does not mention it. Do not write a
category letter anywhere: categories are derived by rule from this line.
Leave `reported_slugs` empty (but set `coder`) for an artifact that reports
nothing at all -- that is a finding, not a skipped row.

Record **every** panel benchmark the artifact reports, not only the ones listed
for the release. The per-release list is the eligible choice set under the
current worklist, and the worklist is rebuilt whenever the upstream snapshot
moves: a benchmark outside today's list can be inside tomorrow's. Recording it
costs nothing, because `derive_coding` intersects the evidence with whatever
worklist is current. Failing to record it is a false omission that appears only
after the rebuild, when nobody is looking. The full slug table is at the foot of
this file.
"""


def packets(sheet, worklist, title="Second-coder reading packets"):
    eligible = worklist[worklist["group"] == "eligible"]
    by_release = dict(list(eligible.groupby("release_id")))

    out = [f"# {title}", ""]
    out.append(f"{len(sheet)} releases, "
               f"{sum(len(by_release.get(r, [])) for r in sheet['release_id'])} "
               "eligible cells.")
    out.append("")
    out.append(SYNTAX)
    out.append("---")
    out.append("")

    for _, release in sheet.iterrows():
        cells = by_release.get(release["release_id"])
        out.append(f"## {release['release_id']}")
        if cells is None or cells.empty:
            out.append("")
            out.append("**NOT IN THE CURRENT WORKLIST** — this release has no "
                       "eligible cells under the checked-in worklist, so "
                       "nothing derives from it today. Read and record it "
                       "anyway; see docs/reconciliation.md.")
        out.append("")
        if str(release.get("fetch_status")) == "blocked":
            out.append("**BLOCKED** — the first pass could not read this "
                       "artifact. Coding an unreadable artifact as silent is "
                       "the false-omission error that manufactures evidence "
                       "for the hypothesis; leave it blocked unless you can "
                       "actually read it.")
            out.append("")
        out.append(f"- artifact: {release['artifact_kind']} "
                   f"(tier {release['source_tier']}, "
                   f"dated {release['source_date']})")
        out.append(f"- read: {release['source_url']}")
        extra = release.get("extra_source_urls")
        if isinstance(extra, str) and extra.strip():
            for url in extra.split():
                out.append(f"- also official for this release: {url}")
        out.append("")

        if cells is None or cells.empty:
            out.append("    reported_slugs:")
            out.append("")
            continue

        out.append(f"Eligible benchmarks ({len(cells)}) -- search the artifact "
                   "for each surface form:")
        out.append("")
        for _, cell in cells.sort_values("benchmark_slug").iterrows():
            slug = cell["benchmark_slug"]
            terms = ", ".join(ALIASES.get(slug, [cell["benchmark_name"]]))
            out.append(f"- `{slug}` — {cell['benchmark_name']} — {terms}")
        out.append("")
        out.append("    reported_slugs:")
        out.append("")

    out.append("---")
    out.append("")
    out.append("## Every panel benchmark, with its surface forms")
    out.append("")
    out.append("Record any of these the artifact reports, whether or not it is "
               "listed for the release above.")
    out.append("")
    for slug in sorted(ALIASES):
        out.append(f"- `{slug}` — {', '.join(ALIASES[slug])}")
    out.append("")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--all", action="store_true",
                        help="packets for every release in the panel, not "
                             "just the reliability draw")
    args = parser.parse_args()

    if args.all:
        sheet = pd.read_csv(ARTIFACTS, dtype=str)
        worklist = pd.read_csv(WORKLIST, dtype=str)
        out = ROOT / (args.out or str(FULL_OUT))
        with open(out, "w") as handle:
            handle.write(packets(sheet, worklist,
                                 title="Reading packets — full panel"))
        print(f"wrote {out}")
        print(f"{len(sheet)} releases to read "
              f"({(sheet['fetch_status'] == 'blocked').sum()} blocked)")
        return

    if not SECOND_EVIDENCE.exists():
        raise SystemExit(f"no draw yet at {SECOND_EVIDENCE}; "
                         "run `python -m src.reliability` first")

    sheet = pd.read_csv(SECOND_EVIDENCE, dtype=str)
    worklist = pd.read_csv(WORKLIST, dtype=str)

    text = packets(sheet, worklist)
    out = ROOT / (args.out or str(DEFAULT_OUT))
    with open(out, "w") as handle:
        handle.write(text)
    print(f"wrote {out}")
    print(f"{len(sheet)} releases to read")


if __name__ == "__main__":
    main()
