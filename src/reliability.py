"""Double-coding draw and inter-rater agreement.

protocol/coding-protocol.md commits to double-coding a random 20% and reporting
Cohen's kappa on two splits: the collapsed high-suspicion (D, E, G) versus
low-suspicion (A, B, C, F, H, I) dichotomy, and the full nine-category
assignment.

**The second coding is a second extraction, not a second category sheet.** In
this pipeline a human never assigns an ORBIT category: they record what a
release's artifact reports, and `derive_coding` turns that into categories by
rule. Handing a second coder a sheet of cells to re-categorise would measure
nothing, because two people reading the same evidence file must produce the same
categories by construction -- the rules are deterministic. The judgment that can
actually differ is upstream: *does this artifact report this benchmark, and in
what form*. So the draw samples releases, the second coder re-reads those
artifacts into a blank evidence sheet, and kappa is computed on the categories
each extraction derives.

What the blank sheet keeps and what it hides matters too. It keeps
`source_url` and `extra_source_urls`, because both coders must read the same
documents -- blinding the artifact would measure the second coder's search
skills, not their reading. It hides everything that encodes the first coder's
judgment: the reported set, variant names, the prior-model attribution, the
notes, and the flags. A flag in particular says "the first coder was unsure
here", which is precisely the nudge that inflates agreement.

ORBIT's own validation is the benchmark to compare against and it is not
flattering -- trained reviewers applying the G/H distinction reached 92%
sensitivity and 77% specificity against ground truth. This instrument should
not be expected to beat that. The disagreement rate is reported rather than
suppressed.
"""

import sys

import numpy as np
import pandas as pd

from .config import (ARTIFACTS, CODING_SHEET, INTERIM, RELIABILITY_SHEET, ROOT,
                     WORKLIST)

HIGH_SUSPICION = {"D", "E", "G"}
REPORTED = {"A", "B", "C"}
SAMPLE_SHARE = 0.20

# What the second coder is given, and what is withheld.
KEEP = ["release_id", "organization", "model_name", "release_date", "n_cells",
        "family_rank", "source_tier", "source_url", "extra_source_urls",
        "source_date", "artifact_kind"]
BLANK = ["reported_slugs", "coder", "flagged_for_review", "notes"]

SECOND_EVIDENCE = ROOT / "data" / "artifacts_second_coder.csv"


def cohens_kappa(a, b):
    """Unweighted Cohen's kappa. Returns (kappa, observed agreement)."""
    a = pd.Series(list(a), dtype="object")
    b = pd.Series(list(b), dtype="object")
    categories = sorted(set(a.dropna()) | set(b.dropna()))
    if len(categories) < 2:
        return np.nan, float((a == b).mean())

    observed = float((a == b).mean())
    expected = sum(
        (a == category).mean() * (b == category).mean() for category in categories
    )
    if np.isclose(expected, 1.0):
        return np.nan, observed
    return (observed - expected) / (1 - expected), observed


def draw_releases(artifacts, share=SAMPLE_SHARE, seed=0):
    """Releases for re-extraction, stratified by provider.

    Stratifying matters: providers differ in artifact style -- one ships a text
    table, the next a picture, the next a client-rendered chart -- so a simple
    random draw can leave a provider unchecked and hide a systematic misreading
    of one company's documents.
    """
    coded = artifacts[artifacts["fetch_status"] == "ok"]
    if coded.empty:
        return coded

    rng = np.random.default_rng(seed)
    picks = []
    for _, group in coded.groupby("organization"):
        k = max(1, int(round(len(group) * share)))
        picks.append(group.iloc[rng.choice(len(group), size=k, replace=False)])
    return pd.concat(picks).sort_values(["organization", "release_date"])


def blank_sheet(sample):
    sheet = sample.reindex(columns=KEEP + BLANK).copy()
    for column in BLANK:
        sheet[column] = ""
    sheet["fetch_status"] = "ok"
    return sheet


def derive_from(evidence):
    """Run the same derivation rules over one evidence file."""
    from .derive_coding import build
    from .families import load_families

    worklist = pd.read_csv(WORKLIST)
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    return build(worklist, evidence, load_families(), panel)


def report(first, second):
    merged = first.merge(
        second, on=["release_id", "benchmark_slug"], suffixes=("_1", "_2")
    )
    if merged.empty:
        print("no overlapping cells between the two extractions")
        return

    a = merged["orbit_category_1"]
    b = merged["orbit_category_2"]
    kappa_full, agree_full = cohens_kappa(a, b)

    a_high = a.isin(HIGH_SUSPICION).map({True: "high", False: "low"})
    b_high = b.isin(HIGH_SUSPICION).map({True: "high", False: "low"})
    kappa_split, agree_split = cohens_kappa(a_high, b_high)

    # The primitive judgment, before any rule is applied: did the artifact
    # report this benchmark at all? Everything else is derived from it, so a
    # kappa here that is worse than the category kappa would mean the rules are
    # hiding disagreement rather than the coders sharing it.
    a_rep = a.isin(REPORTED).map({True: "reported", False: "not"})
    b_rep = b.isin(REPORTED).map({True: "reported", False: "not"})
    kappa_rep, agree_rep = cohens_kappa(a_rep, b_rep)

    print(f"double-coded cells: {len(merged)} "
          f"across {merged['release_id'].nunique()} releases")
    print("\nreported vs not (the judgment the coder actually makes):")
    print(f"  agreement {agree_rep:.1%}   kappa {kappa_rep:.3f}")
    print("\ncollapsed high (D/E/G) vs low (A/B/C/F/H/I):")
    print(f"  agreement {agree_split:.1%}   kappa {kappa_split:.3f}")
    print("full nine-category assignment:")
    print(f"  agreement {agree_full:.1%}   kappa {kappa_full:.3f}")

    disagreements = merged[a != b]
    print(f"\ndisagreements: {len(disagreements)} "
          f"({len(disagreements)/len(merged):.1%})")
    if len(disagreements):
        pairs = (
            disagreements.groupby(["orbit_category_1", "orbit_category_2"])
            .size()
            .sort_values(ascending=False)
        )
        print("  most common confusions:")
        for (first_code, second_code), count in pairs.head(8).items():
            print(f"    {first_code} vs {second_code}: {count}")
        worst = (
            disagreements.groupby("release_id").size().sort_values(ascending=False)
        )
        print("  releases contributing most disagreement:")
        for release, count in worst.head(5).items():
            print(f"    {release}: {count}")

    print("\nORBIT's own reviewers: 92% sensitivity, 77% specificity on G/H.")
    print("Do not expect to beat that; report the gap either way.")


def main():
    if not ARTIFACTS.exists():
        print(f"no artifact evidence at {ARTIFACTS}")
        sys.exit(1)
    artifacts = pd.read_csv(ARTIFACTS, dtype=str)

    if not SECOND_EVIDENCE.exists():
        sample = draw_releases(artifacts)
        if sample.empty:
            print("nothing extracted yet, so there is nothing to double-code.")
            sys.exit(0)
        blank_sheet(sample).to_csv(SECOND_EVIDENCE, index=False)
        share = len(sample) / (artifacts["fetch_status"] == "ok").sum()
        print(f"drew {len(sample)} releases ({share:.0%}) for re-extraction")
        print(f"wrote {SECOND_EVIDENCE}")
        print("\nIt carries the pinned source_url and extra_source_urls, so the")
        print("second coder reads the same documents; reported_slugs, notes and")
        print("flags are blank. Fill reported_slugs in the evidence syntax:")
        print("  slug=value      numeric score reported")
        print("  slug            named, no number given")
        print("  slug~Variant=v  a variant reported instead")
        print("  slug!reason     absent, with a benign reason quoted")
        print("Then rerun to get kappa.")
        sys.exit(0)

    second = pd.read_csv(SECOND_EVIDENCE, dtype=str)
    second = second[second["reported_slugs"].notna()]
    if second.empty:
        print(f"{SECOND_EVIDENCE} exists but no release is filled in yet.")
        sys.exit(0)

    done = set(second["release_id"])
    first = artifacts[artifacts["release_id"].isin(done)
                      & (artifacts["fetch_status"] == "ok")]
    report(derive_from(first), derive_from(second))


if __name__ == "__main__":
    main()
