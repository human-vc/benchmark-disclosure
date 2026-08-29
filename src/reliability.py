
import sys

import numpy as np
import pandas as pd

from .config import (ARTIFACTS, CODING_SHEET, INTERIM, RELIABILITY_SHEET, ROOT,
                     WORKLIST)

LEAKED = {
    "Anthropic | Claude Opus 4.5 | 2025-11-24",
    "Anthropic | Claude Opus 4.8 | 2026-05-28",
    "Google DeepMind | Gemini 3 Flash | 2025-12-17",
    "Google DeepMind | Gemini 3.5 Flash | 2026-05-19",
    "OpenAI | GPT-5 | 2025-08-07",
    "OpenAI | GPT-5.2 | 2025-12-11",
    "OpenAI | GPT-5.4 | 2026-03-05",
    "OpenAI | GPT-5.5 | 2026-04-23",
    "OpenAI | GPT-3.5 Turbo | 2023-11-06",
    "OpenAI | GPT-4 Turbo (Apr 2024) | 2024-04-09",
    "OpenAI | GPT-4 (Jun 2023) | 2023-06-13",
    "xAI | Grok 4.20 | 2026-02-17",
    "Baichuan | Baichuan2-13B | 2023-09-06",
    "Mistral AI | Mixtral 8x7B | 2023-12-11",
}

HIGH_SUSPICION = {"D", "E", "G"}
REPORTED = {"A", "B", "C"}
SAMPLE_SHARE = 0.20

KEEP = ["release_id", "organization", "model_name", "release_date", "n_cells",
        "family_rank", "source_tier", "source_url", "extra_source_urls",
        "source_date", "artifact_kind"]
BLANK = ["reported_slugs", "coder", "flagged_for_review", "notes"]

SECOND_EVIDENCE = ROOT / "data" / "artifacts_second_coder.csv"

def cohens_kappa(a, b):
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

def provider(artifacts):
    return artifacts["release_id"].str.split(" | ", regex=False).str[0]

def draw_releases(artifacts, share=SAMPLE_SHARE, seed=0, exclude=LEAKED):
    coded = artifacts[artifacts["fetch_status"] == "ok"]
    if exclude:
        coded = coded[~coded["release_id"].isin(exclude)]
    if coded.empty:
        return coded

    coded = coded.assign(provider=provider(coded))
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in coded.groupby("provider", sort=True):
        k = max(1, int(round(len(group) * share)))
        picks.append(group.iloc[rng.choice(len(group), size=k, replace=False)])
    return pd.concat(picks).sort_values(["provider", "release_date"])

def blank_sheet(sample):
    sheet = sample.reindex(columns=KEEP + BLANK).copy()
    for column in BLANK:
        sheet[column] = ""
    sheet["fetch_status"] = "ok"
    return sheet

def derive_from(evidence):
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

    redraw = "--redraw" in sys.argv
    if redraw and SECOND_EVIDENCE.exists():
        existing = pd.read_csv(SECOND_EVIDENCE, dtype=str)
        if existing["reported_slugs"].notna().any():
            print(f"{SECOND_EVIDENCE} has filled rows; refusing to redraw over "
                  f"work already done. Move it aside first.")
            sys.exit(1)

    if redraw or not SECOND_EVIDENCE.exists():
        sample = draw_releases(artifacts)
        if sample.empty:
            print("nothing extracted yet, so there is nothing to double-code.")
            sys.exit(0)
        blank_sheet(sample).to_csv(SECOND_EVIDENCE, index=False)
        readable = (artifacts["fetch_status"] == "ok").sum()
        share = len(sample) / readable
        print(f"drew {len(sample)} releases ({share:.0%} of {readable} readable) "
              f"across {sample['provider'].nunique()} providers")
        if LEAKED:
            print(f"  {len(LEAKED)} release(s) held out as un-blindable:")
            for release in sorted(LEAKED):
                print(f"    {release}")
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
