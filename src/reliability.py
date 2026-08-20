"""Double-coding draw and inter-rater agreement.

protocol/coding-protocol.md commits to double-coding a random 20% of rows and
reporting Cohen's kappa on two splits: the collapsed high-suspicion (D, E, G)
versus low-suspicion (A, B, C, F, H, I) dichotomy, and the full nine-category
assignment.

ORBIT's own validation is the benchmark to compare against and it is not
flattering -- trained reviewers applying the G/H distinction reached 92%
sensitivity and 77% specificity against ground truth. This instrument should
not be expected to beat that. The disagreement rate is reported rather than
suppressed.
"""

import sys

import numpy as np
import pandas as pd

from .config import CODING_SHEET, RELIABILITY_SHEET, ROOT

HIGH_SUSPICION = {"D", "E", "G"}
SAMPLE_SHARE = 0.20


def cohens_kappa(a, b):
    """Unweighted Cohen's kappa. Returns (kappa, observed agreement)."""
    a = pd.Series(list(a), dtype="object")
    b = pd.Series(list(b), dtype="object")
    categories = sorted(set(a.dropna()) | set(b.dropna()))
    if len(categories) < 2:
        return np.nan, float((a == b).mean())

    n = len(a)
    observed = float((a == b).mean())
    expected = sum(
        (a == category).mean() * (b == category).mean() for category in categories
    )
    if np.isclose(expected, 1.0):
        return np.nan, observed
    return (observed - expected) / (1 - expected), observed


def draw_sample(sheet, share=SAMPLE_SHARE, seed=0):
    """Random rows for second coding, stratified by provider.

    Stratifying matters: providers differ in artifact style, so a simple random
    draw can leave a provider entirely unchecked and hide a coder's systematic
    misreading of one company's model cards.
    """
    coded = sheet[sheet["orbit_category"].notna() & (sheet["orbit_category"] != "")]
    coded = coded[coded.get("coder", "") != "auto"]
    if coded.empty:
        return coded

    rng = np.random.default_rng(seed)
    picks = []
    for _, group in coded.groupby("organization"):
        k = max(1, int(round(len(group) * share)))
        picks.append(group.iloc[rng.choice(len(group), size=k, replace=False)])
    return pd.concat(picks).sort_values(["organization", "release_id"])


def report(first, second):
    merged = first.merge(
        second, on=["release_id", "benchmark_slug"], suffixes=("_1", "_2")
    )
    if merged.empty:
        print("no overlapping rows between the two codings")
        return

    a = merged["orbit_category_1"]
    b = merged["orbit_category_2"]
    kappa_full, agree_full = cohens_kappa(a, b)

    a_high = a.isin(HIGH_SUSPICION).map({True: "high", False: "low"})
    b_high = b.isin(HIGH_SUSPICION).map({True: "high", False: "low"})
    kappa_split, agree_split = cohens_kappa(a_high, b_high)

    print(f"double-coded rows: {len(merged)}")
    print(f"\ncollapsed high (D/E/G) vs low (A/B/C/F/H/I):")
    print(f"  agreement {agree_split:.1%}   kappa {kappa_split:.3f}")
    print(f"full nine-category assignment:")
    print(f"  agreement {agree_full:.1%}   kappa {kappa_full:.3f}")

    disagreements = merged[a != b]
    print(f"\ndisagreements: {len(disagreements)} ({len(disagreements)/len(merged):.1%})")
    if len(disagreements):
        pairs = (
            disagreements.groupby(["orbit_category_1", "orbit_category_2"])
            .size()
            .sort_values(ascending=False)
        )
        print("  most common confusions:")
        for (first_code, second_code), count in pairs.head(8).items():
            print(f"    {first_code} vs {second_code}: {count}")

    print("\nORBIT's own reviewers: 92% sensitivity, 77% specificity on G/H.")
    print("Do not expect to beat that; report the gap either way.")


def main():
    if not CODING_SHEET.exists():
        print(f"no coding sheet at {CODING_SHEET}")
        sys.exit(1)

    sheet = pd.read_csv(CODING_SHEET)

    if not RELIABILITY_SHEET.exists():
        sample = draw_sample(sheet)
        if sample.empty:
            print("nothing coded yet, so there is nothing to double-code.")
            print(f"code {CODING_SHEET} first, then rerun to draw the 20% sample.")
            sys.exit(0)
        blanked = sample.copy()
        for column in ("orbit_category", "reported", "reported_value",
                       "source_tier", "source_url", "notes", "coder"):
            if column in blanked.columns:
                blanked[column] = ""
        RELIABILITY_SHEET.parent.mkdir(parents=True, exist_ok=True)
        blanked.to_csv(RELIABILITY_SHEET, index=False)
        print(f"drew {len(sample)} rows ({len(sample)/len(sheet):.0%}) for second coding")
        print(f"wrote {RELIABILITY_SHEET} with the codes blanked.")
        print("have a second coder fill it without consulting the first, then rerun.")
        sys.exit(0)

    second = pd.read_csv(RELIABILITY_SHEET)
    second = second[second["orbit_category"].notna() & (second["orbit_category"] != "")]
    if second.empty:
        print(f"{RELIABILITY_SHEET} exists but is not coded yet.")
        sys.exit(0)
    report(sheet, second)


if __name__ == "__main__":
    main()
