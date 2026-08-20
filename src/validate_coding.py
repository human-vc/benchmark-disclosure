"""Validate a disclosure coding sheet against protocol/coding-protocol.md.

Coding errors are expensive to find downstream: a mis-keyed row silently drops
out of the merge in selectivity.py and shrinks the sample without saying so, and
a category E with no named predecessor is not evidence of anything. This checks
the rules the protocol commits to, at the point where they are cheap to fix.

Exits non-zero on any error so it can gate the estimator.
"""

import sys

import pandas as pd

from .config import CODING_SHEET, INTERIM, RELEASE_COL
from .families import load_families

CATEGORIES = set("ABCDEFGHI")
REPORTED = {"A", "B", "C"}
HIGH_SUSPICION = {"D", "E", "G"}
MAX_SOURCE_TIER = 4

BANNED_SOURCE_HOSTS = (
    "paperswithcode", "huggingface.co/spaces", "artificialanalysis",
    "llm-stats", "lmarena", "wikipedia",
)


def check(sheet, panel, families):
    """Return (errors, warnings) as lists of human-readable strings."""
    errors, warnings = [], []

    def fail(mask, message):
        bad = sheet[mask]
        if len(bad):
            sample = bad.head(3)[["release_id", "benchmark_slug"]].to_dict("records")
            errors.append(f"{message}: {len(bad)} row(s), e.g. {sample}")

    missing = [c for c in ("release_id", "benchmark_slug", "orbit_category")
               if c not in sheet.columns]
    if missing:
        return [f"missing required columns: {missing}"], warnings

    coded = sheet[sheet["orbit_category"].notna() & (sheet["orbit_category"] != "")]

    # --- keys resolve, and are unique -------------------------------------
    dupes = sheet.duplicated(["release_id", "benchmark_slug"], keep=False)
    fail(dupes, "duplicate (release_id, benchmark_slug)")

    known_cells = set(zip(panel[RELEASE_COL], panel["slug"]))
    unresolved = ~sheet.apply(
        lambda r: (r["release_id"], r["benchmark_slug"]) in known_cells, axis=1
    )
    fail(unresolved, "row does not resolve to a (release, benchmark) cell in the panel")

    # --- categories -------------------------------------------------------
    bad_cat = coded[~coded["orbit_category"].isin(CATEGORIES)]
    if len(bad_cat):
        errors.append(
            f"illegal orbit_category: {sorted(set(bad_cat['orbit_category']))}"
        )

    # --- reported flag must follow from the category ----------------------
    if "reported" in sheet.columns:
        expected = coded["orbit_category"].isin(REPORTED).astype(int)
        actual = pd.to_numeric(coded["reported"], errors="coerce")
        mismatch = actual.notna() & (actual != expected)
        if mismatch.any():
            errors.append(
                f"reported flag contradicts orbit_category on {int(mismatch.sum())} row(s); "
                "A/B/C are reported, D-I are not"
            )

    # --- per-category required fields (protocol "Decision rules") ---------
    def requires(category, column, why):
        if column not in coded.columns:
            errors.append(f"missing column {column!r}, required for category {category}")
            return
        rows = coded[coded["orbit_category"] == category]
        blank = rows[rows[column].isna() | (rows[column].astype(str).str.strip() == "")]
        if len(blank):
            errors.append(
                f"category {category} requires {column!r} ({why}): "
                f"{len(blank)} row(s) blank"
            )

    requires("E", "prior_model_reported", "E must name the earlier model")
    requires("E", "prior_source_url", "E must cite where it was previously reported")
    requires("C", "variant_name", "C must record the variant actually reported")
    requires("F", "notes", "F must quote the stated benign reason")
    requires("A", "reported_value", "A is a numeric score and must carry it")

    # --- E must actually have a predecessor in the family ------------------
    if "family_id" in families.columns:
        rank = families.set_index("release_id")["family_rank"]
        e_rows = coded[coded["orbit_category"] == "E"]
        orphan = e_rows[e_rows["release_id"].map(rank).fillna(1) <= 1]
        if len(orphan):
            errors.append(
                f"category E on a rank-1 release, which has no predecessor to drop "
                f"from: {len(orphan)} row(s)"
            )

    # --- sources ----------------------------------------------------------
    if "source_tier" in coded.columns:
        tier = pd.to_numeric(coded["source_tier"], errors="coerce")
        bad = coded[tier.notna() & ((tier < 1) | (tier > MAX_SOURCE_TIER))]
        if len(bad):
            errors.append(
                f"source_tier outside 1-{MAX_SOURCE_TIER} on {len(bad)} row(s); "
                "the protocol permits only the four official artifact types"
            )
        blank_tier = coded[tier.isna() & (coded.get("coder", "") != "auto")]
        if len(blank_tier):
            warnings.append(f"{len(blank_tier)} coded row(s) carry no source_tier")

    if "source_url" in coded.columns:
        url = coded["source_url"].fillna("").astype(str).str.lower()
        banned = coded[url.str.contains("|".join(BANNED_SOURCE_HOSTS), regex=True)]
        if len(banned):
            errors.append(
                f"third-party aggregator used as source on {len(banned)} row(s); "
                "the protocol forbids coding from anything but official artifacts"
            )
        need_url = coded[coded["orbit_category"].isin(REPORTED) & (url == "")]
        if len(need_url):
            errors.append(
                f"reported category with no source_url: {len(need_url)} row(s)"
            )

    # --- completeness -----------------------------------------------------
    uncoded = sheet[
        (sheet.get("group", "eligible") == "eligible")
        & (sheet["orbit_category"].isna() | (sheet["orbit_category"] == ""))
    ]
    if len(uncoded):
        warnings.append(
            f"{len(uncoded)} of {(sheet.get('group', 'eligible') == 'eligible').sum()} "
            "eligible cells not yet coded"
        )

    return errors, warnings


def main():
    if not CODING_SHEET.exists():
        print(f"no coding sheet at {CODING_SHEET}")
        print("run `python -m src.worklist` and copy data/worklist.csv there.")
        sys.exit(1)

    sheet = pd.read_csv(CODING_SHEET)
    panel = pd.read_csv(INTERIM / "panel.csv")
    families = load_families()

    errors, warnings = check(sheet, panel, families)

    coded = sheet["orbit_category"].notna() & (sheet["orbit_category"] != "")
    print(f"rows: {len(sheet)}   coded: {coded.sum()}")
    if coded.any():
        counts = sheet.loc[coded, "orbit_category"].value_counts().sort_index()
        print("categories: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        high = sheet.loc[coded, "orbit_category"].isin(HIGH_SUSPICION).mean()
        print(f"high-suspicion share (D/E/G): {high:.1%}")

    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")

    if errors:
        print(f"\n{len(errors)} error(s). Sheet is not analysis-ready.")
        sys.exit(1)
    print("\nvalidation passed")


if __name__ == "__main__":
    main()
