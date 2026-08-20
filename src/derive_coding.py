"""Derive ORBIT codes mechanically from extracted artifact evidence.

The coding protocol asks two very different kinds of question, and conflating
them is how a coding sheet becomes unfalsifiable.

  1. What does this release's official artifact actually report? A factual
     question, answerable only by reading the artifact, recorded in
     data/artifacts.csv with the URL it came from.

  2. Given that, which ORBIT category applies? A *rule*, not a judgment:
     - reported here                                   -> A (or B/C by form)
     - not reported, but a predecessor reported it     -> E, the drop
     - not reported, no predecessor, contemporaries did-> G
     - not reported, and nobody comparable did either  -> H

Only (1) requires reading. Deriving (2) in code means every category is
reproducible from the evidence file, an analyst can re-run the rules after
changing them, and no category can be assigned by vibes.

Releases absent from artifacts.csv are left uncoded rather than assumed
undisclosed. A missing artifact is missing evidence, not evidence of omission.
"""

import sys

import numpy as np
import pandas as pd

from .config import ARTIFACTS, CODING_SHEET, INTERIM, RELEASE_COL, WORKLIST
from .families import load_families

CONTEMPORARY_WINDOW_DAYS = 45


def parse_reported(cell):
    """'gpqa_diamond=0.83|mmlu~MMLU-Pro=0.87|hle' -> dict of slug -> (value, variant).

    Three forms, matching the protocol's A/B/C split:
      slug=value        numeric score reported          -> A
      slug              named but no number given       -> B
      slug~Variant=val  a variant reported instead      -> C
    """
    out = {}
    if not isinstance(cell, str) or not cell.strip():
        return out
    for token in cell.split("|"):
        token = token.strip()
        if not token:
            continue
        variant = None
        value = None
        if "=" in token:
            token, value = token.split("=", 1)
            value = value.strip()
        if "~" in token:
            token, variant = token.split("~", 1)
        out[token.strip()] = (value, variant)
    return out


def category_for(slug, reported, predecessor_reported, contemporaries):
    if slug in reported:
        value, variant = reported[slug]
        if variant:
            return "C", value, variant
        if value in (None, ""):
            return "B", None, None
        return "A", value, None
    if slug in predecessor_reported:
        return "E", None, None
    if slug in contemporaries:
        return "G", None, None
    return "H", None, None


def build(worklist, artifacts, families, panel):
    artifacts = artifacts.copy()
    artifacts["parsed"] = artifacts["reported_slugs"].map(parse_reported)
    evidence = dict(zip(artifacts["release_id"], artifacts["parsed"]))
    meta = artifacts.set_index("release_id")

    dates = (
        panel.groupby(RELEASE_COL)["Release date"].first().to_dict()
    )
    orgs = panel.groupby(RELEASE_COL)["primary_org"].first().to_dict()

    fam = families.set_index("release_id")

    # Eligible rows only. Placebo cells are pre-filled reported=0 by
    # construction and must not be assigned an ORBIT category: coding a
    # postdating benchmark as "not mentioned" would double-count it and
    # inflate the low-suspicion categories.
    rows = []
    for row in worklist[worklist["group"] == "eligible"].itertuples():
        release = row.release_id
        if release not in evidence:
            continue
        reported = evidence[release]

        # every earlier release in this family that we have evidence for
        predecessor_reported = {}
        prior_model = prior_url = None
        if release in fam.index:
            family_id = fam.loc[release, "family_id"]
            rank = fam.loc[release, "family_rank"]
            earlier = fam[(fam["family_id"] == family_id) & (fam["family_rank"] < rank)]
            for prior in earlier.sort_values("family_rank", ascending=False).index:
                for slug in evidence.get(prior, {}):
                    if slug not in predecessor_reported:
                        predecessor_reported[slug] = prior

        # what comparable providers reported within the window
        contemporaries = set()
        this_date = dates.get(release)
        this_org = orgs.get(release)
        if this_date is not None:
            for other, other_reported in evidence.items():
                if orgs.get(other) == this_org:
                    continue
                other_date = dates.get(other)
                if other_date is None:
                    continue
                if abs((other_date - this_date).days) <= CONTEMPORARY_WINDOW_DAYS:
                    contemporaries |= set(other_reported)

        category, value, variant = category_for(
            row.benchmark_slug, reported, predecessor_reported, contemporaries
        )
        prior_release = predecessor_reported.get(row.benchmark_slug)
        rows.append({
            "release_id": release,
            "benchmark_slug": row.benchmark_slug,
            "group": row.group,
            "orbit_category": category,
            "reported": int(category in {"A", "B", "C"}),
            "reported_value": value or "",
            "source_tier": meta.loc[release, "source_tier"],
            "source_url": meta.loc[release, "source_url"],
            "source_date": meta.loc[release, "source_date"],
            "variant_name": variant or "",
            "prior_model_reported": (
                fam.loc[prior_release, "model_name"] if prior_release else ""
            ),
            "prior_source_url": (
                meta.loc[prior_release, "source_url"] if prior_release else ""
            ),
            "reverse_gap": "",
            "coder": meta.loc[release, "coder"],
            "flagged_for_review": meta.loc[release, "flagged_for_review"],
            "notes": "",
            "organization": row.organization,
            "model_name": row.model_name,
            "release_date": row.release_date,
            "family_id": row.family_id,
            "family_rank": row.family_rank,
            "prior_release": prior_release or "",
            "prior_model_name": row.prior_model_name,
            "n_benchmarks_scored": row.n_benchmarks_scored,
            "independent_score": row.independent_score,
            "benchmark_name": row.benchmark_name,
            "benchmark_release_date": row.benchmark_release_date,
        })
    return pd.DataFrame(rows)


def reverse_gaps(artifacts, panel):
    """Benchmarks an artifact reports that Epoch does not score for that release.

    design.md keeps these deliberately: under a pure concealment story the set
    a provider reaches for beyond the independent source should not be
    systematically favourable.
    """
    scored = panel.groupby(RELEASE_COL)["slug"].apply(set).to_dict()
    rows = []
    for row in artifacts.itertuples():
        reported = set(parse_reported(row.reported_slugs))
        extra = reported - scored.get(row.release_id, set())
        if extra:
            rows.append({"release_id": row.release_id,
                         "reverse_gap": "|".join(sorted(extra))})
    return pd.DataFrame(rows)


def main():
    if not ARTIFACTS.exists():
        print(f"no artifact evidence at {ARTIFACTS}")
        sys.exit(1)

    worklist = pd.read_csv(WORKLIST)
    artifacts = pd.read_csv(ARTIFACTS)
    artifacts = artifacts[artifacts["fetch_status"] == "ok"]
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    families = load_families()

    sheet = build(worklist, artifacts, families, panel)

    gaps = reverse_gaps(artifacts, panel)
    if len(gaps):
        sheet = sheet.merge(gaps, on="release_id", how="left", suffixes=("_x", ""))
        sheet["reverse_gap"] = sheet["reverse_gap"].fillna("")
        sheet = sheet.drop(columns=[c for c in sheet.columns if c.endswith("_x")])

    # placebo rows carry through pre-filled; they are not hand-coded
    placebo = worklist[worklist["group"] == "placebo"].copy()
    placebo["orbit_category"] = ""
    placebo["reported"] = 0
    placebo["coder"] = "auto"
    placebo["notes"] = "benchmark postdates release; reported=0 by construction"
    sheet = pd.concat([sheet, placebo], ignore_index=True)

    CODING_SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(CODING_SHEET, index=False)

    coded = sheet[sheet["orbit_category"].notna() & (sheet["orbit_category"] != "")]
    print(f"artifacts with evidence: {len(artifacts)} releases")
    print(f"derived coded cells: {len(coded)}")
    if len(coded):
        counts = coded["orbit_category"].value_counts().sort_index()
        print("categories: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"drops (E): {int((coded['orbit_category'] == 'E').sum())}")
    print(f"wrote {CODING_SHEET}")


if __name__ == "__main__":
    main()
