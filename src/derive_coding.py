
import sys

import numpy as np
import pandas as pd

from .config import ARTIFACTS, CODING_SHEET, INTERIM, RELEASE_COL, WORKLIST
from .families import load_families

CONTEMPORARY_WINDOW_DAYS = 45

def parse_reported(cell):
    out = {}
    if not isinstance(cell, str) or not cell.strip():
        return out
    for token in cell.split("|"):
        token = token.strip()
        if not token:
            continue
        reason = variant = value = None
        if "!" in token:
            token, reason = token.split("!", 1)
            out[token.strip()] = (None, None, reason.strip())
            continue
        if "=" in token:
            token, value = token.split("=", 1)
            value = value.strip()
        if "~" in token:
            token, variant = token.split("~", 1)
        out[token.strip()] = (value, variant, None)
    return out

def category_for(slug, reported, predecessor_reported, contemporaries):
    if slug in reported:
        value, variant, reason = reported[slug]
        if reason:
            return "F", None, None, reason
        if variant:
            return "C", value, variant, None
        if value in (None, ""):
            return "B", None, None, None
        return "A", value, None, None
    if slug in predecessor_reported:
        return "E", None, None, None
    if slug in contemporaries:
        return "G", None, None, None
    return "H", None, None, None

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

    rows = []
    for row in worklist[worklist["group"] == "eligible"].itertuples():
        release = row.release_id
        if release not in evidence:
            continue
        reported = evidence[release]

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

        category, value, variant, reason = category_for(
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
            "notes": reason or "",
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
