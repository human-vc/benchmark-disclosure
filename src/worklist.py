"""Emit the targeted disclosure-coding worklist.

Coding every eligible (release, benchmark) cell means reading a model card for
1,502 cells. Most of them can never produce the study's primary unit. A drop
(ORBIT E) requires a predecessor in the same family that reported the
benchmark, so only cells belonging to a multi-release family can ever be one.
Restricting to those, and to releases Epoch scores densely enough to support a
within-model comparison, cuts the reading to a few hundred cells without
discarding a single potential drop.

Placebo cells are emitted too, but pre-filled rather than hand-coded: a
benchmark that postdates the release could not have been reported in the
release-date artifact, so reported=0 holds by construction under the source
hierarchy in the protocol. They cost no reading time and they are what the
placebo falsification test consumes.
"""

import pandas as pd

from .config import FAMILIES, INTERIM, MIN_BENCHMARKS, RELEASE_COL, WORKLIST
from .families import load_families

CODING_COLUMNS = [
    "orbit_category", "reported", "reported_value", "source_tier", "source_url",
    "source_date", "variant_name", "prior_model_reported", "prior_source_url",
    "reverse_gap", "coder", "flagged_for_review", "notes",
]

CONTEXT_COLUMNS = [
    "group", "release_id", "benchmark_slug", "organization", "model_name",
    "release_date", "benchmark_name", "benchmark_release_date", "family_id",
    "family_rank", "prior_release", "prior_model_name", "n_benchmarks_scored",
    "independent_score",
]


def annotate(panel, families):
    panel = panel.merge(
        families[["release_id", "family_id", "family_rank"]],
        left_on=RELEASE_COL, right_on="release_id", how="left", validate="m:1",
    )
    sizes = families.groupby("family_id").size()
    panel["family_size"] = panel["family_id"].map(sizes)
    panel["n_benchmarks_scored"] = panel.groupby(RELEASE_COL)["slug"].transform("nunique")
    return panel


def nearest_predecessor(families):
    """Map each release to the previous release in its family."""
    ordered = families.sort_values(["family_id", "family_rank"])
    prior = ordered.groupby("family_id")[["release_id", "model_name"]].shift(1)
    return pd.DataFrame(
        {
            "release_id": ordered["release_id"].values,
            "prior_release": prior["release_id"].values,
            "prior_model_name": prior["model_name"].values,
        }
    )


def build(panel, families, min_benchmarks=MIN_BENCHMARKS):
    panel = annotate(panel, families)
    panel = panel.merge(nearest_predecessor(families), on="release_id", how="left")

    to_code = panel[
        panel["eligible"]
        & (panel["family_size"] >= 2)
        & (panel["n_benchmarks_scored"] >= min_benchmarks)
    ].copy()
    to_code["group"] = "eligible"
    for column in CODING_COLUMNS:
        to_code[column] = ""

    placebo = panel[panel["placebo"] & (panel["family_size"] >= 2)].copy()
    placebo["group"] = "placebo"
    for column in CODING_COLUMNS:
        placebo[column] = ""
    # Not hand-coded: a benchmark released after the model cannot appear in the
    # release-date artifact, so non-disclosure is mechanical, not a choice.
    placebo["reported"] = 0
    placebo["coder"] = "auto"
    placebo["notes"] = "benchmark postdates release; reported=0 by construction"

    out = pd.concat([to_code, placebo], ignore_index=True)
    out = out.rename(
        columns={
            "slug": "benchmark_slug",
            "Organization": "organization",
            "Model name": "model_name",
            "Release date": "release_date",
            "score": "independent_score",
        }
    )
    out = out.sort_values(
        ["group", "organization", "family_id", "family_rank", "benchmark_slug"]
    )
    return out[CONTEXT_COLUMNS + CODING_COLUMNS]


def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    families = load_families()
    out = build(panel, families)

    WORKLIST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WORKLIST, index=False)

    to_code = out[out["group"] == "eligible"]
    placebo = out[out["group"] == "placebo"]
    print(f"wrote {WORKLIST}\n")
    print(f"cells to hand-code : {len(to_code)}")
    print(f"  releases         : {to_code['release_id'].nunique()}")
    print(f"  families         : {to_code['family_id'].nunique()}")
    print(f"  median per release: {to_code.groupby('release_id').size().median():.0f}")
    print(f"placebo cells (auto): {len(placebo)}")

    print("\nreading order -- one provider at a time:")
    by_org = (
        to_code.groupby("organization")
        .agg(cells=("benchmark_slug", "size"), releases=("release_id", "nunique"))
        .sort_values("cells", ascending=False)
    )
    total = 0
    for org, row in by_org.iterrows():
        total += row["cells"]
        print(
            f"  {org:34} {row['cells']:4d} cells  {row['releases']:3d} releases"
            f"   cum {total / len(to_code):5.1%}"
        )

    no_prior = to_code["prior_release"].isna().sum()
    print(
        f"\nrank-1 cells (no predecessor, cannot be a drop but anchor one): {no_prior}"
    )


if __name__ == "__main__":
    main()
