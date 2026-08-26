"""Order the artifact reading so the drop estimator turns on first.

The coding unit is the release artifact, not the cell: one document yields one
`reported_slugs` string and `derive_coding` expands it into every cell of that
release. So the queue is 108 documents, and the only real decision is what
order to read them in.

Reading by date leaves the drop estimator at n=0 until nearly the end, because
an ORBIT E needs the predecessor's artifact as well as the successor's. This
orders whole families together, predecessor before successor, and puts the
families that can produce drops first, weighted by how many shared benchmarks
they put at risk.
"""

import pandas as pd

from .config import ARTIFACTS, QUEUE, WORKLIST

ORDER_COLUMNS = [
    "read_order", "drop_capable", "pair_member", "shared_with_prior",
    "prior_release",
]

COLUMNS = [
    "read_order", "release_id", "organization", "model_name", "release_date",
    "family_id", "family_rank", "prior_release", "drop_capable",
    "pair_member", "shared_with_prior", "n_cells", "source_tier",
    "source_url", "notes",
]


def eligible_sets(worklist):
    """Benchmark slugs each release must be coded on."""
    eligible = worklist[worklist["group"] == "eligible"]
    return eligible.groupby("release_id")["benchmark_slug"].apply(set)


def pair_overlap(worklist, readable):
    """Shared eligible benchmarks between each release and its predecessor.

    A pair counts only when both artifacts are readable, since an E cannot be
    assigned from one side of it.
    """
    sets = eligible_sets(worklist)
    priors = (worklist[worklist["group"] == "eligible"]
              .drop_duplicates("release_id")
              .set_index("release_id")["prior_release"])
    overlap = {}
    for release, prior in priors.items():
        if not isinstance(prior, str) or not prior.strip():
            continue
        if release not in readable or prior not in readable:
            continue
        shared = sets.get(release, set()) & sets.get(prior, set())
        if shared:
            overlap[release] = (prior, len(shared))
    return overlap


def build(worklist, artifacts):
    readable = set(artifacts["release_id"])
    overlap = pair_overlap(worklist, readable)

    meta = (worklist[worklist["group"] == "eligible"]
            .drop_duplicates("release_id")
            .set_index("release_id"))
    cells = (worklist[worklist["group"] == "eligible"]
             .groupby("release_id").size())

    rows = []
    for release in sorted(readable):
        prior, shared = overlap.get(release, ("", 0))
        rows.append({
            "release_id": release,
            "organization": meta["organization"].get(release, ""),
            "model_name": meta["model_name"].get(release, ""),
            "release_date": meta["release_date"].get(release, ""),
            "family_id": meta["family_id"].get(release, ""),
            "family_rank": meta["family_rank"].get(release, ""),
            "prior_release": prior,
            "drop_capable": int(release in overlap),
            "shared_with_prior": shared,
            "n_cells": int(cells.get(release, 0)),
        })
    queue = pd.DataFrame(rows)
    members = set(overlap) | {prior for prior, _ in overlap.values()}
    queue["pair_member"] = queue["release_id"].isin(members).astype(int)

    at_risk = queue.groupby("family_id")["shared_with_prior"].sum()
    queue["family_at_risk"] = queue["family_id"].map(at_risk).fillna(0)
    queue["rank_order"] = pd.to_numeric(queue["family_rank"], errors="coerce")
    queue = queue.sort_values(
        ["family_at_risk", "family_id", "rank_order", "release_date", "release_id"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    queue["read_order"] = queue.index + 1

    for column in ("source_tier", "source_url", "notes"):
        queue[column] = ""
    return queue[COLUMNS]


def apply_to_artifacts(queue, artifacts):
    """Put the reading order into the sheet itself.

    The coder fills `artifacts.csv`, so the order belongs there rather than in a
    second file they have to cross-reference 108 times. Existing columns and
    values are preserved; only row order and the ordering columns change.
    """
    order = queue.set_index("release_id")
    merged = artifacts.copy()
    for column in ORDER_COLUMNS:
        merged[column] = merged["release_id"].map(order[column])
    merged = merged.sort_values("read_order").reset_index(drop=True)
    front = ["read_order", "release_id", "organization", "model_name", "release_date"]
    rest = [c for c in merged.columns if c not in front]
    return merged[front + rest]


def main():
    worklist = pd.read_csv(WORKLIST, dtype=str).fillna("")
    artifacts = pd.read_csv(ARTIFACTS, dtype=str).fillna("")
    queue = build(worklist, artifacts)
    queue.to_csv(QUEUE, index=False)
    apply_to_artifacts(queue, artifacts).to_csv(ARTIFACTS, index=False)

    capable = queue[queue["drop_capable"] == 1]
    members = queue[queue["pair_member"] == 1]
    print(f"wrote {QUEUE}  ({len(queue)} artifacts)")
    print(f"  drop-capable successors  : {len(capable)}")
    print(f"  artifacts in those pairs : {len(members)}")
    print(f"  shared benchmarks at risk: {int(capable['shared_with_prior'].sum())}")
    print(f"  every pair is covered by row {int(members['read_order'].max())}, "
          f"families kept contiguous")
    print("\n  by organisation, drop-capable first:")
    for org, n in capable["organization"].value_counts().head(8).items():
        print(f"    {org:24} {n}")


if __name__ == "__main__":
    main()
