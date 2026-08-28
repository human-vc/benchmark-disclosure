"""How far the record can carry the possession question.

Selective withholding is a statement about results a provider held and did
not print. The disclosure label observes what was printed; possession is
observed only where a document says a run happened. Four cuts organise what
that leaves. Category D counts the direct cases, an announced evaluation with
no number. Stratifying the omission deficit by evidence tier asks whether the
gap strengthens as possession evidence strengthens, which withholding predicts
and composition does not. The change-based drop test replaces the level of a
dropped benchmark with its change against the same benchmark under the
family's previous release, differencing out persistent salience and relevance.
And the swap sample isolates same-sized reporting tables where one benchmark
replaced another, the case a fixed-table-size explanation cannot cover.
"""

import numpy as np
import pandas as pd

from .config import ARTIFACTS, RELEASE_COL

DRAWS = 9999


def predecessors(families):
    """release_id -> immediately preceding release_id within its family."""
    fam = families.copy()
    fam["family_rank"] = fam["family_rank"].astype(float)
    fam = fam.sort_values(["family_id", "family_rank"])
    prior = {}
    for _, group in fam.groupby("family_id"):
        ids = group["release_id"].tolist()
        for earlier, later in zip(ids, ids[1:]):
            prior[later] = earlier
    return prior


def transition_frame(merged, families):
    """Cells reported by the predecessor and scored under both releases.

    A cell is `dropped` when the successor derives category E, `retained`
    when the successor reports it again. Cells the predecessor never reported
    are outside the frame: the change-based test conditions on demonstrated
    relevance. Score changes are kept only where both scores sit on the unit
    interval, since a difference across unlike metrics is not a quantity.
    """
    prior = predecessors(families)
    keyed = merged.set_index([RELEASE_COL, "slug"])
    pct = keyed["percentile"]
    score = keyed["score"]
    category = keyed["orbit_category"]
    reported = keyed["reported"]
    rows = []
    for release in merged[RELEASE_COL].unique():
        previous = prior.get(release)
        if previous is None:
            continue
        for slug in merged.loc[merged[RELEASE_COL] == release, "slug"]:
            now, before = (release, slug), (previous, slug)
            if before not in pct.index or now not in pct.index:
                continue
            if not reported.get(before, 0) == 1:
                continue
            label = category.get(now)
            if label == "E":
                status = "dropped"
            elif reported.get(now, 0) == 1:
                status = "retained"
            else:
                continue
            s0, s1 = score[before], score[now]
            unit = 0 <= min(s0, s1) and max(s0, s1) <= 1.01
            rows.append({
                "transition": f"{previous}|{release}",
                "slug": slug, "status": status,
                "d_percentile": pct[now] - pct[before],
                "d_score": 100 * (s1 - s0) if unit else np.nan,
            })
    return pd.DataFrame(rows)


def change_gap(frame, column="d_percentile", draws=DRAWS, seed=0):
    """Dropped-minus-retained mean change, permuting within transitions.

    The permutation keeps each transition's drop count fixed, so the null is
    which of the previously reported benchmarks were dropped, not how many.
    Withholding predicts a negative gap.
    """
    sub = frame.dropna(subset=[column])
    dropped = sub.loc[sub["status"] == "dropped", column]
    retained = sub.loc[sub["status"] == "retained", column]
    observed = dropped.mean() - retained.mean()
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        shuffled = sub.copy()
        shuffled["perm"] = shuffled.groupby("transition")["status"].transform(
            lambda s: rng.permutation(s.values))
        draw = (shuffled.loc[shuffled["perm"] == "dropped", column].mean()
                - shuffled.loc[shuffled["perm"] == "retained", column].mean())
        if abs(draw) >= abs(observed):
            exceed += 1
    return {
        "gap": round(float(observed), 2),
        "p": round((exceed + 1) / (draws + 1), 4),
        "n_cells": int(len(sub)),
        "n_dropped": int((sub["status"] == "dropped").sum()),
        "n_retained": int((sub["status"] == "retained").sum()),
        "n_transitions": int(sub["transition"].nunique()),
    }


def tier_deficits(merged, tiers=("E", "G", "H")):
    """The omission deficit computed one evidence tier at a time.

    E carries the strongest possession evidence, G availability only, H
    neither. A deficit that deepens from H to E is what withholding predicts;
    a flat or inverted gradient is what composition predicts.
    """
    out = {}
    for tier in tiers:
        frame = merged.copy()
        frame["set"] = np.where(
            frame["group"] == "placebo", "placebo",
            np.where(frame["orbit_category"] == tier, "omitted", "other"))
        wide = (frame[frame["set"].isin(["placebo", "omitted"])]
                .pivot_table(index=RELEASE_COL, columns="set",
                             values="percentile", aggfunc="mean").dropna())
        gap = wide["omitted"] - wide["placebo"]
        out[tier] = {"gap": round(float(gap.mean()), 2),
                     "n_releases": int(len(wide))}
    return out


def direct_evidence(coding):
    """Category D against every cell where a run is documented."""
    coded = coding[(coding["orbit_category"].notna())
                   & (coding["orbit_category"] != "")
                   & (coding.get("coder") != "auto")]
    known_run = coded["orbit_category"].isin(list("ABCD")).sum()
    d_cells = int((coded["orbit_category"] == "D").sum())
    return {"d_cells": d_cells, "known_run_cells": int(known_run)}


def swap_cases(merged, families, artifacts=None):
    """Equal-sized reporting tables where a dropped benchmark was replaced.

    For each such substitution the inserted benchmark's standing is compared
    with the removed one's under the new release; withholding-by-substitution
    predicts the insert stands higher.
    """
    from .derive_coding import parse_reported
    prior = predecessors(families)
    if artifacts is None:
        artifacts = pd.read_csv(ARTIFACTS, dtype=str, keep_default_na=False)
    artifacts = artifacts.set_index("release_id")
    keyed = merged.set_index([RELEASE_COL, "slug"])
    pct = keyed["percentile"]
    category = keyed["orbit_category"]

    def table(release):
        parsed = parse_reported(artifacts.loc[release, "reported_slugs"])
        # F-form entries record an absence, so they are not table rows;
        # off-panel rows, prefixed "+", still occupy space in the table
        printed = {k for k, (_, _, reason) in parsed.items() if reason is None}
        return printed, {k for k in printed if not k.startswith("+")}

    cases = []
    for release in merged[RELEASE_COL].unique():
        previous = prior.get(release)
        if previous is None or release not in artifacts.index \
                or previous not in artifacts.index:
            continue
        (now_all, now), (before_all, before) = table(release), table(previous)
        if len(now_all) != len(before_all):
            continue
        removed = [s for s in before - now
                   if category.get((release, s)) == "E" and (release, s) in pct.index]
        inserted = [s for s in now - before if (release, s) in pct.index]
        for out_slug in removed:
            for in_slug in inserted:
                cases.append({
                    "transition": f"{previous}|{release}",
                    "removed": out_slug, "inserted": in_slug,
                    "removed_pct": round(float(pct[(release, out_slug)]), 1),
                    "inserted_pct": round(float(pct[(release, in_slug)]), 1),
                })
    return cases
