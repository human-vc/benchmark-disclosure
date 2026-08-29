
import numpy as np
import pandas as pd

from .config import ARTIFACTS, RELEASE_COL

DRAWS = 9999

def predecessors(families):
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
    coded = coding[(coding["orbit_category"].notna())
                   & (coding["orbit_category"] != "")
                   & (coding.get("coder") != "auto")]
    known_run = coded["orbit_category"].isin(list("ABCD")).sum()
    d_cells = int((coded["orbit_category"] == "D").sum())
    return {"d_cells": d_cells, "known_run_cells": int(known_run)}

def swap_cases(merged, families, artifacts=None):
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
