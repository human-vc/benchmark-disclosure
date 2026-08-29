
import numpy as np
import pandas as pd
import pytest

from src.config import RELEASE_COL
from src import possession

def families(pairs):
    rows = []
    for fam, releases in pairs.items():
        for rank, rid in enumerate(releases, start=1):
            rows.append({"family_id": fam, "release_id": rid,
                         "family_rank": rank})
    return pd.DataFrame(rows)

def merged_frame(rows):
    return pd.DataFrame(rows, columns=[
        RELEASE_COL, "slug", "percentile", "score", "orbit_category",
        "reported", "group"])

@pytest.fixture
def two_step():
    rows = []
    for slug, before, after, category, reported in [
        ("kept", 50.0, 55.0, "A", 1),
        ("gone", 60.0, 20.0, "E", 0),
        ("new", np.nan, 70.0, "A", 1),
    ]:
        if not np.isnan(before):
            rows.append(("R1", slug, before, before / 100, "A", 1, "eligible"))
        rows.append(("R2", slug, after, after / 100, category, reported, "eligible"))
    return merged_frame(rows), families({"F": ["R1", "R2"]})

def test_predecessors_links_adjacent_ranks():
    prior = possession.predecessors(families({"F": ["a", "b", "c"], "G": ["x"]}))
    assert prior == {"b": "a", "c": "b"}

def test_transition_frame_conditions_on_prior_report(two_step):
    merged, fam = two_step
    frame = possession.transition_frame(merged, fam)
    assert set(frame["slug"]) == {"kept", "gone"}
    assert frame.set_index("slug").loc["gone", "status"] == "dropped"
    assert frame.set_index("slug").loc["kept", "status"] == "retained"

def test_transition_frame_blanks_incommensurable_scores(two_step):
    merged, fam = two_step
    merged.loc[merged["slug"] == "kept", "score"] = 1500.0
    frame = possession.transition_frame(merged, fam).set_index("slug")
    assert np.isnan(frame.loc["kept", "d_score"])
    assert np.isfinite(frame.loc["kept", "d_percentile"])

def planted(gap, n_transitions=20, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, fams = [], {}
    for t in range(n_transitions):
        r0, r1 = f"A{t}", f"B{t}"
        fams[f"F{t}"] = [r0, r1]
        for j in range(per):
            slug = f"b{t}_{j}"
            dropped = j == 0
            change = gap if dropped else 0.0
            before = 50.0
            after = before + change + rng.normal(0, 2)
            rows.append((r0, slug, before, before / 100, "A", 1, "eligible"))
            rows.append((r1, slug, after, np.clip(after, 0, 100) / 100,
                         "E" if dropped else "A", 0 if dropped else 1,
                         "eligible"))
    return merged_frame(rows), families(fams)

def test_change_gap_finds_planted_withholding():
    merged, fam = planted(gap=-25.0)
    frame = possession.transition_frame(merged, fam)
    result = possession.change_gap(frame, draws=499)
    assert result["gap"] < -15
    assert result["p"] < 0.05

def test_change_gap_is_quiet_under_the_null():
    merged, fam = planted(gap=0.0)
    frame = possession.transition_frame(merged, fam)
    result = possession.change_gap(frame, draws=499)
    assert result["p"] > 0.10

def test_change_gap_counts_add_up():
    merged, fam = planted(gap=-10.0)
    frame = possession.transition_frame(merged, fam)
    result = possession.change_gap(frame, draws=99)
    assert result["n_cells"] == result["n_dropped"] + result["n_retained"]
    assert result["n_transitions"] == 20

def test_direct_evidence_counts_the_known_run_cells():
    coding = pd.DataFrame({
        "orbit_category": ["A", "B", "C", "D", "E", "G", "H", ""],
        "coder": ["jc"] * 7 + ["auto"],
    })
    counts = possession.direct_evidence(coding)
    assert counts == {"d_cells": 1, "known_run_cells": 4}

def test_tier_deficits_reads_the_planted_gradient():
    rows = []
    for r in range(12):
        rid = f"R{r}"
        rows.append((rid, "p", 40.0, 0.4, "", 0, "placebo"))
        rows.append((rid, "e", 60.0, 0.6, "E", 0, "eligible"))
        rows.append((rid, "h", 45.0, 0.45, "H", 0, "eligible"))
    merged = merged_frame(rows)
    tiers = possession.tier_deficits(merged, tiers=("E", "H"))
    assert tiers["E"]["gap"] == pytest.approx(20.0)
    assert tiers["H"]["gap"] == pytest.approx(5.0)
    assert tiers["E"]["n_releases"] == 12

def test_swap_cases_needs_equal_tables_and_a_derived_drop(two_step):
    merged, fam = two_step
    artifacts = pd.DataFrame({
        "release_id": ["R1", "R2"],
        "reported_slugs": ["kept=0.5|gone=0.6", "kept=0.55|new=0.7"],
    })
    cases = possession.swap_cases(merged, fam, artifacts=artifacts)
    assert len(cases) == 1
    case = cases[0]
    assert case["removed"] == "gone" and case["inserted"] == "new"
    unequal = artifacts.copy()
    unequal.loc[1, "reported_slugs"] = "kept=0.55|new=0.7|+extra=0.9"
    assert possession.swap_cases(merged, fam, artifacts=unequal) == []
