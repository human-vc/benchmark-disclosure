import pandas as pd
import pytest

from src.config import RELEASE_COL
from src.derive_coding import build, category_for, parse_reported

def test_parse_handles_the_four_reporting_forms():
    got = parse_reported("gpqa_diamond=0.94|hle|cybench~pass@30=1.0|bbh!leaked")
    assert got["gpqa_diamond"] == ("0.94", None, None)
    assert got["hle"] == (None, None, None)
    assert got["cybench"] == ("1.0", "pass@30", None)
    assert got["bbh"] == (None, None, "leaked")

def test_category_rules():
    assert category_for("x", {"x": ("0.5", None, None)}, {}, set())[0] == "A"
    assert category_for("x", {"x": (None, None, None)}, {}, set())[0] == "B"
    assert category_for("x", {"x": ("0.5", "Pro", None)}, {}, set())[0] == "C"
    assert category_for("x", {"x": (None, None, "leaked")}, {}, set())[0] == "F"
    assert category_for("x", {}, {"x": "prior"}, set())[0] == "E"
    assert category_for("x", {}, {}, {"x"})[0] == "G"
    assert category_for("x", {}, {}, set())[0] == "H"

def test_predecessor_report_makes_a_drop():
    worklist = pd.DataFrame([
        dict(release_id="R2", benchmark_slug="b1", group="eligible",
             organization="O", model_name="M2", release_date="2025-02-01",
             family_id="F", family_rank=2, prior_model_name="M1",
             n_benchmarks_scored=5, independent_score=0.5,
             benchmark_name="B1", benchmark_release_date="2024-01-01"),
    ])
    artifacts = pd.DataFrame([
        dict(release_id="R1", reported_slugs="b1=0.4", source_tier=1,
             source_url="u1", source_date="2025-01-01", coder="c",
             flagged_for_review=""),
        dict(release_id="R2", reported_slugs="b2=0.9", source_tier=1,
             source_url="u2", source_date="2025-02-01", coder="c",
             flagged_for_review=""),
    ])
    families = pd.DataFrame([
        dict(release_id="R1", family_id="F", family_rank=1, model_name="M1"),
        dict(release_id="R2", family_id="F", family_rank=2, model_name="M2"),
    ])
    panel = pd.DataFrame([
        dict(**{RELEASE_COL: "R1"}, slug="b1", primary_org="O",
             **{"Release date": pd.Timestamp("2025-01-01")}),
        dict(**{RELEASE_COL: "R2"}, slug="b1", primary_org="O",
             **{"Release date": pd.Timestamp("2025-02-01")}),
    ])
    out = build(worklist, artifacts, families, panel)
    row = out.iloc[0]
    assert row["orbit_category"] == "E"
    assert row["prior_model_reported"] == "M1"
    assert row["prior_source_url"] == "u1"
    assert row["reported"] == 0

def test_placebo_rows_never_get_a_category():
    worklist = pd.DataFrame([
        dict(release_id="R1", benchmark_slug="b1", group="placebo",
             organization="O", model_name="M1", release_date="2025-01-01",
             family_id="F", family_rank=1, prior_model_name="",
             n_benchmarks_scored=5, independent_score=0.5,
             benchmark_name="B1", benchmark_release_date="2026-01-01"),
    ])
    artifacts = pd.DataFrame([
        dict(release_id="R1", reported_slugs="b9=0.4", source_tier=1,
             source_url="u", source_date="2025-01-01", coder="c",
             flagged_for_review=""),
    ])
    families = pd.DataFrame([
        dict(release_id="R1", family_id="F", family_rank=1, model_name="M1")])
    panel = pd.DataFrame([
        dict(**{RELEASE_COL: "R1"}, slug="b1", primary_org="O",
             **{"Release date": pd.Timestamp("2025-01-01")})])
    assert build(worklist, artifacts, families, panel).empty

def test_releases_without_evidence_are_left_uncoded():
    worklist = pd.DataFrame([
        dict(release_id="R_unknown", benchmark_slug="b1", group="eligible",
             organization="O", model_name="M", release_date="2025-01-01",
             family_id="F", family_rank=1, prior_model_name="",
             n_benchmarks_scored=5, independent_score=0.5,
             benchmark_name="B1", benchmark_release_date="2024-01-01"),
    ])
    artifacts = pd.DataFrame(columns=["release_id","reported_slugs","source_tier",
                                      "source_url","source_date","coder",
                                      "flagged_for_review"])
    families = pd.DataFrame(columns=["release_id","family_id","family_rank","model_name"])
    panel = pd.DataFrame(columns=[RELEASE_COL,"slug","primary_org","Release date"])
    assert build(worklist, artifacts, families, panel).empty

class TestCategoryF:

    def test_reason_form_parses(self):
        from src.derive_coding import parse_reported

        parsed = parse_reported("mmlu=86.4|bbh!contaminated by training data")
        assert parsed["bbh"] == (None, None, "contaminated by training data")
        assert parsed["mmlu"] == ("86.4", None, None)

    def test_reason_derives_F_not_E(self):
        from src.derive_coding import category_for

        parsed = {"bbh": (None, None, "contaminated by training data")}
        category, _, _, reason = category_for(
            "bbh", parsed, {"bbh": "prior release"}, {"bbh"}
        )
        assert category == "F"
        assert reason == "contaminated by training data"

    def test_F_is_not_counted_as_reported(self):
        from src.derive_coding import category_for

        category, *_ = category_for("bbh", {"bbh": (None, None, "why")}, {}, set())
        assert category not in {"A", "B", "C"}
