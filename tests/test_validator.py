"""The validator has to reject bad sheets, not just accept good ones."""
import pandas as pd
import pytest

from src.validate_coding import check

PANEL = pd.DataFrame({
    "release_id": ["R1", "R1", "R2"],
    "slug": ["b1", "b2", "b1"],
})
FAMILIES = pd.DataFrame({
    "release_id": ["R1", "R2"],
    "family_id": ["F", "F"],
    "family_rank": [1, 2],
    "model_name": ["M1", "M2"],
})

BASE = {
    "release_id": "R2", "benchmark_slug": "b1", "orbit_category": "A",
    "reported": 1, "reported_value": 0.5, "source_tier": 1,
    "source_url": "https://openai.com/index/x", "group": "eligible",
    "variant_name": "", "prior_model_reported": "", "prior_source_url": "",
    "notes": "", "coder": "kz",
}


def sheet(**overrides):
    row = {**BASE, **overrides}
    return pd.DataFrame([row])


def errors_for(**overrides):
    errs, _ = check(sheet(**overrides), PANEL, FAMILIES)
    return " ".join(errs)


def test_clean_row_passes():
    assert errors_for() == ""


def test_illegal_category_rejected():
    assert "illegal orbit_category" in errors_for(orbit_category="Z")


def test_reported_flag_must_follow_category():
    assert "contradicts" in errors_for(orbit_category="A", reported=0)
    assert "contradicts" in errors_for(orbit_category="E", reported=1)


def test_category_e_requires_named_predecessor():
    out = errors_for(orbit_category="E", reported=0, reported_value="")
    assert "prior_model_reported" in out
    assert "prior_source_url" in out


def test_category_e_on_rank_one_release_rejected():
    """R1 is the first release in its family; it has nothing to drop from."""
    out = errors_for(
        release_id="R1", orbit_category="E", reported=0, reported_value="",
        prior_model_reported="M0", prior_source_url="https://x.com",
    )
    assert "rank-1" in out


def test_variant_required_for_c():
    assert "variant_name" in errors_for(orbit_category="C", reported=1)


def test_benign_reason_required_for_f():
    assert "notes" in errors_for(orbit_category="F", reported=0, reported_value="")


def test_aggregator_source_rejected():
    out = errors_for(source_url="https://artificialanalysis.ai/models/x")
    assert "aggregator" in out


def test_source_tier_bounded():
    assert "source_tier" in errors_for(source_tier=7)


def test_unresolvable_key_rejected():
    assert "does not resolve" in errors_for(release_id="Nope", benchmark_slug="b9")


def test_duplicate_rows_rejected():
    doubled = pd.concat([sheet(), sheet()], ignore_index=True)
    errs, _ = check(doubled, PANEL, FAMILIES)
    assert any("duplicate" in e for e in errs)


def test_uncoded_rows_warn_but_do_not_error():
    errs, warns = check(sheet(orbit_category="", reported=""), PANEL, FAMILIES)
    assert errs == []
    assert any("not yet coded" in w for w in warns)
