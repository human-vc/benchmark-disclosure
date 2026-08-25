"""Kappa must behave: 1 on perfect agreement, ~0 on chance, and it must
penalise agreement that is only prevalence."""
import numpy as np
import pandas as pd

from src.reliability import blank_sheet, cohens_kappa, draw_releases


def test_perfect_agreement():
    codes = list("AABBCDEFG")
    kappa, agreement = cohens_kappa(codes, codes)
    assert agreement == 1.0
    assert np.isclose(kappa, 1.0)


def test_chance_agreement_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.choice(list("ABCDE"), size=4000)
    b = rng.choice(list("ABCDE"), size=4000)
    kappa, _ = cohens_kappa(a, b)
    assert abs(kappa) < 0.05


def test_kappa_below_raw_agreement_when_one_category_dominates():
    """The reason the protocol asks for kappa and not percent agreement:
    two coders who both say A almost always agree by construction."""
    a = ["A"] * 95 + list("BCDEF")
    b = ["A"] * 95 + list("BCDEG")
    kappa, agreement = cohens_kappa(a, b)
    assert agreement > 0.95
    assert kappa < agreement


def test_total_disagreement_is_negative():
    kappa, agreement = cohens_kappa(list("AAAABBBB"), list("BBBBAAAA"))
    assert agreement == 0.0
    assert kappa < 0


class TestSecondExtraction:
    """The second coding is a second *extraction*. Re-categorising cells would
    measure nothing: the categories are derived by deterministic rule, so two
    people given the same evidence file must agree by construction."""

    SAMPLE = pd.DataFrame([
        dict(release_id="R1", organization="O", model_name="M", n_cells=5,
             release_date="2025-01-01", family_rank=1, source_tier="1",
             source_url="https://example.invalid/card.pdf", extra_source_urls="",
             source_date="2025-01-01", artifact_kind="model_card",
             reported_slugs="gpqa_diamond=0.9|hle", coder="kevin",
             flagged_for_review="1", notes="read from the page-5 raster",
             fetch_status="ok"),
    ])

    def test_blank_sheet_keeps_the_artifact(self):
        """Blinding the URL would measure the second coder's search, not their
        reading; both must open the same documents."""
        from src.reliability import blank_sheet

        sheet = blank_sheet(self.SAMPLE)
        assert sheet.loc[0, "source_url"] == "https://example.invalid/card.pdf"
        assert "extra_source_urls" in sheet.columns

    def test_blank_sheet_hides_every_judgment(self):
        from src.reliability import blank_sheet

        sheet = blank_sheet(self.SAMPLE)
        for column in ("reported_slugs", "notes", "flagged_for_review", "coder"):
            assert sheet.loc[0, column] == "", column

    def test_blank_sheet_hides_the_flag(self):
        """A flag says 'the first coder was unsure here', which is the nudge
        that inflates agreement."""
        from src.reliability import blank_sheet

        assert blank_sheet(self.SAMPLE).loc[0, "flagged_for_review"] == ""

    def test_draw_is_stratified_by_provider(self):
        from src.reliability import draw_releases

        many = pd.DataFrame([
            dict(release_id=f"{org}{i}", organization=org, release_date="2025-01-01",
                 fetch_status="ok")
            for org in ("A", "B") for i in range(10)
        ])
        drawn = draw_releases(many, share=0.2, seed=0)
        assert set(drawn["organization"]) == {"A", "B"}

    def test_blocked_releases_are_not_drawn(self):
        from src.reliability import draw_releases

        rows = pd.DataFrame([
            dict(release_id="ok1", organization="A", release_date="2025-01-01",
                 fetch_status="ok"),
            dict(release_id="blk", organization="A", release_date="2025-01-02",
                 fetch_status="blocked"),
        ])
        assert list(draw_releases(rows, share=1.0)["release_id"]) == ["ok1"]
