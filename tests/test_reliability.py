"""Kappa must behave: 1 on perfect agreement, ~0 on chance, and it must
penalise agreement that is only prevalence."""
import numpy as np
import pandas as pd

from src.reliability import cohens_kappa, draw_sample


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


def test_draw_sample_covers_every_provider():
    sheet = pd.DataFrame({
        "organization": ["OpenAI"] * 30 + ["Anthropic"] * 10 + ["Meta AI"] * 3,
        "orbit_category": ["A"] * 43,
        "coder": ["kz"] * 43,
        "release_id": [f"r{i}" for i in range(43)],
        "benchmark_slug": ["b"] * 43,
    })
    sample = draw_sample(sheet, share=0.2, seed=0)
    assert set(sample["organization"]) == {"OpenAI", "Anthropic", "Meta AI"}
    assert 0.15 < len(sample) / len(sheet) < 0.35


def test_draw_sample_skips_autofilled_placebo_rows():
    sheet = pd.DataFrame({
        "organization": ["OpenAI"] * 10,
        "orbit_category": [""] * 10,
        "coder": ["auto"] * 10,
        "release_id": [f"r{i}" for i in range(10)],
        "benchmark_slug": ["b"] * 10,
    })
    assert draw_sample(sheet).empty
