"""Assemble the model x benchmark availability matrix and apply the temporal gate.

Output is long-format: one row per (model, benchmark) pair for which an
independent score exists, annotated with whether that benchmark was eligible
for disclosure at the model's release date.
"""

import re

import pandas as pd

from .config import BENCHMARK_META, INDEX_FILE, INTERIM, MODEL_COL, RAW, SCORE_COL

SLUG_OVERRIDES = {
    "ANLI": "adversarial_nli",
    "ARC AI2": "arc_ai2",
    "MATH level 5": "math_level_5",
    "SWE-Bench verified": "swe_bench_verified",
    "GPQA diamond": "gpqa_diamond",
    "OTIS Mock AIME 2024-2025": "otis_mock_aime",
}


SLUG_ALIASES = {
    "terminal_bench": "terminalbench",
    "osworld": "osworld_2",
    "osworld_2_0": "osworld_2",
    "fiction_livebench": "fictionlivebench",
    "gso_bench": "gso",
    "deepresearch_bench": "deepresearchbench",
    "otis_mock_aime": "otis_mock_aime_2024_2025",
    "frontiermath_2025_02_28_private": "frontiermath",
    "frontiermath_tiers_1_3_v2_private": "frontiermath",
    "frontiermath_tier_4_2025_07_01_private": "frontiermath_tier_4",
    "frontiermath_tier_4_v2_private": "frontiermath_tier_4",
}


def slugify(name):
    slug = SLUG_OVERRIDES.get(name)
    if slug is None:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return SLUG_ALIASES.get(slug, slug)


def load_index(root):
    index = pd.read_csv(root / INDEX_FILE)
    index["Release date"] = pd.to_datetime(index["Release date"], errors="coerce")
    return index[
        [
            MODEL_COL,
            "Model name",
            "Release date",
            "Organization",
            "Model accessibility",
            "Training compute (FLOP)",
        ]
    ]


def load_benchmark_meta(root):
    meta = pd.read_csv(root / BENCHMARK_META)
    meta["benchmark_release_date"] = pd.to_datetime(
        meta["benchmark_release_date"], errors="coerce"
    )
    meta["slug"] = meta["benchmark_name"].map(slugify)
    return meta[["benchmark_name", "slug", "benchmark_release_date"]]


def score_column(frame):
    """Epoch names the score column per benchmark. Take the first numeric
    column after the model identifier, preferring the explicit best-score
    column when the file provides one."""
    if SCORE_COL in frame.columns:
        return SCORE_COL
    for column in frame.columns[1:]:
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return None


def load_scores(root):
    frames = []
    for path in sorted(root.rglob("*.csv")):
        if path.name.startswith("_") or "index" in path.name:
            continue
        if "eci_benchmark" in path.name:
            continue
        frame = pd.read_csv(path)
        if frame.empty or frame.columns[0] != MODEL_COL:
            continue
        column = score_column(frame)
        if column is None:
            continue
        slug = path.stem.removesuffix("_external")
        tidy = pd.DataFrame(
            {
                MODEL_COL: frame[MODEL_COL],
                "score": pd.to_numeric(frame[column], errors="coerce"),
                "slug": slug,
            }
        ).dropna(subset=["score"])
        frames.append(tidy.groupby([MODEL_COL, "slug"], as_index=False)["score"].max())
    return pd.concat(frames, ignore_index=True)


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    index = load_index(RAW)
    meta = load_benchmark_meta(RAW)
    scores = load_scores(RAW)

    matched = set(scores["slug"]) & set(meta["slug"])
    print(f"benchmark files: {scores['slug'].nunique()}")
    print(f"metadata rows: {len(meta)}, joined on slug: {len(matched)}")
    print(f"unjoined metadata names: {sorted(set(meta['slug']) - matched)}")

    panel = scores.merge(index, on=MODEL_COL, how="left").merge(
        meta, on="slug", how="left"
    )

    panel["benchmark_predates_model"] = (
        panel["benchmark_release_date"] < panel["Release date"]
    )
    panel["date_known"] = panel["benchmark_release_date"].notna()

    panel["eligible"] = panel["benchmark_predates_model"] & panel["date_known"]
    panel["placebo"] = ~panel["benchmark_predates_model"] & panel["date_known"]

    known = panel["date_known"].mean()
    print(f"\npairs: {len(panel)}")
    print(f"benchmark release date known for {known:.1%} of pairs")
    print(f"eligible (benchmark predates model): {panel['eligible'].sum()}")
    print(f"placebo (benchmark postdates model): {panel['placebo'].sum()}")
    print(f"unknown vintage: {(~panel['date_known']).sum()}")

    panel.to_csv(INTERIM / "panel.csv", index=False)
    print(f"\nwrote {INTERIM / 'panel.csv'}")


if __name__ == "__main__":
    main()
