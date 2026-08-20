"""Assemble the model x benchmark availability matrix and apply the temporal gate.

Output is long-format: one row per (model, benchmark) pair for which an
independent score exists, annotated with whether that benchmark was eligible
for disclosure at the model's release date.
"""

import re

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_DATES,
    BENCHMARK_META,
    INDEX_FILE,
    INTERIM,
    MODEL_COL,
    RAW,
    RELEASE_COL,
    SCORE_COL,
)

SLUG_OVERRIDES = {
    "ANLI": "adversarial_nli",
    "ARC AI2": "arc_ai2",
    "MATH level 5": "math_level_5",
    "SWE-Bench verified": "swe_bench_verified",
    "GPQA diamond": "gpqa_diamond",
    "OTIS Mock AIME 2024-2025": "otis_mock_aime",
}


# Maps the slug derived from Epoch's metadata name onto the slug used by the
# score file, which is the filename. The two sides normalise differently --
# metadata says "HellaSwag", the file is hella_swag_external.csv -- and an
# unmapped pair is silently dropped, so this table is load-bearing. Anything
# missing from it costs the panel a whole benchmark.
SLUG_ALIASES = {
    "terminal_bench": "terminalbench",
    "fiction_livebench": "fictionlivebench",
    "gso_bench": "gso",
    "deepresearch_bench": "deepresearchbench",
    "otis_mock_aime": "otis_mock_aime_2024_2025",
    # OSWorld and OSWorld 2.0 are different benchmarks with different score
    # files (os_world, 9 models; osworld_2, 8 models). Mapping both onto
    # osworld_2, as this table previously did, orphaned the OSWorld file and
    # attributed OSWorld's 2024-04-11 date to OSWorld 2.0 -- which would admit
    # models to a choice set containing a benchmark that did not yet exist.
    "osworld": "os_world",
    "osworld_2_0": "osworld_2",
    # Epoch's metadata drops the word boundary; the score files keep it. These
    # eight were previously documented as "no score file in the public bundle"
    # when in fact the files ship and the join was failing on normalisation.
    "hellaswag": "hella_swag",
    "winogrande": "wino_grande",
    "triviaqa": "trivia_qa",
    "scienceqa": "science_qa",
    "openbookqa": "open_book_qa",
    "cadeval": "cad_eval",
    "remote_labor_index": "rli",
    # FrontierMath ships several private variants in metadata against one
    # score file per tier. Collapsed in collapse_meta().
    "frontiermath_2025_02_28_private": "frontiermath",
    "frontiermath_tiers_1_3_v2_private": "frontiermath",
    "frontiermath_tier_4_2025_07_01_private": "frontiermath_tier_4",
    "frontiermath_tier_4_v2_private": "frontiermath_tier_4",
}

# Metadata benchmarks with genuinely no score file in the public bundle.
# An unjoined slug outside this set means the alias table has gone stale
# against a new Epoch release, and is an error rather than a warning.
KNOWN_UNJOINED = {"ebr_bench"}


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


def collapse_meta(meta):
    """Reduce the metadata to one row per slug.

    Epoch lists several private variants of the same benchmark as separate
    metadata rows (FrontierMath tiers, OSWorld and OSWorld 2.0), while the
    score side ships a single file per canonical slug. SLUG_ALIASES maps the
    variants onto that slug, which leaves duplicate slugs on the metadata side
    and makes the downstream merge fan out rather than join. Collapse here so
    the merge can be asserted many-to-one.

    Date rule: take the *latest* known release date among the collapsed names.
    In every observed case the score file is named after the newest variant
    (osworld_2, frontiermath_tier_4), so the latest date is the better
    attribution, and it is the conservative direction for the temporal gate --
    it admits fewer models to the choice set rather than more. Attributing the
    older variant's date to a newer benchmark would manufacture omissions of a
    benchmark that did not yet exist.

    Collapsed slugs are flagged so the hand-filled date file can override them.
    """
    grouped = meta.groupby("slug", as_index=False).agg(
        benchmark_name=("benchmark_name", "first"),
        benchmark_release_date=("benchmark_release_date", "max"),
        collapsed_from=("benchmark_name", lambda s: " | ".join(sorted(s))),
        n_collapsed=("benchmark_name", "size"),
    )
    return grouped


def apply_date_overrides(meta, path=BENCHMARK_DATES):
    """Merge the hand-collected benchmark release dates over Epoch's.

    Epoch's metadata carries a release date for a minority of benchmarks and
    omits ~20 benchmarks it nonetheless scores. Those pairs are neither
    eligible nor placebo, so they leave the analysis entirely -- filling them
    is the single highest-leverage manual task in the repository.

    Precedence: Epoch's date wins where it exists, because it is the published
    source. A hand row with status="override" beats it, which is the escape
    hatch for cases where Epoch's row is known wrong -- notably slugs that
    collapse several metadata variants onto one score file. Rows with no date
    are worklist stubs and are ignored.
    """
    if not path.exists():
        meta["date_source"] = np.where(
            meta["benchmark_release_date"].notna(), "epoch", "none"
        )
        return meta

    hand = pd.read_csv(path)
    hand["benchmark_release_date"] = pd.to_datetime(
        hand["benchmark_release_date"], errors="coerce"
    )
    hand = hand.dropna(subset=["benchmark_release_date"])
    if hand["slug"].duplicated().any():
        raise ValueError(f"duplicate slugs in {path}")

    # Outer, not left: 20 of the benchmarks needing a hand date are score files
    # with no metadata row at all, so a left join would silently drop exactly
    # the rows this file exists to supply.
    merged = meta.merge(
        hand[["slug", "benchmark_release_date", "status"]],
        on="slug",
        how="outer",
        suffixes=("", "_hand"),
        validate="1:1",
    )
    introduced = merged["benchmark_name"].isna()
    merged.loc[introduced, "benchmark_name"] = merged.loc[introduced, "slug"]
    merged.loc[introduced, "collapsed_from"] = merged.loc[introduced, "slug"]
    merged.loc[introduced, "n_collapsed"] = 1

    epoch_date = merged["benchmark_release_date"]
    hand_date = merged["benchmark_release_date_hand"]
    override = merged["status"].eq("override") & hand_date.notna()
    fills_gap = epoch_date.isna() & hand_date.notna()
    use_hand = override | fills_gap

    merged["benchmark_release_date"] = epoch_date.where(~use_hand, hand_date)
    merged["date_source"] = np.select(
        [use_hand, merged["benchmark_release_date"].notna()],
        ["hand", "epoch"],
        default="none",
    )
    return merged.drop(columns=["benchmark_release_date_hand", "status"])


def load_benchmark_meta(root):
    meta = pd.read_csv(root / BENCHMARK_META)
    meta["benchmark_release_date"] = pd.to_datetime(
        meta["benchmark_release_date"], errors="coerce"
    )
    meta["slug"] = meta["benchmark_name"].map(slugify)
    meta = collapse_meta(meta)
    if meta["slug"].duplicated().any():
        raise ValueError("benchmark metadata still has duplicate slugs after collapse")
    meta = apply_date_overrides(meta)
    return meta[
        [
            "benchmark_name",
            "slug",
            "benchmark_release_date",
            "collapsed_from",
            "n_collapsed",
            "date_source",
        ]
    ]


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


# Epoch labels the same provider inconsistently across releases: five Gemini
# models sit under "Google" while the other 29 sit under "Google DeepMind".
# The entity making the disclosure decision is the same either way, and leaving
# them split breaks the Gemini Flash and Flash-Lite families in half.
ORG_ALIASES = {"Google": "Google DeepMind"}

MERGE_WINDOW_DAYS = 45


def canonical_release_date(panel):
    """Collapse same-model entries that sit within MERGE_WINDOW_DAYS.

    Epoch's index sometimes carries one shipped model twice with dates a few
    days apart -- Llama 4 Maverick one day, GLM-4.5 two, Qwen2.5-Coder one.
    Those are not two disclosure events; there is one model card. Left split
    they inflate family ranks and can manufacture a "drop" between a release
    and itself.

    The threshold has to exist because the same-name-different-release case is
    real and must survive: GPT-4o covers five snapshots 57 to 106 days apart,
    each with its own release post. 45 days separates the two populations
    cleanly in this build -- the duplicate gaps run 1 to 38 days and the
    genuine snapshot gaps start at 55. It is a judgment call on a bimodal gap
    distribution, not a principled constant, and it is printed at build time so
    it can be checked rather than trusted.
    """
    dates = panel["Release date"]
    keys = panel["primary_org"] + "|" + panel["Model name"]
    canonical = dates.copy()
    for _, index in panel.groupby(keys).groups.items():
        group = dates.loc[index].sort_values()
        anchor = None
        for position, value in group.items():
            if anchor is None or (value - anchor).days > MERGE_WINDOW_DAYS:
                anchor = value
            canonical.loc[position] = anchor
    return canonical


def collapse_to_releases(panel):
    """Collapse scaffold variants to one row per (release, benchmark).

    Epoch's "Model version" separates reasoning-effort and context-window
    variants of the same shipped model: GPT-5.5 appears six times, Claude 3.7
    Sonnet ten. A provider publishes one model card per release, so leaving
    these split would enter the same release into a benchmark's percentile
    distribution up to ten times and would make "one row per model release"
    -- the coding unit in the protocol -- ill-defined.

    Model name alone is not the key: "GPT-4o" spans five snapshots across ten
    months, each with its own release post and its own disclosure decision.
    The key is (Organization, Model name, Release date), which collapses
    scaffolds while preserving those snapshots.

    Score rule: maximum across scaffolds, which is the rule design.md already
    states for scaffolds within a benchmark file, applied consistently. The
    spread is retained so the choice can be audited rather than assumed.
    """
    panel = panel.dropna(subset=["Organization", "Model name", "Release date"]).copy()
    panel["primary_org"] = (
        panel["Organization"].str.split(",").str[0].str.strip().replace(ORG_ALIASES)
    )
    panel["Release date"] = canonical_release_date(panel)
    panel[RELEASE_COL] = (
        panel["primary_org"]
        + " | "
        + panel["Model name"]
        + " | "
        + panel["Release date"].dt.strftime("%Y-%m-%d")
    )

    keys = [RELEASE_COL, "slug"]
    collapsed = panel.groupby(keys, as_index=False).agg(
        score=("score", "max"),
        scaffold_spread=("score", lambda s: s.max() - s.min()),
        n_scaffolds=("score", "size"),
        model_versions=(MODEL_COL, lambda s: " | ".join(sorted(s.astype(str)))),
        **{
            col: (col, "first")
            for col in [
                "Model name",
                "Release date",
                "Organization",
                "primary_org",
                "Model accessibility",
                "Training compute (FLOP)",
                "benchmark_name",
                "benchmark_release_date",
                "date_source",
                "collapsed_from",
                "n_collapsed",
            ]
        },
    )
    return collapsed


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)

    index = load_index(RAW)
    meta = load_benchmark_meta(RAW)
    scores = load_scores(RAW)

    matched = set(scores["slug"]) & set(meta["slug"])
    print(f"benchmark files: {scores['slug'].nunique()}")
    print(f"metadata slugs: {len(meta)}, joined on slug: {len(matched)}")
    print(f"unjoined metadata slugs: {sorted(set(meta['slug']) - matched)}")
    undated = sorted(set(scores["slug"]) - matched)
    print(f"score slugs with no metadata ({len(undated)}): {undated}")

    unjoined = set(meta["slug"]) - matched
    if unjoined - KNOWN_UNJOINED:
        raise ValueError(
            "metadata slugs with no score file, and not in KNOWN_UNJOINED: "
            f"{sorted(unjoined - KNOWN_UNJOINED)}. The alias table is stale."
        )

    collapsed = meta[meta["n_collapsed"] > 1]
    if len(collapsed):
        print(f"\ncollapsed metadata slugs ({len(collapsed)}), date = latest known:")
        for row in collapsed.itertuples():
            date = row.benchmark_release_date
            shown = "unknown" if pd.isna(date) else date.date()
            print(f"  {row.slug}: {row.collapsed_from}  -> {shown}")

    panel = scores.merge(index, on=MODEL_COL, how="left", validate="m:1").merge(
        meta, on="slug", how="left", validate="m:1"
    )
    if panel.duplicated([MODEL_COL, "slug"]).any():
        raise ValueError("panel has duplicate (model, benchmark) rows")

    panel.to_csv(INTERIM / "panel_versions.csv", index=False)
    print(f"\nversion-level rows: {len(panel)} (audit copy)")

    before = panel.dropna(subset=["Organization", "Model name", "Release date"])
    before_count = (
        before["Organization"].str.split(",").str[0].str.strip()
        + before["Model name"]
        + before["Release date"].astype(str)
    ).nunique()
    panel = collapse_to_releases(panel)
    merged_away = before_count - panel[RELEASE_COL].nunique()
    print(f"near-duplicate releases merged (<={MERGE_WINDOW_DAYS}d apart): {merged_away}")

    panel["benchmark_predates_model"] = (
        panel["benchmark_release_date"] < panel["Release date"]
    )
    panel["date_known"] = panel["benchmark_release_date"].notna()

    panel["eligible"] = panel["benchmark_predates_model"] & panel["date_known"]
    panel["placebo"] = ~panel["benchmark_predates_model"] & panel["date_known"]

    known = panel["date_known"].mean()
    print(f"releases: {panel[RELEASE_COL].nunique()}")
    print(f"pairs: {len(panel)}")
    print(f"benchmark release date known for {known:.1%} of pairs")
    print(f"eligible (benchmark predates model): {panel['eligible'].sum()}")
    print(f"placebo (benchmark postdates model): {panel['placebo'].sum()}")
    print(f"unknown vintage: {(~panel['date_known']).sum()}")

    multi = panel["n_scaffolds"] > 1
    print(f"\ncells collapsed from >1 scaffold: {multi.sum()} ({multi.mean():.1%})")
    disagree = panel["scaffold_spread"] > 0
    print(f"  of those, scaffolds disagreed on score: {disagree.sum()}")

    panel.to_csv(INTERIM / "panel.csv", index=False)
    print(f"\nwrote {INTERIM / 'panel.csv'}")


if __name__ == "__main__":
    main()
