"""Group releases into model families so that drops (ORBIT E) are computable.

The primary analytic unit in docs/design.md is a *drop*: a provider reported
benchmark X for model v1, then omitted X for v2 in the same family while an
independent score for v2 exists. Nothing else in the pipeline knows that
"Claude Opus 4.5" and "Claude Opus 4.6" are successive releases of one product
line, so without this module the drop design cannot be run at all.

Family assignment is a judgment call, so it is seeded by heuristic and then
hand-corrected. The seed is deliberately conservative and the output carries a
status column: rows marked "auto" have not been reviewed by a human and should
not be trusted for analysis. Regenerating preserves anything marked
"confirmed" or "corrected".
"""

import re

import pandas as pd

from .config import FAMILIES, INTERIM, RELEASE_COL

COLUMNS = ["family_id", "release_id", "organization", "model_name",
           "release_date", "family_rank", "status"]

# Tokens that identify a product tier and belong in the family name, as
# distinct from tokens that identify a version and do not.
SIZE = re.compile(r"^\d+(\.\d+)?\s*[bmBM]$")
VERSION = re.compile(r"^v?\d+(\.\d+)*[a-z]?$")


def derive_family(organization, model_name):
    """Strip version markers from a release name, keep tier and size markers.

    "Claude Opus 4.5"          -> Anthropic / Claude Opus
    "Gemini 2.5 Pro (Jun 2025)"-> Google DeepMind / Gemini Pro
    "GPT-5.4 Mini"             -> OpenAI / GPT Mini
    "Llama 3.1 405B"           -> Meta AI / Llama 405B
    """
    # Parentheses carry two different things: a disambiguating date, which is
    # version information and must go, and a parameter count, which is product
    # identity and must stay. "Qwen2.5-Coder (7B)" and "(32B)" are different
    # products; "Gemini 2.5 Pro (Jun 2025)" and "(May 2025)" are one line.
    sizes = []
    for chunk in re.findall(r"\(([^)]*)\)", model_name):
        sizes += re.findall(r"\d+(?:\.\d+)?\s*[BbMm]\b", chunk)
    name = re.sub(r"\([^)]*\)", " ", model_name)
    if sizes:
        name = f"{name} {sizes[0].upper().replace(' ', '')}"
    name = name.replace("-", " ").replace("_", " ")
    tokens = []
    for token in name.split():
        if SIZE.match(token):
            tokens.append(token.upper())
            continue
        if VERSION.match(token):
            continue
        stripped = re.sub(r"\d+(\.\d+)*", "", token).strip()
        if stripped:
            tokens.append(stripped)
    family = " ".join(tokens).strip() or model_name
    return f"{organization} / {family}"


def build_families(panel):
    if "primary_org" not in panel.columns:
        panel = panel.copy()
        panel["primary_org"] = panel["Organization"]
    releases = (
        panel.groupby(RELEASE_COL, as_index=False)
        .agg(
            organization=("primary_org", "first"),
            model_name=("Model name", "first"),
            release_date=("Release date", "first"),
        )
        .sort_values(["organization", "model_name", "release_date"])
    )
    releases["family_id"] = [
        derive_family(row.organization, row.model_name) for row in releases.itertuples()
    ]
    releases = releases.sort_values(["family_id", "release_date", "model_name"])
    releases["family_rank"] = releases.groupby("family_id").cumcount() + 1
    releases["status"] = "auto"
    return releases[COLUMNS]


def load_families(path=FAMILIES):
    """Load and validate. Raises rather than silently returning a bad linkage."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m src.families` to seed it."
        )
    fam = pd.read_csv(path, parse_dates=["release_date"])
    if fam["release_id"].duplicated().any():
        dupes = fam.loc[fam["release_id"].duplicated(), "release_id"].tolist()
        raise ValueError(f"a release belongs to more than one family: {dupes}")
    ranks = fam.groupby("family_id")["family_rank"]
    if not ranks.apply(lambda s: sorted(s) == list(range(1, len(s) + 1))).all():
        raise ValueError("family_rank must be dense and 1-based within each family")
    return fam


def predecessors(fam, release_id):
    """Every earlier release in the same family, newest first.

    Newest first because the nearest predecessor is the strongest evidence for
    a drop: the closer the two releases, the weaker the "we stopped running it
    for unrelated reasons" story.
    """
    row = fam.loc[fam["release_id"] == release_id]
    if row.empty:
        return fam.iloc[0:0]
    row = row.iloc[0]
    earlier = fam[
        (fam["family_id"] == row["family_id"])
        & (fam["family_rank"] < row["family_rank"])
    ]
    return earlier.sort_values("family_rank", ascending=False)


def main():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    seeded = build_families(panel)

    if FAMILIES.exists():
        existing = pd.read_csv(FAMILIES, parse_dates=["release_date"])
        keep = existing[existing["status"].isin({"confirmed", "corrected"})]
        seeded = pd.concat(
            [keep, seeded[~seeded["release_id"].isin(keep["release_id"])]],
            ignore_index=True,
        )
        seeded = seeded.sort_values(["family_id", "release_date"])
        seeded["family_rank"] = seeded.groupby("family_id").cumcount() + 1

    FAMILIES.parent.mkdir(parents=True, exist_ok=True)
    seeded.to_csv(FAMILIES, index=False)

    sizes = seeded.groupby("family_id").size().sort_values(ascending=False)
    print(f"releases: {len(seeded)}   families: {len(sizes)}")
    print(f"families with >=2 releases (drops possible): {(sizes >= 2).sum()}")
    print(f"releases inside a multi-release family: {sizes[sizes >= 2].sum()}")
    print(f"singleton families (no drop possible): {(sizes == 1).sum()}")
    print(f"\nwrote {FAMILIES}")
    print(f"\nstatus: {seeded['status'].value_counts().to_dict()}")
    print("\nlargest families, review these first:")
    for family, n in sizes.head(25).items():
        members = seeded[seeded["family_id"] == family].sort_values("family_rank")
        chain = " -> ".join(members["model_name"])
        print(f"  [{n}] {family}\n        {chain}")


if __name__ == "__main__":
    main()
