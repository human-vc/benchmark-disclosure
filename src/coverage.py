
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import INTERIM, MIN_BENCHMARKS, RELEASE_COL

def load_panel():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    return panel

def coverage_by_model(panel):
    return (
        panel.groupby(RELEASE_COL)
        .agg(
            n_benchmarks=("slug", "nunique"),
            n_eligible=("eligible", "sum"),
            organization=("Organization", "first"),
            accessibility=("Model accessibility", "first"),
            release_date=("Release date", "first"),
            compute=("Training compute (FLOP)", "first"),
        )
        .reset_index()
    )

def access_group(value):
    if not isinstance(value, str):
        return None
    if value.startswith("Open weights"):
        return "open"
    if value == "API access":
        return "api"
    return None

def main():
    panel = load_panel()
    models = coverage_by_model(panel)
    models["access"] = models["accessibility"].map(access_group)

    print(f"releases with >=1 independent score: {len(models)}")
    bins = [0, 1, 5, 7, 9, 1000]
    labels = ["1", "2-5", "6-7", "8-9", "10+"]
    counts = pd.cut(models["n_benchmarks"], bins=bins, labels=labels).value_counts()
    print("\nbenchmarks per release:")
    for label in labels:
        share = counts.get(label, 0) / len(models)
        print(f"  {label:>5}: {counts.get(label, 0):4d}  ({share:.1%})")

    print("\ncoverage by access type:")
    for name, group in models.dropna(subset=["access"]).groupby("access"):
        print(
            f"  {name:>4}: n={len(group):4d}  "
            f"mean={group['n_benchmarks'].mean():.2f}  "
            f"median={group['n_benchmarks'].median():.0f}"
        )

    api = models.loc[models["access"] == "api", "n_benchmarks"]
    openw = models.loc[models["access"] == "open", "n_benchmarks"]
    stat, p = mannwhitneyu(api, openw, alternative="two-sided")
    print(f"\nMann-Whitney api vs open: U={stat:.0f}, p={p:.2e}")
    if p < 0.05:
        print("  coverage differs by access type: condition on n_benchmarks")

    sample = models[
        (models["n_benchmarks"] >= MIN_BENCHMARKS)
        & models["organization"].notna()
        & models["release_date"].notna()
    ]
    print(f"\nanalysis sample (>={MIN_BENCHMARKS} benchmarks): {len(sample)}")
    print(sample["access"].value_counts(dropna=False).to_string())

    for threshold in (5, 6, 8, 10):
        n = (
            (models["n_benchmarks"] >= threshold)
            & models["organization"].notna()
            & models["release_date"].notna()
        ).sum()
        print(f"  >={threshold:2d}: {n}")

if __name__ == "__main__":
    main()
