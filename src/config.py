from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"

EPOCH_ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"

INDEX_FILE = "epoch_capabilities_index.csv"

# Hand-collected, checked in: benchmark release dates Epoch does not publish.
BENCHMARK_DATES = ROOT / "data" / "benchmark_dates.csv"

# Hand-reviewed, checked in: which releases belong to the same product line.
FAMILIES = ROOT / "data" / "families.csv"

# Generated: the targeted set of cells a human actually has to read.
WORKLIST = ROOT / "data" / "worklist.csv"

# The hand-coded disclosure sheet: the outcome variable.
# Extracted evidence: what each release's official artifact actually reports.
# One row per release, each traceable to the URL it was read from.
ARTIFACTS = ROOT / "data" / "artifacts.csv"

CODING_SHEET = ROOT / "data" / "disclosures.csv"

# The blinded 20% second coding, for Cohen's kappa.
RELIABILITY_SHEET = ROOT / "data" / "disclosures_second_coder.csv"
BENCHMARK_META = "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"

MIN_BENCHMARKS = 8

# Half-width of the contemporaneity window for within-benchmark percentiles,
# in days. design.md ranks a release against models of its own era; this is the
# operational definition of "its own era" and is reported as a sensitivity.
WINDOW_DAYS = 182

SCORE_COL = "Best score (across scorers)"
MODEL_COL = "Model version"

# The disclosure unit is a *release*, not a scaffold variant: a provider ships
# one model card per release, not one per reasoning-effort setting. Epoch's
# "Model version" splits a release across scaffolds (_high, _max, _32K), so the
# panel is keyed on (Organization, Model name, Release date) instead.
RELEASE_COL = "release_id"
