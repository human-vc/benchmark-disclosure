from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"

EPOCH_ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"

INDEX_FILE = "epoch_capabilities_index.csv"

SNAPSHOT = ROOT / "data" / "snapshot.json"

BENCHMARK_DATES = ROOT / "data" / "benchmark_dates.csv"

FAMILIES = ROOT / "data" / "families.csv"

WORKLIST = ROOT / "data" / "worklist.csv"

ARTIFACTS = ROOT / "data" / "artifacts.csv"

CODING_SHEET = ROOT / "data" / "disclosures.csv"

RELIABILITY_SHEET = ROOT / "data" / "disclosures_second_coder.csv"
BENCHMARK_META = "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"

MIN_BENCHMARKS = 8

WINDOW_DAYS = 182

SCORE_COL = "Best score (across scorers)"
MODEL_COL = "Model version"

RELEASE_COL = "release_id"
