from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"

EPOCH_ZIP_URL = "https://epoch.ai/data/benchmark_data.zip"

INDEX_FILE = "epoch_capabilities_index.csv"
BENCHMARK_META = "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"

MIN_BENCHMARKS = 8

SCORE_COL = "Best score (across scorers)"
MODEL_COL = "Model version"
