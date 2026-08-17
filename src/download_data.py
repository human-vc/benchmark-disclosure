"""Fetch and unpack Epoch AI's benchmark data (CC-BY)."""

import io
import zipfile

import requests

from .config import EPOCH_ZIP_URL, RAW


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get(EPOCH_ZIP_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(RAW)

    csvs = sorted(RAW.rglob("*.csv"))
    print(f"extracted {len(csvs)} csv files to {RAW}")


if __name__ == "__main__":
    main()
