"""Fetch and unpack Epoch AI's benchmark data (CC-BY).

Epoch's hub is a moving target. Downloading is therefore two decisions, not
one: fetch the current bundle, and decide whether that bundle becomes the build
this repository reports numbers against. The second is deliberate and requires
--capture, which rewrites data/snapshot.json.
"""

import argparse
import hashlib
import io
import zipfile
from datetime import date

import requests

from .config import EPOCH_ZIP_URL, RAW
from .snapshot import capture, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="pin this download as the reported build, rewriting data/snapshot.json",
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get(EPOCH_ZIP_URL, timeout=120)
    response.raise_for_status()

    payload = response.content
    zip_sha256 = hashlib.sha256(payload).hexdigest()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(RAW)

    csvs = sorted(RAW.rglob("*.csv"))
    print(f"extracted {len(csvs)} csv files to {RAW}")
    print(f"zip sha256: {zip_sha256}")

    if args.capture:
        manifest = capture(zip_sha256=zip_sha256, captured=date.today().isoformat())
        print(
            f"pinned snapshot {manifest['captured']}: "
            f"{manifest['csv_files']} files, {manifest['index_rows']} model-versions"
        )
        print("every number in the repo should now be regenerated against this build")
    else:
        print(stamp())


if __name__ == "__main__":
    main()
