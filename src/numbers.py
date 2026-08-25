"""Write every quantity the write-up reports into one machine-readable file.

A paper that quotes forty numbers from a pipeline has forty chances to quote a
stale one, and prose is where staleness hides: nothing errors when a figure in
a sentence no longer matches the code that produced it. So every reported
quantity is emitted here, once, by one command, into `data/processed/numbers.json`,
and the write-up cites that file rather than a remembered value.

The file carries the snapshot digest it was computed against. Two runs that
disagree are then immediately separable into "the data moved" and "the code
moved", which is the distinction a reader of a drifting number actually needs.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from .config import ARTIFACTS, CODING_SHEET, INTERIM, OUT, RELEASE_COL, WINDOW_DAYS
from .percentiles import add_percentiles
from .snapshot import MANIFEST
from .stats import bootstrap_mean

OUTPUT = OUT / "numbers.json"


def snapshot_digest():
    """One digest over the pinned manifest, so a moved input is visible here."""
    if not MANIFEST.exists():
        return None
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:16]


def _clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def collect():
    panel = pd.read_csv(INTERIM / "panel.csv", parse_dates=["Release date"])
    panel = add_percentiles(panel)
    panel["group"] = np.where(panel["eligible"], "eligible",
                              np.where(panel["placebo"], "placebo", "unknown"))

    out = {
        "snapshot_manifest_digest": snapshot_digest(),
        "window_days": WINDOW_DAYS,
        "panel": {
            "releases": panel[RELEASE_COL].nunique(),
            "benchmarks": panel["slug"].nunique(),
            "scored_pairs": len(panel),
            "eligible_pairs": int(panel["eligible"].sum()),
            "placebo_pairs": int(panel["placebo"].sum()),
            "unknown_vintage_pairs": int(
                (~panel["eligible"] & ~panel["placebo"]).sum()
            ),
            "organisations": panel["primary_org"].nunique(),
            "date_known_share": float(
                (panel["eligible"] | panel["placebo"]).mean()
            ),
        },
    }

    # --- the placebo contrast that uses no disclosure coding ---------------
    from .falsification import eligible_vs_placebo

    placebo = {}
    for value, key in (("percentile", "windowed"),
                       ("percentile_balanced", "side_balanced")):
        gaps = eligible_vs_placebo(panel, value)
        if gaps.empty:
            continue
        mean, (lo, hi) = bootstrap_mean(
            gaps["gap"].to_numpy(float), cluster=gaps["Organization"].to_numpy()
        )
        placebo[key] = {
            "releases": len(gaps),
            "providers": int(gaps["Organization"].nunique()),
            "mean": mean,
            "median": float(gaps["gap"].median()),
            "share_positive": float((gaps["gap"] > 0).mean()),
            "ci_low": lo,
            "ci_high": hi,
        }
    out["placebo_without_labels"] = placebo

    # --- peer-window geometry ---------------------------------------------
    shares = panel.groupby("group")["newer_share"].mean()
    out["peer_window"] = {
        "newer_share_eligible": _clean(shares.get("eligible", np.nan)),
        "newer_share_placebo": _clean(shares.get("placebo", np.nan)),
        "balanced_defined_share": float(panel["percentile_balanced"].notna().mean()),
    }
    ok = panel.dropna(subset=["percentile_sided", "newer_share"])
    recomposed = (
        (1 - ok["newer_share"]) * ok["percentile_older"].fillna(0)
        + ok["newer_share"] * ok["percentile_newer"].fillna(0)
    )
    out["peer_window"]["decomposition_max_abs_error"] = float(
        np.abs(ok["percentile_sided"] - recomposed).max()
    )

    # --- the coding -------------------------------------------------------
    if ARTIFACTS.exists():
        artifacts = pd.read_csv(ARTIFACTS, dtype=str)
        out["coding"] = {
            "releases_on_worklist": len(artifacts),
            "releases_read": int((artifacts["fetch_status"] == "ok").sum()),
            "releases_blocked": int((artifacts["fetch_status"] == "blocked").sum()),
            "releases_flagged": int(
                artifacts["flagged_for_review"].fillna("").astype(str).str.strip().ne("").sum()
            ),
        }
    if CODING_SHEET.exists():
        sheet = pd.read_csv(CODING_SHEET)
        coded = sheet[sheet["orbit_category"].notna()
                      & (sheet["orbit_category"] != "")]
        out.setdefault("coding", {})["coded_cells"] = len(coded)
        out["coding"]["categories"] = {
            str(k): int(v) for k, v in coded["orbit_category"].value_counts().items()
        }
        out["coding"]["drops"] = int((coded["orbit_category"] == "E").sum())
        out["coding"]["high_suspicion_share"] = float(
            coded["orbit_category"].isin({"D", "E", "G"}).mean()
        )

    return json.loads(json.dumps(out, default=_clean))


def main():
    numbers = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps(numbers, indent=2, sort_keys=True))
    print(f"\nwrote {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
