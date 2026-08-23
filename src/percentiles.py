"""Within-benchmark standing, computed against contemporaneous models.

docs/design.md: "Percentiles are taken within benchmark across contemporaneous
models, so 'standing' means position in the field rather than raw score."

The contemporaneity is not decoration. Ranking a 2023 model against the 2026
frontier pushes every early release toward the bottom of every benchmark
regardless of how it stood when it shipped, which is not what "standing" means
and which correlates with release date -- the same axis the temporal gate and
the drop design run along. A release is compared against the models a reader
could have compared it against at the time.
"""

import numpy as np
import pandas as pd

from .config import RELEASE_COL, WINDOW_DAYS


def within_benchmark_percentile(panel, window_days=WINDOW_DAYS):
    """Percentile of each score among peers on the same benchmark within
    +/- window_days of the focal release date.

    Ties take the midrank, so a benchmark where many models saturate at the
    same score does not hand an arbitrary advantage to whichever row sorts
    first. n_peers is retained: a percentile computed against two peers is not
    the same evidence as one computed against forty, and thin cells have to be
    visible rather than silently equal-weighted.

    share_newer is retained for a sharper reason. The window is symmetric in
    days but a benchmark's model coverage is not: it begins when the benchmark
    is built. A release sitting at that left boundary is ranked against a peer
    set drawn almost entirely from models newer than itself, and it therefore
    scores low for a reason that has nothing to do with its standing. Measured
    on this panel, share_newer carries -29.1 percentile points per unit even
    among cells whose benchmark predates them, so this is a property of the
    outcome variable everywhere and not a quirk of one comparison. Any estimator
    that contrasts groups with different peer-window composition has to
    condition on it. See src/placebo_calibration.py.
    """
    panel = panel.copy()
    panel["Release date"] = pd.to_datetime(panel["Release date"])

    percentile = np.full(len(panel), np.nan)
    n_peers = np.zeros(len(panel), dtype=int)
    share_newer = np.full(len(panel), np.nan)

    for _, group in panel.groupby("slug", sort=False):
        positions = group.index.to_numpy()
        rows = panel.index.get_indexer(positions)
        days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
        scores = group["score"].to_numpy(float)

        within = np.abs(days[:, None] - days[None, :]) <= window_days
        below = (scores[None, :] < scores[:, None]) & within
        equal = (scores[None, :] == scores[:, None]) & within
        newer = (days[None, :] > days[:, None]) & within
        counts = within.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100.0 * (below.sum(axis=1) + 0.5 * equal.sum(axis=1)) / counts
        percentile[rows] = pct
        n_peers[rows] = counts
        share_newer[rows] = newer.sum(axis=1) / counts

    panel["percentile"] = percentile
    panel["n_peers"] = n_peers
    panel["share_newer"] = share_newer
    return panel


def side_balanced_percentile(panel, window_days=WINDOW_DAYS):
    """Standing that does not depend on which side of a benchmark's birthday a
    model happened to ship.

    The windowed percentile decomposes exactly into the two sides of its own
    window. Writing P_old for the midrank fraction of within-window peers dated
    at or before the focal release that it beats, self counted at a half, and
    P_new for the same over strictly newer peers,

        percentile = (1 - share_newer) * P_old + share_newer * P_new

    holds cell by cell. The contamination is therefore not in either side. It is
    in the weight, which is the empirical share of the window that happens to be
    newer, and which is driven to one at the left edge of a benchmark's coverage.
    Replacing the empirical weight with a half gives a measure whose weighting no
    longer varies with position in that coverage.

    This is a partial repair and the paper says so. It fixes the weight, not
    which models an evaluator chose to score early in a benchmark's life, and
    P_old still carries a residual gradient on window asymmetry. Cells with an
    empty side have no balanced value and come back as NaN rather than being
    filled with the one-sided number.
    """
    panel = panel.copy()
    panel["Release date"] = pd.to_datetime(panel["Release date"])

    old_side = np.full(len(panel), np.nan)
    new_side = np.full(len(panel), np.nan)

    for _, group in panel.groupby("slug", sort=False):
        rows = panel.index.get_indexer(group.index.to_numpy())
        days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
        scores = group["score"].to_numpy(float)

        within = np.abs(days[:, None] - days[None, :]) <= window_days
        beats = ((scores[None, :] < scores[:, None]).astype(float)
                 + 0.5 * (scores[None, :] == scores[:, None]))
        newer = days[None, :] > days[:, None]

        for mask, target in ((within & ~newer, old_side), (within & newer, new_side)):
            counts = mask.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                target[rows] = np.where(counts > 0,
                                        (beats * mask).sum(axis=1) / counts, np.nan)

    panel["pct_old_side"] = 100.0 * old_side
    panel["pct_new_side"] = 100.0 * new_side
    panel["pct_balanced"] = 0.5 * panel["pct_old_side"] + 0.5 * panel["pct_new_side"]
    return panel


def add_percentiles(panel, window_days=WINDOW_DAYS):
    """Attach both the windowed percentile and the all-time one.

    The all-time column exists so the windowing choice can be shown not to
    drive the result, rather than asserted not to.
    """
    panel = within_benchmark_percentile(panel, window_days)
    panel["percentile_alltime"] = (
        panel.groupby("slug")["score"].rank(pct=True) * 100
    )
    return panel


def capability_level(panel):
    """A release's overall standing: mean windowed percentile across every
    benchmark it has an independent score on. This is the conditioning
    variable design.md calls the model's overall capability level."""
    return panel.groupby(RELEASE_COL)["percentile"].mean().rename("level")
