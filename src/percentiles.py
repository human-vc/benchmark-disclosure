"""Within-benchmark standing, computed against contemporaneous models."""

import numpy as np
import pandas as pd

from .config import RELEASE_COL, WINDOW_DAYS


def within_benchmark_percentile(panel, window_days=WINDOW_DAYS):
    """Percentile of each score among peers on the same benchmark within"""
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
    """Standing that does not depend on which side of a benchmark's birthday a"""
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
    """Attach both the windowed percentile and the all-time one."""
    panel = within_benchmark_percentile(panel, window_days)
    panel["percentile_alltime"] = (
        panel.groupby("slug")["score"].rank(pct=True) * 100
    )
    return panel


def capability_level(panel):
    """A release's overall standing: mean windowed percentile across every"""
    return panel.groupby(RELEASE_COL)["percentile"].mean().rename("level")
