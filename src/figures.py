"""Every figure and every table in the manuscript, regenerated from one command."""

import json

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import helm_external as helm
from .config import INTERIM, RELEASE_COL, ROOT, WINDOW_DAYS
from .percentiles import side_balanced_percentile, within_benchmark_percentile

FIGDIR = ROOT / "paper" / "figures"
NUMBERS = ROOT / "data" / "paper_numbers.json"

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
GREY = "#9A9A9A"
FAINT = "#D6D6D6"

TEXTWIDTH = 5.5


def style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "lines.linewidth": 1.0,
        "lines.solid_capstyle": "round",
        "grid.linewidth": 0.4,
        "grid.color": FAINT,
    })


def load_panel():
    """The analysis panel with both standing measures attached."""
    panel = pd.read_csv(
        INTERIM / "panel.csv",
        parse_dates=["Release date", "benchmark_release_date"],
    )
    return side_balanced_percentile(within_benchmark_percentile(panel))


def load_numbers(path=NUMBERS):
    return json.loads(path.read_text())


def cluster_se(values, clusters):
    """Standard error of a mean, clustered on the provider."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    n = len(values)
    if n < 2:
        return np.nan
    resid = values - values.mean()
    totals = pd.Series(resid).groupby(pd.Series(clusters)).sum().to_numpy()
    n_g = len(totals)
    if n_g < 2:
        return np.nan
    adjust = n_g / (n_g - 1)
    return float(np.sqrt(adjust * (totals ** 2).sum()) / n)


def binned(frame, x, y, edges, cluster="primary_org", min_cells=20):
    """Bin means of y on x with clustered 95 percent intervals."""
    frame = frame.dropna(subset=[x, y])
    idx = np.digitize(frame[x].to_numpy(), edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        cell = frame[idx == b]
        if len(cell) < min_cells:
            continue
        rows.append({
            "x": float(cell[x].mean()),
            "y": float(cell[y].mean()),
            "se": cluster_se(cell[y], cell[cluster]),
            "n": len(cell),
        })
    return pd.DataFrame(rows)


def _demean(frame, column, by=RELEASE_COL):
    return frame[column] - frame.groupby(by)[column].transform("mean")


def figure_one(panel, numbers, path=None):
    """One curve, two positions on it."""
    path = path or FIGDIR / "fig1_one_curve.pdf"
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(
        subset=["percentile", "share_newer"])
    edges = np.arange(0.0, 1.0001, 0.1)

    pooled = binned(cells, "share_newer", "percentile", edges, min_cells=20)
    elig = binned(cells[cells["eligible"]], "share_newer", "percentile", edges)
    plac = binned(cells[cells["placebo"]], "share_newer", "percentile", edges)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.1))
    ax.axvline(0.5, color=OKABE["black"], lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.annotate("symmetric window", xy=(0.5, 97), xytext=(0.485, 97),
                ha="right", va="top", fontsize=6.5, color=OKABE["black"])

    ax.plot(pooled["x"], pooled["y"], color=GREY, lw=2.4, alpha=0.55,
            solid_capstyle="round", zorder=2,
            label="all eligible and placebo cells, one curve")

    for frame, colour, marker, label in (
        (elig, OKABE["blue"], "o", "eligible cells (benchmark predates release)"),
        (plac, OKABE["vermillion"], "s", "placebo cells (benchmark postdates release)"),
    ):
        ax.errorbar(frame["x"], frame["y"], yerr=1.96 * frame["se"],
                    fmt=marker, ms=3.6, lw=0, elinewidth=0.8, capsize=1.8,
                    capthick=0.8, color=colour, ecolor=colour, zorder=3,
                    label=label)

    means = {}
    for key, colour in (("eligible", OKABE["blue"]), ("placebo", OKABE["vermillion"])):
        group = cells[cells[key]]
        mx, my = group["share_newer"].mean(), group["percentile"].mean()
        means[key] = (mx, my)
        ax.plot([mx], [my], marker="D", ms=6.2, color=colour, mec="white",
                mew=0.8, zorder=5)
        ax.annotate(f"{key} mean\n{mx:.2f}", xy=(mx, my),
                    xytext=(mx, my - 13 if key == "eligible" else my + 11),
                    ha="center", va="top" if key == "eligible" else "bottom",
                    fontsize=6.5, color=colour,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=colour,
                                    shrinkA=0.5, shrinkB=4))

    gap = means["eligible"][1] - means["placebo"][1]
    within = numbers["placebo_null"]["windowed_percentile"]["mean"]
    ax.text(0.015, 0.045,
            f"eligible minus placebo: {gap:+.1f} points pooled, "
            f"{within:+.2f} within release,\nand the two groups sit "
            f"{means['placebo'][0] - means['eligible'][0]:.2f} apart in window composition",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            color=OKABE["black"])

    ax.set_xlabel("share of the ranking window released after the focal model "
                  "(0 = all peers older, 1 = all peers newer)")
    ax.set_ylabel("mean within-benchmark standing\n(windowed percentile, points)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)

    fig.savefig(path)
    plt.close(fig)
    return path


def _drift_series(path, headline):
    payload, order = helm.load(path)
    models = helm.frozen_models(payload, order)
    drift = helm.headline_drift(payload, order, models, headline=headline)
    pools = [len(payload["releases"][v]["rows"]) for v in order]
    return order, drift, pools


def _crossings(series, pairs):
    """Where a pair changes order, in release-index coordinates."""
    marks = []
    for a, b in pairs:
        ya, yb = np.array(series[a]), np.array(series[b])
        gap = ya - yb
        for i in range(len(gap) - 1):
            if gap[i] == 0 or gap[i] * gap[i + 1] >= 0:
                continue
            t = gap[i] / (gap[i] - gap[i + 1])
            marks.append((i + t, ya[i] + t * (ya[i + 1] - ya[i])))
    return marks


def figure_two(numbers, path=None):
    """The same bias on a leaderboard we did not build, and its control."""
    path = path or FIGDIR / "fig2_helm.pdf"
    order, lite, lite_pool = _drift_series(helm.FROZEN, helm.HEADLINE)
    ctl_order, ctl, ctl_pool = _drift_series(helm.CONTROL, helm.CONTROL_HEADLINE)

    reversing_pairs = [(c["a"], c["b"]) for c in
                       numbers["helm_lite"]["headline"]["reversal_examples"]]
    involved = {m for pair in reversing_pairs for m in pair}

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9), sharey=True,
                             gridspec_kw={"wspace": 0.06})

    ax = axes[0]
    for model, values in lite["series"].items():
        if model in involved:
            continue
        ax.plot(range(len(order)), values, color=GREY, lw=0.5, alpha=0.6, zorder=2)
    for model in sorted(involved):
        ax.plot(range(len(order)), lite["series"][model],
                color=OKABE["vermillion"], lw=1.0, alpha=0.95, zorder=3)
    marks = _crossings(lite["series"], reversing_pairs)
    ax.scatter([m[0] for m in marks], [m[1] for m in marks], s=13,
               facecolors="none", edgecolors=OKABE["black"], linewidths=0.7,
               zorder=4)

    head = numbers["helm_lite"]["headline"]
    ax.set_title("HELM Lite: headline is Mean win rate,\nwhich is pool-relative",
                 pad=5)
    ax.set_xlabel("release (v1.$x$.0), pool "
                  f"{lite_pool[0]} to {lite_pool[-1]} models")
    ax.set_ylabel("published headline value")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([v.split(".")[1] for v in order])
    ax.text(0.02, 0.03,
            f"{numbers['helm_lite']['frozen_cells']['cells']} published scenario "
            f"scores identical (max spread "
            f"{numbers['helm_lite']['frozen_cells']['max_spread']:.0f})\n"
            f"{head['moved']} of {head['models']} headline values move, "
            f"mean |change| {head['mean_abs_change']:.3f}\n"
            f"{head['reversals_endpoint']} of {head['pairs']} pairs finish in the "
            "opposite order",
            transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left")
    fig.legend(handles=[
        Line2D([], [], color=OKABE["vermillion"], lw=1.0,
               label="model in a pair that finishes reordered"),
        Line2D([], [], color=GREY, lw=0.6, label="other model present in every release"),
        Line2D([], [], color=OKABE["black"], lw=0, marker="o", ms=3.4,
               markerfacecolor="none", markeredgewidth=0.7,
               label="a pair changes order here"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=3,
        handlelength=1.5, columnspacing=1.4)

    ax = axes[1]
    for values in ctl["series"].values():
        ax.plot(range(len(ctl_order)), values, color=OKABE["blue"], lw=0.6,
                alpha=0.75, zorder=2)
    ax.set_title("HELM Capabilities: headline is Mean score,\nwhich is absolute",
                 pad=5)
    ax.set_xlabel("release (v1.$x$.0), pool "
                  f"{ctl_pool[0]} to {ctl_pool[-1]} models")
    ax.set_xticks(range(len(ctl_order)))
    ax.set_xticklabels([v.split(".")[1] for v in ctl_order])
    ax.text(0.02, 0.03,
            f"{len(ctl['series'])} models present throughout\n"
            f"{ctl['moved']} headline values move\n"
            f"{ctl['reversals_endpoint']} pairs reorder, at the endpoints or ever",
            transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left")

    for axis in axes:
        axis.set_ylim(0, 1)
        axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.margins(x=0.02)

    fig.savefig(path)
    plt.close(fig)
    return path


def _window_facts(panel, slug, focal_date, window_days=WINDOW_DAYS):
    group = panel[panel["slug"] == slug].sort_values("Release date")
    days = group["Release date"].to_numpy("datetime64[D]").astype(np.int64)
    focal = np.datetime64(pd.Timestamp(focal_date), "D").astype(np.int64)
    within = np.abs(days - focal) <= window_days
    newer = within & (days > focal)
    older = within & (days < focal)
    row = group[group["Release date"] == pd.Timestamp(focal_date)].iloc[0]
    return {
        "dates": group["Release date"],
        "model": row["Model name"],
        "focal": pd.Timestamp(focal_date),
        "n_peers": int(within.sum()),
        "n_newer": int(newer.sum()),
        "n_older": int(older.sum()),
        "share_newer": float(row["share_newer"]),
        "percentile": float(row["percentile"]),
        "benchmark": row["benchmark_name"],
        "benchmark_date": row["benchmark_release_date"],
    }

def figure_three(panel, slug="hle", boundary="2024-09-24", interior="2025-12-11",
                 path=None, window_days=WINDOW_DAYS):
    """Why the window is symmetric in days and asymmetric in peers."""
    path = path or FIGDIR / "fig3_window_geometry.pdf"
    low = _window_facts(panel, slug, boundary, window_days)
    high = _window_facts(panel, slug, interior, window_days)
    dates = low["dates"]
    half = pd.Timedelta(days=window_days)
    first, last = dates.min(), dates.max()

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.9))
    strip = 0.16
    lanes = ((low, 1.0, OKABE["vermillion"],
              "one-sided window at the start of coverage"),
             (high, 0.0, OKABE["blue"],
              "balanced window in the interior of coverage"))

    span_lo = min(first, low["focal"] - half) - pd.Timedelta(days=45)
    span_hi = last + pd.Timedelta(days=45)

    for facts, y, colour, title in lanes:
        ax.add_patch(Rectangle(
            (mdates.date2num(first), y - strip / 2),
            mdates.date2num(last) - mdates.date2num(first), strip,
            facecolor="#EFEFEF", edgecolor="none", zorder=1))
        left, right = facts["focal"] - half, facts["focal"] + half
        ax.add_patch(Rectangle(
            (mdates.date2num(left), y - strip / 2),
            mdates.date2num(right) - mdates.date2num(left), strip,
            facecolor=colour, alpha=0.15, edgecolor=colour, lw=0.7, zorder=2))

        empty = left < first
        if empty:
            ax.add_patch(Rectangle(
                (mdates.date2num(left), y - strip / 2),
                mdates.date2num(first) - mdates.date2num(left), strip,
                facecolor="none", edgecolor=colour, lw=0.6, hatch="/////",
                alpha=0.6, zorder=3))

        inside = (dates >= left) & (dates <= right)
        ax.vlines(dates[~inside], y - strip / 2 + 0.012, y + strip / 2 - 0.012,
                  color=GREY, lw=0.7, zorder=4)
        ax.vlines(dates[inside], y - strip / 2 + 0.012, y + strip / 2 - 0.012,
                  color=colour, lw=1.0, zorder=5)
        ax.plot([facts["focal"]], [y + strip / 2 + 0.055], marker="v", ms=4.5,
                color=colour, mec="white", mew=0.5, zorder=6)
        ax.annotate(facts["model"],
                    xy=(facts["focal"], y + strip / 2 + 0.095),
                    ha="center", va="bottom", fontsize=6.4, color=colour)
        caption = (f"{facts['n_peers']} peers in the window: "
                   f"{facts['n_older']} older, {facts['n_newer']} newer, "
                   f"{facts['share_newer']:.0%} newer")
        if empty:
            caption += "\nthe hatched half of the window has no scored models in it"
        ax.annotate(caption, xy=(left, y - strip / 2 - 0.05), ha="left",
                    va="top", fontsize=6.8, color=colour)
        ax.annotate(title, xy=(mdates.date2num(span_lo), y + strip / 2 + 0.28),
                    ha="left", va="bottom", fontsize=7.2, color=colour)

    bench = low["benchmark_date"]
    for bottom, top in ((0.90, 1.55), (-0.09, 0.28)):
        ax.vlines(bench, bottom, top, color=OKABE["black"], lw=0.7,
                  ls=(0, (3, 2)), zorder=6)
    ax.annotate(f"{low['benchmark']} published {bench:%Y-%m-%d}",
                xy=(bench + pd.Timedelta(days=20), 1.62), ha="left", va="top",
                fontsize=6.8)
    ax.annotate("coverage of this benchmark begins here",
                xy=(first, -strip / 2 - 0.02),
                xytext=(first + pd.Timedelta(days=70), -0.60),
                ha="left", va="center", fontsize=6.8, color=OKABE["black"],
                arrowprops=dict(arrowstyle="->", lw=0.6, color=OKABE["black"],
                                connectionstyle="angle,angleA=0,angleB=90,rad=2",
                                shrinkA=2, shrinkB=1))

    ax.set_ylim(-0.8, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(span_lo, span_hi)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="both", direction="out")
    ax.set_xlabel(f"model release date; each tick is a model with a published "
                  f"{low['benchmark']} score, window half-width {window_days} days")

    fig.savefig(path)
    plt.close(fig)
    return path


def figure_four(panel, numbers, path=None):
    """What the repair does to the gradient it is meant to remove."""
    path = path or FIGDIR / "fig4_correction.pdf"
    slopes = numbers["asymmetry"]
    specs = [
        ("percentile", slopes["slope_windowed"], OKABE["vermillion"],
         "windowed percentile"),
        ("pct_balanced", slopes["slope_balanced"], OKABE["blue"],
         "side-balanced percentile"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.85), sharey=True,
                             gridspec_kw={"wspace": 0.07})

    edges = np.array([-0.55, -0.35, -0.25, -0.175, -0.115, -0.06, -0.02,
                      0.02, 0.06, 0.115, 0.175, 0.25, 0.45])
    for ax, (column, slope, colour, label) in zip(axes, specs):
        cells = panel[panel["eligible"]].dropna(subset=[column, "share_newer"]).copy()
        cells["y"] = _demean(cells, column) + cells[column].mean()
        cells["x"] = _demean(cells, "share_newer")
        points = binned(cells, "x", "y", edges, min_cells=25)

        grid = np.array([edges[0], edges[-1]])
        ax.plot(grid, cells["y"].mean() + slope * grid, color=colour, lw=1.2,
                zorder=3)
        ax.errorbar(points["x"], points["y"], yerr=1.96 * points["se"], fmt="o",
                    ms=3.4, lw=0, elinewidth=0.8, capsize=1.8, capthick=0.8,
                    color=colour, ecolor=colour, zorder=4)
        ax.axvline(0, color=GREY, lw=0.6, zorder=1)

        ax.set_title(label, pad=5)
        ax.set_xlabel("peer-window asymmetry,\nwithin-release deviation (share)")
        ax.annotate(f"slope {slope:+.2f} percentile points per unit share\n"
                    f"{len(cells):,} eligible cells",
                    xy=(0.03, 0.04), xycoords="axes fraction", fontsize=6.8,
                    color=colour, ha="left", va="bottom")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

    axes[0].set_ylabel("standing among eligible cells\n(points, within-release "
                       "deviation, recentred)")
    axes[0].set_ylim(15, 75)
    axes[0].set_yticks([20, 30, 40, 50, 60, 70])

    fig.savefig(path)
    plt.close(fig)
    return path


LADDER = [
    ("release FE", r"Release fixed effects only"),
    ("+ benchmark FE", r"\quad + benchmark fixed effects (two-way absorbed)"),
    ("+ peer-window asymmetry", r"\quad + peer-window asymmetry"),
    ("+ peer count", r"\quad + peer count"),
    ("two-way FE + both", r"Two-way fixed effects + both window terms"),
]


def table_one(numbers, path=None):
    """The decomposition ladder, written from data/paper_numbers.json."""
    path = path or FIGDIR / "table1_decomposition.tex"
    ladder = numbers["decomposition"]
    panel = numbers["panel"]
    cells = panel["eligible"] + panel["placebo"]
    clusters = ladder["release FE"]["n_clusters"]
    benchmarks = ladder["release FE"]["n_benchmarks"]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Conditioning set & Placebo coefficient & $t$ & Absorbed \\",
        r"\midrule",
    ]
    for key, label in LADDER:
        row = ladder[key]
        lines.append(
            f"{label} & ${row['coef']:+.3f}$ ({row['se']:.3f}) & "
            f"${row['t']:+.3f}$ & {row['absorbed'] * 100:.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{%",
        r"Each row is the coefficient on a placebo indicator in a regression of "
        r"within-benchmark standing on that indicator, always within release, "
        r"differencing out the release's capability level. "
        f"Standard errors in parentheses cluster on provider and benchmark "
        f"jointly, {clusters} by {benchmarks} clusters; provider-only errors "
        r"are uniformly smaller and sit in Appendix~\ref{sec:supporting}. "
        f"$n = {cells}$ cells "
        f"({panel['eligible']} eligible, {panel['placebo']} placebo). "
        r"Absorbed is the reduction in the absolute coefficient relative to the "
        r"first row. Peer count is a suppressor when entered alone, so the final "
        r"row is joint conditioning on both window terms and must not be read as "
        r"peer count explaining a remainder.}",
        r"\label{tab:decomposition}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def _midrank_alltime(panel):
    """All-time within-benchmark standing, on the same midrank convention."""
    out = pd.Series(np.nan, index=panel.index)
    for _, group in panel.groupby("slug", sort=False):
        scores = group["score"].to_numpy(float)
        below = (scores[None, :] < scores[:, None]).sum(axis=1)
        equal = (scores[None, :] == scores[:, None]).sum(axis=1)
        out.loc[group.index] = 100.0 * (below + 0.5 * equal) / len(scores)
    return out


def _release_contrast(panel, column):
    cells = panel[panel["eligible"] | panel["placebo"]].dropna(subset=[column])
    wide = (
        cells.assign(side=np.where(cells["eligible"], "eligible", "placebo"))
        .pivot_table(index=RELEASE_COL, columns="side", values=column, aggfunc="mean")
        .dropna()
    )
    gap = wide["eligible"] - wide["placebo"]
    return {"mean": float(gap.mean()), "median": float(gap.median()),
            "share_positive": float((gap > 0).mean()), "n_releases": int(len(gap))}


def table_two(numbers, panel=None, path=None):
    """The remedy table, including the remedies that fail."""
    path = path or FIGDIR / "table2_remedies.tex"
    rows = [
        (r"Windowed percentile (under audit)",
         numbers["placebo_null"]["windowed_percentile"], "100.0"),
    ]
    if panel is not None:
        alltime = panel.assign(alltime=_midrank_alltime(panel))
        rows.append((r"Rank against all models ever",
                     _release_contrast(alltime, "alltime"), "100.0"))
        kept = panel["share_newer"].between(0.25, 0.75) & (panel["n_peers"] >= 5)
        trimmed = _release_contrast(panel[kept], "percentile")
        rows.append((r"Trim to two-sided windows",
                     trimmed, f"{100 * kept.mean():.1f}"))
    rows.append((r"Side-balanced percentile",
                 numbers["placebo_null"]["side_balanced"],
                 f"{100 * numbers['balanced_coverage']['defined_share_eligible_or_placebo']:.1f}"))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Standing measure & Mean & Median & Positive & Releases & Cells \\",
        r"\midrule",
    ]
    for label, stat, kept in rows:
        lines.append(
            f"{label} & ${stat['mean']:+.2f}$ & ${stat['median']:+.2f}$ & "
            f"{stat['share_positive'] * 100:.1f}\\% & {stat['n_releases']} & "
            f"{kept}\\% \\\\"
        )
    slopes = numbers["asymmetry"]
    base = numbers["placebo_null"]["windowed_percentile"]["mean"]
    trim_note = ""
    if len(rows) > 2:
        removed = 1 - rows[2][1]["mean"] / base
        trim_note = (f"Trimming to two-sided windows removes only "
                     f"{removed * 100:.0f}\\% of the contamination while discarding "
                     f"{100 - float(rows[2][2]):.0f}\\% of the cells. ")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{%",
        r"The placebo null under each candidate measure: the release-level mean "
        r"standing of eligible benchmarks minus that of postdating ones, computed "
        r"with no disclosure labels of any kind. Positive is the share of releases "
        r"whose gap is above zero and Cells is the share of panel cells on which "
        r"the measure is defined. Every entry should be zero and "
        r"none is. Ranking against every model ever scored nearly doubles the "
        r"contamination. " + trim_note +
        r"Raw scores are not comparable across benchmarks, "
        r"so they have no common unit in which to state the same contrast. The "
        r"side-balanced measure is undefined for cells with an empty side.}",
        r"\label{tab:remedies}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def table_three(numbers, path=None):
    """The coded deficit beside the artifact that predicts it."""
    path = path or FIGDIR / "table3_coding.tex"
    if "coding" not in numbers:
        return path
    specs = numbers["coding"]["deficit_by_spec"]
    rows = [
        ("Windowed percentile, $\\pm$182d", "windowed_182"),
        ("Side-balanced percentile, $\\pm$182d", "side_balanced_182"),
        ("Rank against all models ever", "alltime"),
        ("Windowed percentile, $\\pm$90d", "windowed_90"),
        ("Windowed percentile, $\\pm$365d", "windowed_365"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Standing measure & Coded gap & 95\% CI & No labels & Difference \\",
        r"\midrule",
    ]
    for label, key in rows:
        s = specs[key]
        lines.append(
            f"{label} & ${s['deficit']:+.2f}$ & $[{s['low']:+.1f}, {s['high']:+.1f}]$"
            f" & ${s['null']:+.2f}$ & ${s['deficit'] - s['null']:+.2f}$ \\\\"
        )
    w = specs["windowed_182"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{%",
        "The coded omission deficit beside the label-free artifact. Coded gap is the release-level "
        "mean standing of omitted-eligible minus postdating benchmarks over "
        f"{w['n_releases']} releases and {w['n_providers']} providers "
        f"({specs['side_balanced_182']['n_releases']} side-balanced), provider-clustered bootstrap "
        "intervals; No labels is the same contrast with no disclosure coding. Four of five coded "
        "gaps exclude zero; every one sits within 1.3 points of its label-free counterpart.}",
        r"\label{tab:coding}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_tables(numbers=None, panel=None, out_dir=FIGDIR):
    """Both LaTeX tables, from the regenerated numbers file."""
    numbers = numbers if numbers is not None else load_numbers()
    return [
        table_one(numbers, path=out_dir / "table1_decomposition.tex"),
        table_two(numbers, panel=panel, path=out_dir / "table2_remedies.tex"),
        table_three(numbers, path=out_dir / "table3_coding.tex"),
    ]


def main():
    style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    numbers = load_numbers()
    panel = load_panel()

    produced = [
        figure_one(panel, numbers),
        figure_two(numbers),
        figure_three(panel),
        figure_four(panel, numbers),
        *write_tables(numbers, panel),
    ]
    for item in produced:
        size = item.stat().st_size
        floor = 4000 if item.suffix == ".pdf" else 400
        flag = "ok " if size > floor else "THIN"
        print(f"  {flag} {item.relative_to(ROOT)}  {size / 1024:.1f} kB")
    print(f"snapshot: {numbers['_provenance']['snapshot']}")


if __name__ == "__main__":
    main()
