# Reproducing every number

Python 3.9+ with `pip install -r requirements.txt`. All analysis runs in
seconds on a laptop; no GPU, no training.

## Data

    python -m src.download_data   # fetch Epoch's bundle (CC-BY)
    python -m src.snapshot        # verify against data/snapshot.json (SHA-256 per file)

A mismatch is reported, never absorbed. The HELM evidence is frozen in
`data/external/*.json` with a SHA-256 per source file inside each record.

## One command per claim

| Paper location | Command |
|---|---|
| panel counts (§2.2–2.3) | `python -m src.build_matrix` |
| +12.23 headline, Table 1, shuffle null, 101% trend recovery, drop ceiling, null calibration | `python -m src.placebo_calibration` |
| Table 2, sixteen-specification sweep, leave-one-organisation-out | `python -m src.sensitivity` |
| APC bound, 24–34% (§3.3, App. B) | `python -m src.apc` |
| discontinuity at the boundary (§3.2, App. B) | `python -m src.boundary` |
| joint ability scale (App. B) | `python -m src.ability` |
| HELM Lite and Capabilities (§4) | `python -m src.helm_external` |
| every manuscript number, machine-readable | `python -m src.paper_numbers` → `data/paper_numbers.json` |
| tables and figures, byte-identical regeneration | `python -m src.figures && python -m src.figures_tikz` |
| coverage diagnostics (App. B) | `python -m src.coverage` |

## Manuscript

    python paper/flatten.py       # writes dist/absence-overleaf.zip

The bundle regenerates byte-identically from the sections; figures and tables
are generated, never hand-edited. `pytest` runs the full suite, including a
test pinning the sensitivity sweep's baseline cell to the headline estimator.

## Disclosure coding

`docs/analysis-plan.md` is the frozen plan, committed before any label
existed. `data/artifacts.csv` holds the first coder's evidence record for all
108 locatable release documents and `data/disclosures.csv` the 1,280 derived
cells; `data/artifacts_second_coder.csv` holds the independent second coding
of 20 drawn releases. `python -m src.selectivity` and `python -m
src.falsification` reproduce the coded results, `python -m src.reliability`
the kappas, and the `coding` block of `data/paper_numbers.json` carries every
number Section 5 quotes, including Table 3.
