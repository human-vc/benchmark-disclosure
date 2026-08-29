# Absence Is Not Omission

Code and data for "Absence Is Not Omission: Availability, Vintage, and the Measurement of Selective Benchmark Disclosure."

## Setup

    python3 -m pip install -r requirements.txt

## Reproduce

    python3 -m pytest                  # test suite
    python3 -m src.paper_numbers       # every number quoted in the paper -> data/paper_numbers.json
    python3 -m src.figures             # tables  -> paper/figures/
    python3 -m src.figures_tikz        # figures -> paper/figures/
    tectonic paper/main.tex            # compile paper/main.pdf
    python3 paper/flatten.py           # self-contained submission bundle

Analysis entry points: `src/falsification.py` (availability gap), `src/boundary.py` (discontinuity), `src/apc.py` (identification bound), `src/selectivity.py` (coded contrasts), `src/possession.py` (withholding tests), `src/helm_external.py` (HELM evidence), `src/reliability.py` (second coding). The panel is a pinned snapshot (`data/`); `src/snapshot.py` verifies it. The coding protocol is in `protocol/`.
