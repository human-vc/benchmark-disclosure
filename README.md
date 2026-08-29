# Supplementary code and data

Code and data for the submission. The first-pass coder is coder1; the independent second coder is coder2.

## Setup

    python3 -m pip install -r requirements.txt

## Reproduce

    python3 -m pytest                  # test suite
    python3 -m src.paper_numbers       # every number quoted in the paper -> data/paper_numbers.json
    python3 -m src.figures             # tables
    python3 -m src.figures_tikz        # figures

Analysis entry points: `src/falsification.py` (availability gap), `src/boundary.py` (discontinuity), `src/apc.py` (identification bound), `src/selectivity.py` (coded contrasts), `src/possession.py` (withholding tests), `src/helm_external.py` (HELM evidence), `src/reliability.py` (second coding). The panel is a pinned snapshot (`data/`); `src/snapshot.py` verifies it. The coding protocol is in `protocol/`.
