# Paper–repo reconciliation

What has to be settled before the paper's numbers can be regenerated and
quoted. Written after repairing the `ae7f3e6` merge; the repair is commit
`710679a` and is independent of everything here.

## 1. The tree holds three data vintages at once

The merge brought `data/worklist.csv` back to the paper's vintage while leaving
the coding and the raw bundle at two later ones. Nothing errors; the numbers
just come from different builds depending on which file produced them.

| file | vintage | what it says |
|---|---|---|
| `data/snapshot.json` | pinned 2026-08-17 | 819 model-versions |
| `data/raw/` | fetched later | 839 model-versions, 29 files changed content |
| `data/worklist.csv` | paper's | 1,298 cells / 103 releases / 38 families |
| `data/artifacts.csv` | later | 108 releases coded |
| `data/disclosures.csv` | later | 1,685 rows, 5 releases absent from the worklist |

Concretely: five coded releases are not in the current worklist —

    Alibaba | Qwen 3.8 Max | 2026-08-02
    DeepSeek | DeepSeek V4 Pro 0813 | 2026-08-13
    Google DeepMind | Gemini 3.7 Flash | 2026-08-13
    Mistral AI | Mistral Medium 3.5 | 2026-04-28
    xAI | Grok 4.6 | 2026-08-12

Three of them are in the 26-release reliability draw, so `reliability.py` would
derive no cells for them and the kappa would be computed on 23 releases while
reporting 26.

`python -m src.paper_numbers` already refuses to cite its own output:

    snapshot: DRIFT from 2026-08-17 (30 files differ; index 819 pinned vs 839 on disk)
      REFUSE TO CITE THESE NUMBERS: the data on disk is not the pinned build

**This is the decision.** Not "re-pin or pin to the paper's snapshot" — that
framing assumed one vintage was on disk. All three have to be brought to one,
and `data/raw/` at 819 is not reconstructible from this repo: `zip_sha256` in
the manifest is empty, so even an archived Epoch bundle could not be verified
against the pin. Whoever generated the committed `paper_numbers.json` on
2026-08-25 had a matching tree, so the 819 build exists on their machine.

Options:

- **Recover 819** — get `data/raw/` from whoever holds it, rebuild worklist and
  panel, re-derive. Paper's printed numbers survive except where the convention
  changes them (§2). Requires a collaborator's disk.
- **Re-pin to current** — `python -m src.download_data --capture`, rebuild
  everything, re-derive every number in the paper. Self-contained, and it
  finally records a `zip_sha256` so the next pin is verifiable. Costs the
  paper's current numbers.

Re-pinning is a deliberate act, not a repair; `src/snapshot.py` is written that
way on purpose. Do not let it happen as a side effect of a rebuild.

## 2. The convention change, isolated from the drift

`710679a` moved the older/newer split to the strict convention (a model is not
its own older peer). Both columns below are computed on **today's** data, so
the difference is the convention alone.

| quantity | loose | strict |
|---|---|---|
| placebo mean, side-balanced | +8.698 | **+9.148** |
| slope, side-balanced | −9.269 | **−9.690** |
| share newer, eligible | 0.4385 | **0.4692** |
| share newer, placebo | 0.6322 | **0.7066** |
| balanced coverage, all cells | 96.21% | **92.38%** |

The convention costs 3.8 points of coverage and moves the balanced placebo mean
up 0.45. It does not change the sign or the conclusion.

Quantities the convention does not touch — `percentile`, `slope_windowed`,
panel counts — moved anyway, and that movement is pure data drift:

| quantity | committed (819) | now (839) |
|---|---|---|
| releases | 353 | 354 |
| scored pairs | 3,082 | 3,138 |
| eligible pairs | 2,355 | 2,393 |
| placebo pairs | 476 | 476 |
| slope, windowed | −29.09 | −30.90 |
| placebo mean, windowed | +12.227 | +12.225 |

`slope_balanced` is far more sensitive to the data (−8.02 → −9.27) than to the
convention (−9.27 → −9.69). Worth knowing before attributing its movement.

## 3. The identity claim has to be reworded, not renumbered

`core.tex:29` states

> percentile = (1 − s) P_old + s P_new, exact cell by cell to 2.84×10⁻¹⁴

Under the strict convention this is false against `percentile` and exact — to
**0.0**, not merely small — against `pct_sided`, the same average taken over
peers only. The two differ by the focal model's own midrank contribution, at
most **31.25 percentile points** on a single cell in the current panel.

The loose convention's exactness against `percentile` was not a check on the
arithmetic. It held *because* the model was counted among its own peers, which
is the defect. The sentence needs to say which percentile decomposes, and the
self-inclusion gap is now emitted as `self_inclusion_max_abs_gap`.

## 4. Every paper location carrying an affected number

Nothing here is edited yet — the final values depend on §1.

| file:line | carries | moved by |
|---|---|---|
| `abstract.tex:7` | 12.23 | data |
| `abstract.tex:8` | 8.69 | convention |
| `intro.tex:7` | 353, 3,082, 2,355 | data |
| `intro.tex:9` | 12.23, 78.1%, 8.69 | data + convention |
| `data.tex:19,23` | 353, 3,082, 2,355 | data |
| `core.tex:7` | 12.23, 78.1%, 8.69 | data + convention |
| `core.tex:27` | 0.4374, 0.6320, −29.09 | data + convention |
| `core.tex:29` | 2.84×10⁻¹⁴, 96.4%, 96.5%, −8.02 | **reword** + convention |
| `core.tex:38` | 0.437, 0.632 | convention |
| `core.tex:47` | −29.09 | data |
| `core.tex:59` | 8.69, −8.02 | convention |
| `supporting.tex:54,65` | 12.23, −29.09 | data |
| `reach.tex:7` | 12.23 | data |
| `reach.tex:9` | 1,298, 103 releases, 38 families | §1 |

## 5. Separately, and not affected by any of the above

`checklist.tex:102`, `reach.tex:9` and `data.tex:25` state that no cells are
coded and that the language model's labels "have been cleared". The repo ships
1,280 coded cells in `data/disclosures.csv`, every one stamped with the tool's default coder name.
That contradiction is independent of the vintage question and has to be
resolved before a public push either way.
