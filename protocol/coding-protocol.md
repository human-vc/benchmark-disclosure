# Disclosure coding protocol

Adapted from ORBIT, the Outcome Reporting Bias In Trials classification system
(Kirkham, Dwan, Altman, Gamble, Dodd, Smyth & Williamson, *BMJ* 2010;340:c365).

ORBIT was built for a problem structurally identical to this one. A trial
registers a set of outcomes, publishes a subset, and the reader must decide
whether a missing outcome was suppressed or was never measured. Substitute
model release for trial and benchmark for outcome and the instrument transfers
with almost no modification. Its central virtue is that it refuses to collapse
"was it measured" and "was it reported given that it was measured" into a
single judgment, which is exactly the distinction this study lives or dies on.

## Unit of coding

One row per (model release, benchmark) pair, where the benchmark is in the
model's **eligible choice set**: an independent score exists, and the benchmark
was publicly released before the model shipped.

## Source hierarchy

Code from the earliest available artifact, in this order. Record which was used.

1. Official model card or system card published at release.
2. Official release blog post.
3. Technical report or arXiv paper released alongside the model.
4. Repository README for open-weights releases.

Do **not** code from third-party summaries, aggregators, or later marketing
pages. Where a provider updated an artifact after release, prefer the release-
date snapshot. For open-weights models the Hugging Face model card commit
history gives a reliable snapshot. The Internet Archive is the fallback and was
unavailable when this protocol was written, so do not build a hard dependency
on it.

## Categories

Assign exactly one. Categories A through F apply when the benchmark was clearly
run; G through I apply when that is itself uncertain.

**Benchmark clearly run and reported**

- **A** — Reported with a numeric score. No suspicion.
- **B** — Reported only as a qualitative claim, no number given. Low suspicion.
- **C** — Reported, but presented in a form that cannot be compared against the
  independent score (different subset, different scaffold, different metric,
  no denominator). Moderate suspicion; record what was ambiguous.

**Benchmark clearly run, not reported**

- **D** — Stated to have been evaluated, but no result given anywhere in the
  artifact. High suspicion.
- **E** — Not reported, and the provider reported this same benchmark for an
  earlier model in the same family. High suspicion. This is the **drop** case
  and is the primary analytic unit.
- **F** — Not reported, but there is a documented benign reason in the artifact
  itself (benchmark deprecated, superseded by a named successor, known
  contamination). Low suspicion; quote the stated reason.

**Whether the benchmark was run is unclear**

- **G** — Not mentioned, but comparable providers reported it for comparable
  models released in the same quarter. Moderate suspicion.
- **H** — Not mentioned, and no comparable provider reported it in the same
  window. Low suspicion.
- **I** — Benchmark plainly outside the model's modality or domain (a text
  benchmark for an audio model). No suspicion; exclude from analysis.

## Decision rules

The judgment calls that recur, resolved in advance so they are not resolved
case by case in the analyst's favour.

- A benchmark counts as **reported** only if a score for *this* model appears.
  A score for a prior model in a comparison table does not count.
- If the provider reports a benchmark **variant** (MMLU-Pro when the
  independent score is MMLU), code **C**, not A. Record both names.
- If a benchmark appears only in an appendix, it is still reported. Code A.
- If the artifact reports a benchmark the independent source does not cover,
  record it in the `reverse_gap` field. Do not discard it; the direction of
  that asymmetry is itself informative.
- Category **E** requires naming the earlier model and the artifact where the
  benchmark was previously reported.
- When torn between two categories, assign the **lower**-suspicion one and
  flag the row for second coding.

## Reliability

Double-code a random 20% of rows. Report Cohen's kappa on the collapsed
high-suspicion (D, E, G) versus low-suspicion (A, B, C, F, H, I) split, and
separately on the full nine-category assignment.

ORBIT's own validation is the honest benchmark to compare against, and it is
not flattering: trained reviewers applying the G/H distinction achieved 92%
sensitivity and 77% specificity against ground truth. This instrument should
not be expected to do better. Report the disagreement rate rather than
suppressing it.

## What this instrument cannot do

It cannot separate suppression from a provider who genuinely did not re-run an
evaluation for reasons uncorrelated with the expected score. That case is
observationally identical to concealment for any researcher without access to
internal evaluation logs, and it is the empirical shadow of Dye (1985). The
classification narrows the space; it does not close it. The argument for
strategic omission has to come from the joint pattern across many drops, not
from any single coded row.

## Sheet schema

See [`disclosure-template.csv`](disclosure-template.csv).
