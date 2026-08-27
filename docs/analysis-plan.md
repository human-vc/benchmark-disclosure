# Frozen analysis plan for the disclosure estimate

Committed before any hand-coded label exists. `data/disclosures.csv` carries no
ORBIT category at the time of this commit, which the git history shows. Nothing
below may change in response to coded results; changes after coding begins get
their own dated section at the bottom with a reason.

## Population and stopping rule

The coding population is the 108 release artifacts in `data/artifacts.csv`, in
`read_order`. Coding stops at row 108. If it must stop early, it stops at row
85, the last row that completes a drop-capable pair, and the paper reports the
stopping row. No other stopping point is admissible, so the sample cannot end
where the results look best.

## Definitions, fixed as implemented

- Benchmark vintage tau_b: the earliest verifiable public release of the
  benchmark, from Epoch where published, else the benchmark's primary source
  (`data/benchmark_dates.csv`). Same-day ties (tau_b = t_i) are placebo, per
  `eligible = tau_b < t_i` strictly. Private prerelease access is acknowledged
  as a violation this bound cannot see.
- Release date t_i: the release key's date, not a document date. Artifacts are
  coded from release-time snapshots per the source hierarchy in
  `protocol/coding-protocol.md`; updated pages are not evidence.
- One release-level score per benchmark: the maximum across Epoch scaffold
  variants (`build_matrix.py`), spread kept as a diagnostic.
- Drop (ORBIT E): benchmark reported for the family predecessor, absent for the
  focal release, with an independent score present; requires
  `prior_model_reported` and `prior_source_url` (`validate_coding.py`).

## Estimands and estimators, in order

1. theta = Pr(b not in D_i | b in O_i and A_i): the raw omission rate over
   eligible coded cells. Descriptive; reported with provider-clustered
   sign-flip inference and never called selective on its own.
2. Reported-minus-omitted standing among eligible cells, on the windowed
   percentile AND the side-balanced percentile, always shown against the
   +12.23 / +8.69 placebo null, never against zero. Conditional on window
   composition only under the stated pre-determination assumption
   (Appendix B), reported as an assumption, not a finding.
3. The within-family drop gap, against the null calibration (sd 19.4 at one
   drop). At most 37 qualifying pairs exist; no single drop is interpreted.

## Inference, fixed

Provider-clustered throughout; below twelve clusters, randomization or
restricted wild bootstrap only (`stats.py`, draws=9999, Webb below twelve for
the bootstrap). Two-way provider-and-benchmark dependence reported for
cell-level regressions. Ambiguity rule: when torn between two ORBIT
categories, the lower-suspicion one, flagged for second coding
(`protocol/coding-protocol.md`).

## Reliability, fixed

Second coding is by artifact, not by row: an independent coder receives 20% of
coded releases, drawn stratified by provider, with the first coder's
`reported_slugs`, notes and derived labels withheld. Cells are derived from
each coder's artifact record by the same `derive_coding` rules, and agreement
is computed on the derived cells: Cohen's kappa on the full category set, on
reported-versus-not, and on E-versus-not-E, plus a confusion matrix.
Reliability is computed on pre-adjudication labels; adjudicated labels form a
separate analysis column. Reliability is reported next to every disclosure
estimate, whatever it is.

## Outcomes

Four endings are admissible and none is preferred: evidence consistent with
selective disclosure; no detectable selective disclosure; estimates too
imprecise to separate selection from innocent non-rerunning; reliability too
low to support an estimate.
