# Pre-Submission Referee Report

**Paper**: Absence Is Not Omission: Availability, Vintage, and the Measurement of Selective Benchmark Disclosure
**Authors**: Anonymous (NeurIPS 2026 EconML workshop submission)
**Date**: 2026-08-28
**Review Standard**: Leading Field Journal (top-field; venue context: NeurIPS 2026 Economics for ML workshop)

---

## Overall Assessment

The paper builds the availability-gate design for measuring selective benchmark disclosure — a dated panel of 3,082 release-benchmark pairs plus a hand-coded record of 100 release documents — and demonstrates that the design manufactures its own finding: the coded omitted-versus-unavailable gap (+11.4) is indistinguishable from a label-free artifact predicted before any cell was coded. The contribution reviewers split: the advocate rates it Significant (it forecloses a design the adjacent literature is converging on, with unusually well-defended evidence); the skeptic rates it Insufficient for a field journal (a negative result about a design no published estimate uses, built from imported tools) while conceding it matches the workshop format it is actually written in. The single most critical issue: the paper's central "presses on both sets alike" symmetry claim omits training-data contamination, an innocent explanation that operates on exactly one side of the eligibility boundary in exactly the observed direction — and the abstract carries a p-value (p<10⁻⁴) that is unattainable under the paper's own 2,000-draw permutation convention.

**Contribution Rating**: Significant (advocate) / Insufficient for target journal (skeptic) — the crux is whether foreclosing a not-yet-run design clears a field-journal bar or only a workshop bar.

**Preliminary Recommendation**: Revise before sending to referees.

---

## Coordinator Verification (independent checks run against the text, code, and data)

Every substantive agent claim was independently verified before this report was assembled. Verdicts:

| # | Claim (agents) | Verdict |
|---|---|---|
| V1 | p<10⁻⁴ unattainable with 2,000 draws (A3, A4, A8) | **CONFIRMED** — text says p<10⁻⁴ in abstract.tex:16, intro.tex:13, reach.tex:9; `falsification.permutation_test` uses draws=2000 and computes p = k/draws (returns 0.0), while §2.5 states the (k+1)/(draws+1) convention, flooring at 1/2001 ≈ 5.0×10⁻⁴ |
| V2 | Stale "which no cell in this version carries" in §2.1 (A8) | **CONFIRMED** — data.tex:15, false since Section 5 reports 1,280 coded cells |
| V3 | Stale checklist "its coded sheet is empty" (A3, A8) | **CONFIRMED** — checklist.tex:14, contradicted by reach.tex's 16 drops / +2.9 |
| V4 | supporting.tex "trimming at 12% and side-balancing at 29%" stale (A4) | **CONFIRMED** — supporting.tex:83; regenerated values are 19% (trim) and 25% (side-balancing) |
| V5 | t = −6.7 vs Table 1's −13.529/2.038 = −6.64 (A2) | **CONFIRMED** — supporting.tex:131 |
| V6 | Conclusion "removes the empty-endowment excuse" vs reach's "narrows rather than closes" (A3) | **CONFIRMED** — conclusion.tex:3 |
| V7 | Figure order inversion: fig_correction (line 88) placed before fig_geometry (line 97) in supporting.tex while fig:geometry is cited first (§3.2) | **CONFIRMED** — readers meet "Figure 4" before "Figure 3" |
| V8 | Table 3 "No labels" column population unstated + two different "all-time" estimators (A2, A5) | **CONFIRMED** — all five nulls are computed on the full panel by one code path (`eligible_vs_placebo`), so windowed/side-balanced match Table 2 numerically; but Table 2's rank-all row uses `_midrank_alltime` (+22.75) while Table 3's uses plain rank pct (+21.99). Two estimators share one label. A2's copy-paste hypothesis is wrong in mechanism, right in substance |
| V9 | "Within 1.3 points" is a cross-sample comparison (A6 Q3/Q6) | **CONFIRMED EMPIRICALLY** — label-free null recomputed on the 71 deficit releases is **+14.13**, not +12.23; the matched gap is 11.40 − 14.13 = −2.7, which equals the composition wedge (reported share 23.1% × disclosed-omitted gap; measured eligible-all minus omitted-only = +2.40 within release). The "1.3 points" framing mixes samples; the matched statement is cleaner and should replace it |
| V10 | Composition finding vulnerable to benchmark-identity selection (A3 #3, A6 Required 2) | **RESOLVED FAVORABLY** — run on the coded cells: reported coefficient +7.35 with release FE only, **+9.33 with release + benchmark FEs**, +5.70 with share-newer control, +7.44 with both. The finding survives and strengthens under benchmark FEs; the paper should report this |
| V11 | Sign restrictions described as "measurements" (A3 #1) | **CONFIRMED** — supporting.tex identifies only Π_p and Π_c; the restrictions are on β_p, β_c, β_a, of which β_a is the unidentified parameter itself |
| V12 | APC paragraph renames t_i/τ_b to p/c and reuses s (A4) | **CONFIRMED** — supporting.tex "Bounding the unidentified component" |
| V13 | Twelve-cluster resolution claim (A2, A4) | **CONFIRMED** — 2^11 = 2,048 < 10,000; resolution 0.0001 needs G ≥ 15; "twelve" is the SE-credibility rule reused |
| V14 | zhang2026positional bibliography mismatch (A2) | **REFUTED** — arXiv:2605.23170 fetched: title matches the entry, and the paper does contain the four-flagship-release audit finding the related-work section describes (as a component, not its headline). Citation is legitimate; an optional clarifying clause could note the audit is one part of that paper |
| V15 | fig:geometry caption "nothing was scored" (A3 #5) | **CONFIRMED** — the 476 placebo cells are retroactive scores of pre-benchmark releases; "sparsely scored" is the correct claim |
| V16 | HELM Capabilities control is an arithmetic identity (A3, A6, A8) | **CONFIRMED conceptually** — a mean over bit-identical scores on a fixed suite cannot move; reframe as scope condition |
| V17 | Training-data contamination unnamed (A6 Required 1) | **CONFIRMED as a text gap** — no mention in any file; empirical test requires training-cutoff data not in the repo; minimum fix is naming it and conceding the asymmetry |
| V18 | Abstract conflates the sixteen-spec sweep with the side-balanced measure (A2, A4) | **CONFIRMED** — the sweep runs on the windowed gap |

---

## 1. Central Contribution

### Advocate's Case (Agent 7)

Rating: **Significant**. The paper forecloses a design the adjacent literature is visibly converging on, at scale, with a hand-coded record and verified dates, and demonstrates that it manufactures its own finding. The pairing of the 12.23-point label-free gap with the +11.4 coded gap in Table 3 carries the contribution; the shuffle-null validation of the mechanism (9.8 sd), the null boundary RD exactly where concealment would have to show, the APC bound, and the pre-stated HELM contrast make it unusually well defended. Stops short of Transformative because the positive estimand is never recovered, the surviving composition result is explicitly not identified, and the drop design realises 16 events against a detection threshold ~13× the effect it returns. Insufficient and Incremental are indefensible given that none of the closest cited papers can produce or refute the artifact this paper measures. (Full text preserved in the session record; key deltas: Singh et al. — mechanism isolation with bit-identical evidence and a same-organisation control; Zhang et al. — converts the audit-invited estimator from a worry into a measured artifact; Dye — an empirical verification technology proposed and refuted inside one paper.)

### Skeptic's Case (Agent 8)

Rating: **Insufficient for target journal**. The paper refutes a design that, by its own account, no one has run ("None conditions on availability, the single thing we add") — no published estimate is corrected, no demonstrated user is warned. Every analytical instrument is imported intact (boundary bias, APC bounding, sharp RD, Heckman exclusion); the largest diagnosed component is the ex-ante-obvious "late benchmarks are harder" channel (63.4%); the disclosure theory produces no estimate and no Dye-vs-Verrecchia test; the surviving positive result is an unidentified magnitude attached to a conceded premise. Framing-vs-delivery: the abstract promises a measurement instrument and delivers its obituary; the HELM Capabilities "prediction" is an arithmetic identity; internal inconsistencies (stale no-labels sentences, the p<10⁻⁴ floor violation) read as a spine rewritten around a late-added section. Concedes execution is genuinely high and that the paper is "a strong short-paper contribution and matches the workshop format the manuscript is actually built in." What it would take: a target for the negative result, an informative bound, identification or demotion of the composition result, theory that does work, leading with the identified boundary RD.

### Synthesis

The two agents agree on the facts: execution quality is high (both name the permutation validation, the boundary RD, and the documented broken-blindness release as careful work), the negative result is real and well-defended, the composition finding is the only surviving positive claim, and the paper as built matches its actual venue. The crux of disagreement is a single judgment: whether foreclosing a design the literature is *about to* adopt is a contribution of field-journal weight (advocate) or a warning without a victim (skeptic) — that is the argument the introduction must win. The ratings are Significant (advocate) and Insufficient for target journal (skeptic); they are not averaged. The single change that would most strengthen the contribution, per the skeptic's Part 3: produce a target — a published or working estimate that implicitly conditions on availability and moves under the correction — or reframe as a design note with the identified boundary result promoted. Novelty relative to literature not cited in the paper has not been verified.

---

## 2. Referee Assessment (Agent 6)

**Preliminary recommendation: Revise before sending to referees.** Judged as a workshop submission, comfortably above bar. Judged as a field-journal paper, three first-order and cheap-to-fix defects: (1) training-data contamination is unnamed while §3.1's symmetry sentence depends on its nonexistence; (2) the only positive finding was exempted from the benchmark-composition test the paper itself shows is decisive [coordinator note: now run — it survives, +9.33 with benchmark FEs; report it]; (3) the paper's own identified design (boundary-localized coded contrast) is never run on the paper's own labels.

**Required analyses**: [CRITICAL] 1. Contamination channel: three-way split by training cutoff, and public vs held-out test sets. 2. Benchmark FEs + double-margin null on the composition result [run by coordinator: survives FEs; the doubly-constrained permutation remains open]. 3. A window-free scale that does not build in the confound (the joint scale's difficulty parameters are maturity-selected; refit including placebo cells). 4. Coded record footing: sampling frame for the 108-queue, define "pinned worklist," per-estimate N, decompose the near-equality [coordinator: decomposition computed, see V9], boundary-localized coded contrast with power. 5. The promised scaffold-spread diagnostic (mean/median-over-variants placebo gap) + coverage-endogeneity test (does Epoch score cells the provider reported?).

**Suggested**: [MAJOR] bootstrap the APC bound's confidence limit and address the β_a-restriction circularity; test Verrecchia's cutoff and Dye's possession-prior predictions on the coded cells; provider heterogeneity (open/closed, family position, rival releases); a reversal-count null and margins for the five HELM pairs; Lee-style trimming bounds on the coded contrast.

**Literature gaps**: Jung-Kwon 1988, Shin 2003, Einhorn 2005, Beyer et al. 2010, Board 2009, Jin-Luca-Martin 2021 (disclosure); Manski, Lee 2009, Calonico-Cattaneo-Titiunik 2014 (partial ID / RD); Mitchell et al. 2019 (model cards), Foundation Model Transparency Index, contamination literature (Sainz 2023, Golchin-Surdeanu 2024), Chiang et al. 2024 (ML). Heckman 1979 is invoked decoratively; Ma-Chen 2019 unnecessary.

**Framing**: the general vintage-gate result (any date-defined eligibility gate inherits APC collinearity with any pool-relative outcome) is the paper's ceiling-raising claim and is currently buried in one dense paragraph; it applies to clinical registries, ESG metrics, patent citations, credit-rating coverage. The abstract needs a full rewrite around two sentences of result and one of stakes.

**Fit**: workshop — above bar. Field journals: JAR/JAE with a market outcome; QE/JAE(metrics) with the generalized proposition; AEJ:Micro/RAND with a tested model. Alternative outlets: TMLR (closest to current form), NeurIPS Datasets & Benchmarks (with artifacts released), FAccT.

**Referee questions** (abridged): contamination attribution and the training-cutoff split; the +8.7 with benchmark FEs and a double-margin null; the near-equality decomposition; the single-curve narration vs Table 1's −8.151; the HELM identity; the sampling frame and per-estimate N; the boundary-localized coded contrast and its power.

---

## 3. Unsupported Claims & Identification Integrity (Agent 3)

53 items; the paper's causal-verb sweep is otherwise nearly clean. Top-triaged: [CRITICAL] #25 the impossible p<10⁻⁴ (V1); #24 the undefined 108-release coding queue; #1 sign restrictions called "measurements" (V11); #23 no coding rule, reliability, or sensitivity for benchmark dates (τ_b is treatment, running variable, and hand-collected). [MAJOR] highlights: #2 HELM Capabilities identity (V16); #3 composition finding vs the within-eligible maturity gradient [coordinator: survives, V10]; #5 "nothing was scored" false (V15); #6 "Nothing happens there" narrates an underpowered null (upper bound +8.05); #7 single-curve asserted without an equality-of-slopes test; #8 conclusion "removes" vs reach "narrows" (V6); #9 "the same effect" overstates panel-HELM identity; #15 the impossibility claim wrongly includes the jointly estimated scale under a pool condition it does not satisfy; #26 prepublication access needs the direction-of-bias statement (contamination of the placebo group *shrinks* the artifact, which strengthens the paper's conclusion); #27 evaluator-coverage endogeneity to disclosure; #28 reverse-arrow targeting [partly addressed in current text]; #29 max-over-scaffolds bias; #30 the eight unlocatable documents are not ignorable; #31 same-sample comparison [V9: confirmed, matched null is +14.13]; #33 balance-window selected on non-significance; #36-38 scope the universal negatives to the papers named; #42 abstract reports the composition result as a p-value with no magnitude. Under-confidence: the shuffle-null validation and the HELM Lite deductive result deserve to lead their sections.

---

## 4. Internal Consistency & Cross-Reference Verification (Agent 2)

[CRITICAL] Table 3 "No labels" column: windowed/side-balanced match Table 2's full-panel values exactly while rank-all diverges (+21.99 vs +22.75) — [coordinator, V8: one code path computes all five on the full panel; the divergence is two different all-time estimators (plain rank vs midrank); fix by unifying the estimator and stating the population]. [MAJOR] zhang2026positional characterization vs entry title — [coordinator, V14: REFUTED; entry verified correct via arXiv fetch]. [MAJOR] abstract conflates the sixteen-spec sweep with the side-balanced measure (V18). Minor: t = −6.7 vs −6.64 (V5); 1,280 vs 1,249 basis for the 76.9% figure ambiguous; "45 by 61 clusters" unexplained vs 46/74 panel totals; the twelve-cluster resolution arithmetic (V13); "release" means model-shipping event in §§2-3,5-7 and leaderboard software version in §4. All citation keys resolve; panel arithmetic, HELM figures (verified against TikZ coordinates), κ chain, and the APC derivation all check out.

---

## 5. Mathematics, Equations & Notation (Agent 4)

[CRITICAL] the p<10⁻⁴ floor violation (V1). [MAJOR] supporting.tex's "trimming at 12% and side-balancing at 29%" vs core.tex's 19% (V4 — stale from the previous convention; correct figures 19%/25%); the twelve-vs-fifteen resolution threshold (V13); symbol collisions: the APC paragraph renames (t_i, τ_b, t_i−τ_b) to (p, c, a) and reuses s for β_a while core.tex uses s for the newer share (V12); no formal regression equation is written for Table 1, the RD, or the APC reduced form; "within 1.3 points in every specification" needs its comparator set named (specification-matched, not the 8.59-21.66 range). Minor: G and k undefined at first use; no formula for P_ib (needed for the "exact at double precision" identity claim); the sole display equation is unnumbered. LaTeX math formatting is clean throughout.

---

## 6. Tables, Figures & Documentation (Agent 5)

[MAJOR] Table 1's indented rows read as cumulative when rows 2-4 are independent additions; Table 2 is the only table with no uncertainty measure; Table 3 omits the composition finding it is introduced as carrying, and its "No labels" sourcing is unstated (V8); fig:geometry's time axis is unlabeled with no scale; the fig_correction/fig_geometry blocks are in reverse citation order (V7 — Figure 4 cited before Figure 3). Minor: the third dashed line in fig:one-curve (the 0.5 reference) is uncaptioned; fig:helm's "five" is pairs not hues (four hue groups); fig:correction lacks the error bars its sibling has and says "within-release standing" where everything else says "within-benchmark"; units (percentile points) absent from all table headers; decimal precision differs across tables; Table 3's ±90d row vs the prose's 91-day figure (different widths, unstated); no orphaned floats.

---

## 7. Spelling, Grammar & Style (Agent 1)

Criticals: a comma splice ("that work is theory, this paper is measurement"); Singh et al. treated as singular in §1.1 and plural in §4; digit/spelled-out chaos concentrated in §4 (24/twenty-four, 22/twenty-two within adjacent sentences). Majors include: "the object they need" (singular antecedent); the missing "that" in the Dye verification sentence; "identify the older economic question" (category error); the garden-path "measures presses" and "Whether it was is"; three dangling modifiers (opportunity.tex ×2, supporting.tex ×1); "ranking window" vs the defined "comparison window"; "182 day" missing its hyphen; "unraveling" (American) in an otherwise British-spelled paper; sentence opening with a digit (supporting.tex:109). Patterns: adopt one number rule, one serial-comma rule, one authorial voice (we vs "the paper"), $\kappa$ vs "kappa", one national spelling. Banned discourse markers, tautologies, and "significant" misuse: none found — those categories are clean.

---

## Priority Action Items

**CRITICAL** (must fix — could cause desk rejection or major referee objections):
1. Name training-data contamination and repair §3.1's "presses on both sets alike" sentence; at minimum concede the asymmetric channel and its direction, ideally add the training-cutoff three-way split and public-vs-held-out test-set cut. (A6 R1; V17)
2. Fix the p<10⁻⁴ claim in abstract, intro, and §5: with 2,000 draws under the paper's own convention the attainable value is 5×10⁻⁴; also align `permutation_test` with the stated (k+1)/(draws+1) convention or raise the draw count. (A3 #25, A4 #1, A8; V1)
3. Restate the "within 1.3 points" headline as the matched-sample decomposition: on the 71 deficit releases the label-free null is +14.13, the coded deficit +11.40, and the −2.7 difference equals the composition wedge (23.1% × the disclosed-omitted gap). Recompute Table 3's "No labels" column on the matched sample (or state the population), and unify the two all-time estimators. (A6 Q3/Q6, A2 #1, A5; V8, V9)
4. Delete/replace the two stale false sentences: data.tex §2.1 "which no cell in this version carries" and checklist "its coded sheet is empty." (A8, A3 #48; V2, V3)
5. Define the coding queue and "pinned worklist" in the paper (103 worklist releases + 5 later-vintage; 100 of 108 locatable) and report per-estimate N and coded-vs-uncoded balance. (A3 #24, A6 R4)
6. Report the composition result's survival of benchmark fixed effects (+9.33) and the share control (+7.44) — the stress test both hostile agents demanded now favors the paper and belongs in §5. (A6 R2, A3 #3; V10)
7. Correct "each one a measurement this paper already reports": one restriction is measured, two are assumptions on unidentified parameters. (A3 #1; V11)
8. State the τ_b dating rule and (at minimum) acknowledge that date error attenuates the boundary RD. (A3 #23)

**MAJOR** (should fix — will likely be raised by referees):
9. Reframe HELM Capabilities as an arithmetic contrast/scope condition, not a pre-stated prediction; benchmark the five reversals (margins + simulated pool-growth null). (A3 #2, A6 #5, A8)
10. Fix supporting.tex's stale "trimming at 12% / side-balancing at 29%" → 19% / 25%. (A4; V4)
11. Reconcile the single-curve narration with Table 1's −8.151 (report the slopes-equality test or reframe as assumption). (A6 #4, A3 #7)
12. Conclusion: "removes the empty-endowment excuse" → "narrows." (A3 #8; V6)
13. Swap the fig_geometry/fig_correction blocks to match citation order. (A5; V7)
14. Fix the abstract's sixteen-spec sentence (the sweep is on the windowed gap) and name the comparator for "every specification." (A2 #3, A4 #3; V18)
15. Resolve the s symbol collision and the p/c renaming in the APC paragraph; fix the twelve-vs-fifteen resolution sentence; t = −6.7 → −6.6. (A4, A2; V12, V13, V5)
16. "Nothing was scored" → "sparsely scored" in the geometry caption; "Nothing happens there" → add the +8.05 power qualifier. (A3 #5, #6; V15)
17. Add the direction-of-bias sentence for prepublication access (contamination of the placebo group shrinks the artifact). (A3 #26)
18. Address the remaining §5 caveats: coverage endogeneity test, scaffold-collapse diagnostic, the eight unlocatable documents, max-over-variants sensitivity. (A3 #27-30, A6 R5)
19. Scope the universal negatives ("None conditions on availability" → "None of the four"). (A3 #36-38)
20. Grammar criticals: the comma splice, Singh et al. verb agreement, §4 number formatting. (A1 #1-5)

**MINOR** (polish):
21. Table/figure notes: units, Table 1 row grouping, Table 2 uncertainty column, Table 3 Difference definition and per-row N, the 0.5 reference line, fig:helm "four groups, five pairs," error bars in fig:correction, "within-release" → "within-benchmark."
22. Style patterns: one number rule, serial comma, one voice, $\kappa$ everywhere, British spelling ("unravelling"), the three dangling modifiers, "ranking" → "comparison window," 182-day hyphen.
23. Optional: a clarifying clause that the Zhang et al. audit is a component of a positional-failures paper (the citation itself is verified correct).

