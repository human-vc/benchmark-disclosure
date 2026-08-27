# The general result, stated formally

Drafted for the camera-ready or archival version; not yet woven into the
manuscript. Every claim below is already demonstrated empirically in the paper;
this states what the demonstrations instantiate.

**Setting.** A release ships at date `t`; an instrument carries vintage `tau`.
Availability is `A = 1[t > tau]`. Standing `P` is any statistic of a comparison
set `C` whose membership is constructed from calendar time: a window around
`t`, the pool evaluated by date, or the observed cells of a panel whose
observation depends on `A`.

**Proposition.** If the distribution of `P` given `(t, tau)` depends on
`t - tau`, then `A` cannot satisfy the exclusion restriction that availability
not enter the equation for standing, because `A` is a deterministic function of
`t - tau`. No reweighting can restore it: `Pr(A = 1 | t, tau)` is zero or one
at every point, so the positivity condition of inverse-propensity methods
fails identically rather than approximately.

*Proof.* `A = 1[t - tau > 0]` is measurable with respect to `t - tau`. Any
dependence of `P` on `t - tau` therefore induces dependence on `A` except on a
set where `t - tau` is degenerate. Positivity: the propensity is an indicator,
never interior. QED.

**Corollary 1 (equating).** Any method that links scores across instrument
subsets by assuming a conditional relationship common to the two availability
groups assumes away the dependence above, since the groups are defined by
`t - tau`'s sign. (The NEAT-design assumptions of Sinharay and Holland are of
this form.)

**Corollary 2 (what survives).** Under continuity of `E[P | t - tau]` at zero,
the jump in standing at `t - tau = 0` is identified (Hahn, Todd and Van der
Klaauw). In the panel it is indistinguishable from zero while the global
contrast is +12.23, so the global contrast is extrapolation over `t - tau`,
not an effect of availability.

**Scope.** The proposition covers any dynamic, pool-relative evaluation metric
whose comparison set is coupled to instrument availability: leaderboard win
rates over a growing pool, within-benchmark percentiles against
contemporaries, and jointly estimated scales whose observed cells exist only
where `A = 1` (the appendix's ability model, which removes the window and
still inherits the dependence through cell existence).
