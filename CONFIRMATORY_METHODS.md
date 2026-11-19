# Prospective core-study methods

This document is frozen together with `frozen_protocol.json` in a local Git commit **before any
held-out rollout**. The post-commit receipt records that SHA and the protocol's content hash.
Historical native-reset development results and the failed/reset-repaired pilot are retained as
development history and are not pooled with this study. H5 is an optional stretch and is excluded
from the core study; no H5 claim will be made.

## Population, execution and provenance

Use the fixed split in `robomon/splits.py`: 12 development and 28 held-out monitor tasks. The policy
was trained on all 40 tasks; holdout concerns monitor selection, not policy training. Use all
initial-state IDs 0–15, nominal and one development-calibrated visual shift, for **896 test episodes**.
Policy seeds are `derived_seed(20261110, suite/task/initial)` and are paired across conditions.
All episode specs are enumerated in the committed protocol; no outcome-based replacement or
additional sampling is allowed. Complete all planned data collection regardless of interim results.

Freeze checkpoint/base-VLM revisions, batch-one embedding definition, MPS backend, native precision,
package versions, relevant installed simulator/controller source hashes and all simulator asset
hashes. Execution is K=1 every 50 actions; passive independent K=8 monitoring is every 25. Separate
seed streams key execution, monitor and pixel noise. Use standard always-render observations.

Recreate the environment core before every episode. Save initial physics, static body positions/
quaternions and actual reset context. Every nominal/shift pair must agree exactly in those physical
inputs. Failure aborts analysis rather than silently excluding a condition. Atomic episode/branch
transactions, checksums, locks and exact source archives support deterministic resume. Never pool
CPU/MPS, legacy/isolated, partial or incompatible data. Any scientific-code change after freeze
requires an explicit documented deviation; it cannot silently redefine a confirmatory result.

## Development selection, made before test outcomes

Require all 384 isolated-reset development episodes, 192 paired-scene validations, shifted success
in 50–60%, overall failure in 20–80%, and evaluable primary comparisons. Select H1 by the largest
development primary concordance point estimate among dispersion/consistency/support × max/mean/last.
Break exact ties lexically by signal then reduction. The same rule selects each signal's descriptive
reduction. No held-out re-selection, score sign reversal or threshold fitting is permitted.

Bank extraction pins the named training-source dataset revision, five uniformly sampled demos per
task and ten evenly spaced frames per demo, with task-specific seed 20261111. Same-task support is
nearest L2-normalized prefix-input embedding distance. The checkpoint's exact historical training
dataset revision is unknown; this is a pinned version of its named source. Test-task bank extraction
is permitted only after the committed freeze and uses no rollout labels.

## H1, H2 and H3

Only nonterminal queries are landmarks. First-query consistency is missing, not zero. Compute causal
max/mean/last histories before selecting common rows where all four raw signals exist. The primary
statistic pools failure/success comparisons within **task, condition and exact landmark**, weighting
each comparable pair equally and ties by one half. Single-outcome strata contribute no pairs and
are counted. No comparable pairs means undefined/inconclusive, not 0.5.

H1 asks whether the single dev-selected signal/reduction exceeds concordance 0.5. Use a two-sided
95% percentile bootstrap with **2,000 draws**, seed 20261111, resampling whole initial-state/policy-seed
units **within task**. Keep both conditions and all landmarks together. Include all original paired
units in the resampling universe, including units with no common landmark; their draws contribute
zero rows rather than vanishing from the sampling population. At least 90% of draws must
be evaluable; otherwise H1 is inconclusive. These intervals are approximate and conditional on the
declared task population, not arbitrary unseen tasks.

H2 is descriptive: retain naive full-episode maxima and all time-only, time+condition and primary
task/condition/time estimates for every signal and reduction, with pointwise intervals and coverage.
Elapsed time must equal 0.5 after time matching whenever comparisons exist. Matching changes risk
sets and pair weights; differences are not a causal decomposition of confounding. Do not select a
new held-out winner or make a superiority claim between signals from overlapping marginal intervals.

H3 uses last-value support, dispersion and consistency, with feature scales frozen from complete
development landmarks. Fit the prespecified **pairwise conditional logistic composite likelihood**
within task/condition/time strata, without an intercept or ridge fallback. The support coefficient
tests positive conditional association beyond these two measured linear uncertainty proxies, not
complete model uncertainty or a causal knowledge mechanism. Use the basic paired-unit bootstrap,
2,000 draws, seed 20261111. The optimizer, rank/separation tolerances and invalid-draw handling are
frozen in code. Unidentifiability, separation, no comparable pairs or >10% invalid draws gives an
inconclusive H3 result. The finite simulation validation is documented in VALIDATION.md.

## H4: fixed selected-state experiment

Use H1's frozen signal/reduction. Calibrate a global threshold at the 75th percentile (`higher`) of
one maximum eligible score per nominal dev episode, regardless of its outcome. Keep that threshold
unchanged on test tasks. This is not a conformal threshold or a false-alarm guarantee.

Eligible states are live multiples of 50 actions, starting at 50, with >=50 forward actions left.
Include intervening shadow queries in the causal score. Only the first eligible crossing per
episode may be an alarm. Match a live control at the same task, condition and timestep which has
**not crossed the threshold at any eligible boundary through that time**; later alarm status is
irrelevant. Use only state IDs and the frozen raw signal during selection, with no outcome labels.

Use the existing SHA-256 priority rules and seed 20261111, greedy matching, a cap of **two matched
sets per task/condition**, and no reuse of any original nominal/shift source pair ID anywhere.
Report all 56 cells, missing alarms/controls and the resulting finite cohort. No substitute states
or extra matches may be selected after branch outcomes are seen. Zero matches is inconclusive.
The maximum is 112 matched sets / 224 states; the actual cohort can be smaller.

Every selected state must pass a known-answer original-tail replay gate before any intervention arm
opens. Require exact initial physics/static model agreement and exact prefix states, then future
actions, physics states, first embedding and original outcome. A failed gate aborts the experiment;
do not drop the state. For isolated trunks, recreate the core before **every branch** and replay
its prefix, with an empty prior reset history. Native-history reconstruction is reserved for
legacy development data. This addresses the fixture-sampler failure without retaining state from
one intervention arm to the next.

Use **three fresh independent seed blocks per matched set**, derived from base 20261111, matched-set
ID and replicate. Each block shares noise across both states and both arms. Compare replan in place
with simulation-only oracle rollback by 50 actions. Both receive `original_horizon - checkpoint`
forward actions and may stop early on success; rollback gets no additional forward-action budget.
Execution/pixel seeds use the logical checkpoint clock. Gates are not intervention replicates.

H4a is control-minus-alarm fresh-replan success; H4b is alarm rollback-minus-replan success. Retain
control rollback-minus-replan as exploratory. Weight matched sets equally. Each block difference is
in [-1,1]; use two-sided 95% Hoeffding bounds with radius `sqrt(2 log(40)/N)`, clipped to [-1,1],
where N is matched sets × three seed replicates. These bound branch-noise Monte Carlo error for
the **fixed selected states**, under independent pseudorandom streams. They do not cover selection
or new-state/task transfer, and reset/replay compute is not physical robot recovery cost. Three
seeds can leave small effects unresolved; do not interpret a wide interval as proof of no benefit.

## Multiplicity, inconclusive results and final reporting

Use a fixed confirmatory sequence: **H1 → H3 → H4a → H4b**. A test supports its positive hypothesis
only when its prespecified two-sided 95% lower bound is strictly above 0.5 (H1) or 0 (others), and
every earlier test has supported its hypothesis. At the first failure or inconclusive test, close
the sequence; later estimates/intervals are descriptive only, even if individually above zero.
Still collect and report the full planned cohort and all prespecified estimates.

The general fixed-sequence procedure is described in the
[FDA's October 2022 statistical appendix, §5, p. 20](https://www.fda.gov/media/162416/download).
The error-control argument is that a false rejection requires rejection of the first true null in
the predetermined order. This controls the family if the individual tests are valid; here bootstrap
coverage is approximate, so report nominal rather than guaranteed finite-sample family-wise control.
Using a directional decision from a two-sided 95% bound is conservative relative to a 5% one-sided
test. H2 and sensitivity descriptions are not extra confirmatory claims.

Report missing/degenerate strata, numerical failures, protocol deviations, all effect directions,
source/input hashes and the freeze SHA. Write the final report and shareable figures, clearly
separating development, held-out prediction and fixed-state intervention evidence. Do not claim
novel detector priority, policy-unseen tasks, causal training-data knowledge or real-robot safety.
