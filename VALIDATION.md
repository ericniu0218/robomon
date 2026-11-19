# Validation record — 2026-11-11–12

These are engineering and synthetic-estimator checks. They are not confirmatory robotics findings.
Earlier sections record pre-test engineering work; the final section records the completed
held-out integrity checks. Scientific results are in FINAL_REPORT.md.

## Runtime and representation

- On Apple M5 MPS, K=1/4/8 shared-prefix samples match the reference sampler within the declared
  tolerances. Equal canonical dataset/online inputs give identical batch-one embeddings.
- Four full development episodes (one per suite) match LeRobot's native MPS control path exactly
  in actions and saved physics states. The long episode times out in both implementations.
  See `artifacts/runner_validation/dc0129cc72c6/results.json`.
- Nominal and visually shifted monitor-on/off MPS checks reproduce complete trajectories exactly.
- The pinned bank contains 600 finite 960-dimensional embeddings: 50 frames for each of 12 dev tasks.
  Same-frame/same-task support distance is exactly zero. Missing-task scoring and damaged outputs
  are rejected. The catalog validates 1,693 episode offsets, instruction IDs and video timestamps.
- Source archives preserve exact launch-time bytes and hashes. The active dev run's archived source
  was verified; the complete bank resumed from its archive without loading a model or re-encoding.

## H3 uncertainty: a detected problem and a development correction

The initial pairwise conditional composite-logistic estimator used percentile cluster-bootstrap
intervals. On 24 synthetic positive-effect datasets, those nominal 95% intervals covered the known
coefficient only 20 times (83.3%). The null covered 23/24. This exposed a finite-sample bias issue
before any held-out H3 analysis.

Basic-bootstrap intervals reflect the bootstrap estimation error about the fitted coefficient.
On the initial data they cover both null and positive effects in 23/24 cases. This retrospective
comparison informed a development change; it is not an independent validation result.

A fresh check then used seeds 24–87, 12 tasks, 16 paired initial-state units per task, two correlated
conditions, three repeated landmarks, and 400 bootstrap replicates per dataset. Each condition has
correct logistic marginals; paired conditions share latent features and an outcome uniform variate.
All 128 datasets fitted successfully. Intervals resample whole condition pairs within each task.

| H3 synthetic target | Mean coefficient | Covered / datasets | Observed coverage | 95% interval for coverage rate |
|---|---:|---:|---:|---:|
| Null coefficient 0 | −0.0042 | 63 / 64 | 98.4% | 91.6–100.0% |
| Positive coefficient 0.8 | 0.8060 | 60 / 64 | 93.8% | 84.8–98.3% |

Fresh artifact: `artifacts/conditional_calibration/0d1429ab6f7a/results.json`.
Initial artifact: `artifacts/conditional_calibration/fa97dab24ace/results.json`.
These finite simulations are compatible with nominal coverage but do not establish coverage for
arbitrary robot data, model misspecification, separation, or poorly informative strata. Inconclusive
fit/replicate handling is part of the estimator. These checks informed the subsequently frozen
H3 protocol in CONFIRMATORY_METHODS.md.

## Primary concordance uncertainty

The separate concordance diagnostic uses 12 tasks and 16 paired units per task. Conditions share
observation noise and outcome randomness; failures have five landmarks and successes three. The
per-unit normal score noise is independent of labels. Known true concordance is 0.5 under the null
and Φ(1/√2) = 0.76025 under the positive shift. Elapsed time must score exactly 0.5 after matching.

An initial 48-dataset check covered the truth 44/48 times for both effects. A larger fresh check used
seeds 48–239 and 1,000 bootstrap replicates per dataset, preserving episode pairs and landmarks.

| Concordance target | Mean estimate | Covered / datasets | Observed coverage | 95% interval for coverage rate |
|---|---:|---:|---:|---:|
| Null 0.5 | 0.4962 | 179 / 192 | 93.2% | 88.7–96.4% |
| Positive 0.76025 | 0.7581 | 182 / 192 | 94.8% | 90.6–97.5% |

Fresh artifact: `artifacts/concordance_calibration/81f00f070047/results.json`.
The primary estimator retains its percentile interval. Coverage is approximate; these finite checks
are consistent with the nominal level within Monte Carlo uncertainty. The last column is a binomial
interval for the coverage simulation, not an interval for a robotics effect.

## Counterfactual gating

Toy tests require a replayed prefix to reproduce the stored trajectory, reject any damaged prefix,
and give oracle rollback the same remaining forward-action budget as replanning in place. Real
LIBERO gates then reproduced future actions, states, embeddings and outcomes exactly at all 15
eligible saved development checkpoints: 8 at step 50 and 7 at step 100. One additional step-100
case was skipped because its episode had ended. Artifacts: `artifacts/branch_validation/a86bbc1ede86/`
and `c856be42b41d/`. These were engineering gates rather than an alarm-triggered H4 evaluation;
the later development pilot is recorded below.

## Reset-history failure caught by the H4 pilot

The initial selected-state pilot `artifacts/h4_dev/acce37642d03/` stopped after eight passing gates
and 48 intervention branches. At the next gate (`libero_10/8`, initial state 3, nominal, checkpoint
350), naive prefix replay first diverged at action 184. The partial run is marked failed and cannot
be passed to the H4 analysis; its outcomes are not a completed H4 result.

Inspection found LIBERO appends `object_property_initializers` across hard resets. Extra samplers
consume RNG draws before fixture placement. The stove's `model.body_pos` changed while the saved
qpos/qvel states were initially identical. Matching a physics vector or XML text alone therefore
does not certify that the static scene is the same. This is separate from model sampling noise.

Replaying the six prior same-task resets `[0, 0, 1, 1, 2, 2]` reconstructs the recorded fixture pose.
Two fresh-session diagnostics reproduced all 350 prefix states exactly (`3327cde05b10`). The cached
reset-context implementation then reproduced the states, control/warmstart traces and static model
arrays in three attempts (`15a48f355dbf`), with identical NPZ hashes in all attempts.

Branches now reconstruct that context and restore its pre-reset sampler list before each replay.
New rollouts save actual reset history and static body positions/quaternions. The branch engine
rejects a static-model mismatch even when the recorded physics vector agrees. Future development
runs default to a fresh core per episode; full isolated-reset calibration remains a freeze gate.
The historical nominal/shift success difference cannot be attributed solely to visual corruption.

The fixed-state H4 uncertainty method and prospective precision arithmetic are declared in
[H4_PROTOCOL.md](H4_PROTOCOL.md). With 19 matched sets and three paired seed blocks, its 95%
Hoeffding radius is 0.360. At the original 160-state / three-seed budget it would be 0.175;
achieving a 0.1 radius at 160 states requires 10 seeds and 3,200 intervention branches. These are
Monte Carlo precision calculations, not statistical power or new-state generalization guarantees.
Artifact: `artifacts/h4_precision/c1d696f57307/`.

The replacement pilot `artifacts/h4_dev/e1139cfc2fc7/` passed **all 38 selected-state gates** before
opening its intervention arms. Selection results (threshold, pairs and coverage) remained
byte-identical to the failed pilot. A separate real-simulator reset regression passes, alongside
51 fast tests. Eight isolated smoke rollouts completed and all four static scene pairs agree;
their first-generation snapshots omit initial physics, which the pair validator explicitly reports.

The replacement pilot subsequently completed all 228 intervention branches. Analysis verifies
content hashes, identities, complete paired seed blocks, budgets, finite arrays and simulator time.
Archived resume skips all 266 records without loading model weights. H4_DEV_REPORT.md contains
the development estimates and their fixed-state Monte Carlo bounds; these are not held-out claims.

## Prospective core-study safeguards

Before the held-out freeze, the expanded fast suite passes **67 tests** (14 real integration tests
excluded by `-m 'not slow'`). New checks cover the exact 896-episode plan, real Git-commit/source
freeze verification, source/model/asset drift rejection, paired initial physics and hidden fixture
changes, and rejection of initial-physics mismatch before any replay action. Nonmonotone causal
scores cannot admit a control that crossed the alarm threshold earlier. Bootstrap sampling retains
original paired units with zero common landmarks.

Synthetic end-to-end held-out plumbing uses toy sessions and a stubbed protocol gate, without
accessing real test-task outcomes. It checks both a nonempty H4 cohort (all gates before all arms,
complete paired seeds and contrasts) and a zero-match cohort (explicitly inconclusive). The final
report reads these frozen analysis decisions and adds no new hypothesis tests. The fresh real isolated
reset/replay regression passes (13.69 s). All **42** selected replacement-development known-answer
gates pass exactly; archive resume verifies/skips all 42 without loading weights. Protocol generation
checks those completed transactions and their connection to the chosen development alarm.


## Completed frozen-study integrity — 2026-11-12

The pre-test freeze `5aca844` and its receipt verify against all 30 executing/committed scientific
source files, runtime, 7 installed simulator/controller source files, 970 simulator inputs and
19 model/tokenizer files. Critical scientific code remained unchanged after freeze. Reporting
changes added tables, summaries and rendering provenance without adding tests or selecting results.

All **896** test episodes complete and archive-resume without loading weights. All **448** nominal /
shift initial physics, static body-position and body-quaternion pairs agree exactly. H1 and H3 have
2,000 valid paired-unit bootstrap draws each, with no invalid or degenerate draws. The primary
analysis retains 7,038 common landmarks, 13,208 comparison pairs and 333/735 informative strata.
The elapsed clock equals exactly 0.5 at every matched level.

The frozen H4 selection covers 55/56 cells, 96 sets and 192 states. All **192** known-answer gates
pass exactly before any experimental arm opens. All **1,152** interventions complete with the
prespecified three shared seed blocks per set and equal remaining forward budgets. Analysis checks
all arrays, content hashes, identities, gates, complete role/arm/seed grids and simulator time.
Archived resume verifies/skips all **1,344** records without loading weights, before final analysis.

FINAL_REPORT.md and the 363-word SUMMARY.md reproduce the frozen sequence decisions. All three
PNG figures were visually reviewed. `figures/final_provenance.json` records and verifies the
protocol, input results/manifests, reporting source, render versions, report, summary and PNG hashes.
No raw data or source archives were deleted. These integrity checks support the declared study;
they do not turn bootstrap approximations into exact population guarantees or establish robot safety.
