# RoboMon prospective core-study preregistration

The authoritative prospective specification is [CONFIRMATORY_METHODS.md](CONFIRMATORY_METHODS.md),
embedded verbatim in [frozen_protocol.json](frozen_protocol.json). The protocol becomes frozen only
when it is committed locally and `load_frozen_protocol` verifies the post-commit
[protocol_freeze.json](protocol_freeze.json) receipt, committed protocol/source bytes, executing
source, runtime, model/tokenizer bytes, simulator/controller sources and simulator inputs.
The receipt identifies the freeze commit without a self-referential Git hash. No held-out extraction,
rollout or analysis is permitted before this verification. The final report records the same SHA.

## Scope and population

The core study covers H1–H4; H2 is descriptive. The optional H5 task-substitution stretch is excluded
before testing. This is an independent audit/replication, with no detector novelty, real-robot safety,
policy-unseen-task or causal training-data-knowledge claim. See [LITERATURE.md](LITERATURE.md).

Use the previously fixed 12-dev / 28-test task split and all initial-state IDs 0–15. Monitor-test
tasks were part of policy training. The committed protocol enumerates all **896 test episodes**,
including both conditions and paired policy seeds. Complete every episode regardless of interim
results. No outcome-based substitutions, extra sampling or changed shift strength are allowed.

The model revisions are SmolVLA `a24f1f380461b665158d52acd140152c79e8b36d` and base VLM
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Use native MPS precision, four threads,
K=1 execution every 50 actions and independent K=8 passive monitoring every 25 actions, canonical
batch-one prefix embeddings, always-render observations, and a fresh simulator core per episode.
Initial physics and static body poses must match exactly in all 448 nominal/shift test pairs.

## Development basis and choices

The replacement isolated-reset cohort completed **384 episodes**, with all **192** paired initial
physics/static scenes exactly equal. Nominal success is 131/192 (68.2%), shifted success 102/192
(53.1%), overall failure 39.3%; all calibration and coverage gates pass. The frozen visual condition
is brightness **0.6**, pixel-noise σ=**10** on a 0–255 scale. Historical native-reset cohorts and
pilots are retained separately and are not pooled. See [ISOLATED_DEV_REPORT.md](ISOLATED_DEV_REPORT.md).

Among three signals × three causal history reductions, maximize the development primary point
estimate and break ties lexically. This selects **mean consistency**, development concordance
0.757633. Each signal's separately selected descriptive reduction is also mean. There is no
held-out winner selection or sign reversal. H3's previously declared last-value features remain
unchanged; their development scales are frozen in the protocol.

The H4 threshold is **0.45370636311812385**, the nominal-development episode-max 75th percentile
(`higher`) of mean consistency, using all eligible outcomes. The dev selection has 21 matched sets /
42 states. The protocol generator requires every one of their exact original-tail replay gates to
pass; it checks completed transactions and records their artifact/result hash. The fresh physical
reset/replay regression and the expanded **67-test** fast suite pass before this freeze.

## Tests and multiplicity

H1 compares the selected signal's pair-weighted primary concordance with 0.5, within task, condition
and exact landmark. H2 retains all signals/reductions and naive/time/time+condition/primary audits.
Common landmarks require every raw signal, and histories are causal; first consistency is missing.

H3 fits pairwise conditional logistic composite likelihood with last-value support, dispersion and
consistency, scaled by frozen dev SDs. It tests the positive support coefficient within that working
model. No ridge fallback is permitted; rank, separation and fit failures are explicitly inconclusive.
H1 uses a percentile bootstrap and H3 a basic bootstrap: **2,000** whole paired-unit draws within task,
seed **20261111**, retaining original units with zero common rows. More than 10% invalid draws makes
the relevant result inconclusive.

The confirmatory sequence is **H1 → H3 → H4a → H4b**. Require the two-sided 95% lower bound strictly
above its null, and every earlier hypothesis established. Stop confirmatory claims at the first
failure or inconclusive result; later estimates remain descriptive. Still complete all collection.
Bootstrap coverage and nominal family-wise control are approximate; no finite-sample guarantee is
claimed. Full definitions and the checked statistical source are in CONFIRMATORY_METHODS.md.

## Outcome-blind interventions

Hold the dev threshold fixed. At live 50-action boundaries with at least 50 actions left, match the
first eligible alarm crossing to a control at exactly the same task, condition and timestep that has
never crossed at an earlier eligible boundary. Use only the fixed score and state identity columns,
SHA-256 priorities with seed 20261111, cap **two sets per cell**, and no source-pair reuse across
either condition. Report all 56 cells and missing matches; zero matches is inconclusive.

All selected test states must pass exact original-tail replay gates before any intervention arm.
Recreate the core before every branch; require initial physics/static model, prefix states, future
actions/states, first embedding and terminal outcome to agree exactly for a gate. A failed gate
aborts, without dropping the state.

Run **three independent paired seed blocks** per fixed matched set, sharing noise across both
states and both arms. Compare replan in place with simulation-only oracle rollback by 50 actions,
with equal `horizon - checkpoint` forward budgets. H4a is control-minus-alarm fresh-replan success;
H4b is alarm rollback-minus-replan success. Use conservative two-sided 95% Hoeffding bounds over
independent paired seed-block differences. These cover Monte Carlo noise conditional on selected
states, not state selection, new-task transfer or physical recovery costs.

## Training-source bank and reporting

After freeze only, select five demos × ten frames for each of 28 test tasks, seed 20261111, from
`lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. Retain catalog/video/frame hashes.
Exact historical policy-training bytes are unknown. The same canonical embedding pipeline is used
online and in the bank. No terminal rollout labels enter bank construction or state selection.

Write the complete final report, shareable scientific figures and one-page summary, including all
prespecified effect directions, uncertainty, invalid draws, missing strata/cells, deviations and
source/input provenance. A negative or inconclusive result is a completed study outcome.
