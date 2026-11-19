# RoboMon

Completed evaluation audit for **What do VLA failure monitors know beyond elapsed time and task identity?**
Read the [final report and figures](FINAL_REPORT.md) or [one-page summary](SUMMARY.md).
The core H1–H4 study is complete: 384 replacement dev episodes, **896 held-out episodes**,
**192 exact replay gates** and **1,152 intervention branches**. H2 is descriptive; optional H5 was
excluded before testing. [STATUS.md](STATUS.md) records all final artifact IDs and integrity checks.

The frozen sequence supports matched prediction, conditional support association and selected
alarm-state risk. Oracle rollback improves the observed alarm-state branch rate from 12.5% to
23.3%, but its prespecified difference bound [−0.052, 0.268] spans zero: benefit is not established.
These are results on the declared policy/simulator/tasks and fixed selected states, with no real-robot
safety or detector novelty claim. The policy trained on the monitor-test task identities.

Protocol freeze: `5aca84453a411d9c3b8bc824415a0b95d79edd15`, verified by
[the separate receipt](protocol_freeze.json) before held-out work. [Prospective methods](CONFIRMATORY_METHODS.md)
and [preregistration](preregistration.md) define the analysis. The historical
[research proposal](../projects/robotics_plan.md) is superseded by that executable specification.

[RESET_AUDIT.md](RESET_AUDIT.md) documents the native fixture-reset bug and repair;
[ISOLATED_DEV_REPORT.md](ISOLATED_DEV_REPORT.md) records its full replacement calibration.
[DEV_REPORT.md](DEV_REPORT.md) and [H4_DEV_REPORT.md](H4_DEV_REPORT.md) preserve historical cohorts;
none are pooled with the final isolated-reset study. Raw data/source archives remain local;
checked manifests/results, reports and scientific figures are tracked.

## Setup

```bash
cd robomon
uv venv --python 3.12 .venv
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --python .venv/bin/python -r requirements.txt
printf 'N\n' | .venv/bin/python -c 'import libero.libero'
```

The CMake setting is required by the old EGL probe packages on modern CMake; see
[CMake's documentation](https://cmake.org/cmake/help/latest/envvar/CMAKE_POLICY_VERSION_MINIMUM.html).
LIBERO is installed explicitly because LeRobot's extra applies a Linux platform marker.
`requirements-macos.lock` records the complete environment used for the macOS development checks.

On Linux, install `libosmesa6` and `libegl1`. The session defaults to OSMesa there. On macOS, leave
`MUJOCO_GL` and `PYOPENGL_PLATFORM` unset and allow robosuite to use its native context. Policy
inference defaults to CPU. The first real run downloads public model weights and simulator assets.
After downloads are complete, `HF_HUB_OFFLINE=1` avoids repeated Hub requests. CPU is the default;
the validated MPS option is described below.

## Validation and development runs

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m slow tests/test_policy_env.py
.venv/bin/python scripts/validate_runner.py

# Small, resumable development pilot: four task-0 episodes, nominal and shifted (8 episodes).
.venv/bin/python scripts/run_dev.py --task-ids 0 --episodes 1 --conditions nominal shift

# Shared-prefix sampling is opt-in: first pass the real tests on the current backend.
.venv/bin/python scripts/run_dev.py --task-ids 0 --episodes 1 --conditions nominal shift --shared-prefix
```

`validate_runner.py` compares full episodes with LeRobot's `select_action` queue and always-render
`LiberoEnv.step`, using identical initial states and explicit action noise. It does not invoke the
vectorized `lerobot-eval` CLI. `run_dev.py` rejects held-out task IDs. The new test entry points
require a frozen protocol, its local Git commit and receipt, matching scientific source, runtime,
model bytes and simulator input fingerprints before they can execute held-out work.

The split reserves task 0 in each suite for engineering checks (spatial 0 was already exposed in
the original handoff), then chooses two more dev tasks per suite with seed 20261110:

| Suite | Dev IDs | Remaining test IDs |
|---|---|---|
| spatial | 0, 6, 7 | 1, 2, 3, 4, 5, 8, 9 |
| object | 0, 2, 3 | 1, 4, 5, 6, 7, 8, 9 |
| goal | 0, 1, 5 | 2, 3, 4, 6, 7, 8, 9 |
| long (`libero_10`) | 0, 6, 8 | 1, 2, 3, 4, 5, 7, 9 |

## Important contracts

- **Camera timing:** standard always-render is the default. Call `session.step(action, render=True)` immediately before a policy or
  monitor query; `observe()` returns that step's cached observation. A fresh render after physics
  has advanced is not the same observation as LeRobot's cached camera sample. Camera clocks keep
  running on skipped steps. Reading after a skipped step raises rather than returning stale pixels.
  `--render-mode scheduled` enables the optimization, but it is **not certified**: on macOS, object
  task 0 produced a one-level pixel difference at identical physics states, enough to change later
  actions. Two always-render replays of the same actions were identical. Short random-action camera
  tests alone did not detect this. The default avoids this render-history dependence.
  The reproducer and observed output are in `diagnostics/render_history.py` and
  `diagnostics/render_history_observed.txt`.
- **Passive monitoring:** ordinary execution always uses K=1 reference-policy samples. Additional
  K-sample monitor queries use a separate seed stream. Image perturbations are keyed by timestep,
  so shadow queries cannot consume future execution or image randomness. This costs an additional
  K=1 query per executed chunk compared with the original compute estimate.
- **Precision:** the checkpoint's native LeRobot precision is preserved. The earlier Linux
  shared-prefix discrepancy has not been diagnosed; macOS equivalence alone does not certify Linux.
  Use the default reference sampler until the target backend passes its checks.
- **Artifacts:** output folders depend on the source bytes, runtime versions, pinned policy/base
  VLM revisions, split, and complete configuration. Each episode contains Parquet events, NPZ
  actions/states/chunks/embeddings, a JSON summary, and a completion marker with SHA-256 checksums.
  Only complete, verified episodes are skipped on resume. Partial episodes are rerun; damaged
  completed episodes raise an error for inspection. A run lock prevents concurrent writers.
- **State restore:** the current array checkpoint represents simulator physics, not all controller,
  observable, clock, and cached-image state. A replay test is necessary but does not yet certify
  the full counterfactual-branching protocol.
- **Reset context:** [RESET_AUDIT.md](RESET_AUDIT.md) documents a native LIBERO sampler leak that
  changes static fixture poses across resets with identical seeds. New `run_dev.py` jobs default
  to `--reset-mode isolated` and save reset history, fixture poses and initial physics. Legacy
  branches must reconstruct native reset history as well as replay actions. Historical nominal/
  shift success differences are descriptive. The completed isolated calibration is in ISOLATED_DEV_REPORT.md.

## Signal and estimator definitions

These operational definitions are audit baselines inspired by the papers in the plan; they are not
claims to reproduce the papers' entire detectors.

| Signal | Exact implementation |
|---|---|
| dispersion | Square root of mean unbiased sample variance across normalized action coordinates and the next 25 actions |
| consistency | RMS difference of the previous and current mean chunks on their aligned 25-step overlap; independent sample indices are not paired |
| support | Nearest same-task training-source demo embedding after L2 normalization, using Euclidean distance; 600 dev vectors extracted and checked |
| elapsed | Environment steps completed before the query |

Embeddings are masked means of the VLM prefix **input** embeddings, not transformer hidden states.
Rollouts save embeddings for later scoring; they do not yet compute the support signal without a
demonstration bank. History reductions (max, mean, last) are causal. First-query consistency is
missing, not zero. Reductions and the winning H1 signal must be selected on development tasks.

`analysis/landmarks.py` pools success/failure pairs inside the specified strata, assigning ties
half credit. The primary strata are `(task, condition, landmark)`; time-only strata are
`(landmark,)`. Strata with only one outcome add no pairs and are counted. If there are no comparable
pairs, the estimate is undefined. The naive comparator takes one maximum per episode.

The bootstrap resamples whole trajectories within tasks and retains their complete landmark
histories. Pass `cluster="pair_id"` to retain nominal/shift pairing by initial state and policy
seed. Confidence intervals are percentile intervals; degenerate resamples and missing scores are
reported explicitly. These are uncertainty estimates conditional on the selected tasks, not
generalization intervals over all possible robot tasks.

The visual shift (brightness 0.6, pixel-noise standard deviation 10 on the 0–255 scale) reached
53.1% shifted success in the full development calibration. The confirmatory protocol is not frozen.

## Training-source support bank and Apple GPU execution

The Apple M5's MPS backend is supported explicitly with `--device mps`. CPU remains the default.
Sampling equivalence at K=1/4/8 and whole-episode comparisons against LeRobot were checked on MPS.
CPU and MPS can produce different native-precision trajectories; each experiment must use one
backend consistently. Support scoring rejects a bank extracted on a different backend.

```bash
# One heavy job at a time on the 16 GB Mac.
.venv/bin/python -m pytest -q -m slow --policy-device mps tests/test_demo_embedding.py
.venv/bin/python scripts/validate_runner.py --device mps
.venv/bin/python scripts/build_demo_bank.py --device mps
.venv/bin/python scripts/run_dev.py --episodes 16 --conditions nominal shift --device mps --shared-prefix

# Use the exact output directories printed by those commands.
.venv/bin/python scripts/analyze_dev.py --rollouts artifacts/dev_rollouts/RUN_HASH --bank artifacts/demo_bank/BANK_HASH
.venv/bin/python scripts/plot_dev.py artifacts/dev_analysis/ANALYSIS_HASH
.venv/bin/python scripts/analyze_h3_dev.py --analysis artifacts/dev_analysis/ANALYSIS_HASH --reduction last
.venv/bin/python diagnostics/validate_branches.py artifacts/dev_rollouts/RUN_HASH
```

The demonstration stage pins `lerobot/libero` revision
`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The policy's saved training config names this repository
but leaves its historical dataset revision unspecified. The bank therefore measures distance to a
**pinned version of the named training source**; exact historical training bytes are not established.
The default dev bank samples five demonstrations per task and ten evenly spaced frames per demo
(600 frames from 60 episodes). Selection is outcome-blind and stable when more tasks are added.
All 1,693 episode offsets, frame IDs, timestamps and task assignments are validated before selection;
benchmark task IDs are matched by instruction, not assumed equal to dataset task indices.

LeRobotDataset supplies canonical floating-point CHW images. Bank extraction passes them directly
through the checkpoint processors, without rotating again. Online and demonstration embeddings
both use batch-one prefix encoding (`smolvla.prefix_input_masked_mean.batch1.v1`); reference
K-sample action batches get an extra batch-one embedding pass to prevent batch-size numerics from
changing the support definition. A real integration test verifies identical embeddings for equal
canonical input. Shards record frame IDs and hashes of the actual Parquet/video inputs, and the
complete bank is checksum-verified before use. Partial cache prefetching avoids LeRobot 0.7's
empty-filter download failure.

Development scoring computes all four signals, then causal max/mean/last reductions. Matched
comparisons share landmarks where all four raw signals exist; naive episode maxima retain all
available queries. Paired conditions stay together in the bootstrap, including the naive statistic.
Undefined concordance is reported as JSON `null`. The analysis reports all candidate reductions
and does not automatically declare a winning monitor. Its coverage gate requires the planned
16 episodes per task and condition; an acceptable overall failure rate alone is insufficient.
An additional time-and-condition control separates the visual-shift confound from task identity;
the primary statistic remains time/task/condition matching.

The H3 prototype uses a **pairwise conditional logistic composite likelihood** on failure/success
feature differences inside task/condition/landmark strata. Whole paired episodes are bootstrapped;
Hessian standard errors would treat dependent pairs incorrectly. Singular or separated fits are
reported as inconclusive, and no ridge fallback changes the estimand. This is a development
implementation; the exact H3 protocol and multiplicity rule must be frozen before test analysis.
H3 uses basic-bootstrap intervals after an initial synthetic check found undercoverage with
percentile intervals. In fresh paired-condition simulations, the corrected intervals covered
the null in 63/64 datasets and the planted positive coefficient in 60/64;
the primary concordance estimator's interval is evaluated separately.

The branching prototype restores a checkpoint by resetting and replaying its recorded prefix,
checking every saved physics state exactly. This reconstructs controller, observable and render
history as well as physics. A no-intervention control must reproduce future actions, states,
embeddings and outcome. An oracle rollback arm receives the same remaining number of forward
actions as replanning in place; it receives no extra forward-action budget. Oracle rollback is a
simulation experiment. Its reset/replay compute cost does not measure a physical robot's recovery
cost, and branch-engine tests do not establish the alarm-triggered H4 hypothesis.

Stages now also archive their exact Python source alongside the manifest. A stopped stage can be
resumed from that archive even after unrelated analysis code has been added:

```bash
.venv/bin/python diagnostics/resume_snapshot.py artifacts/dev_rollouts/RUN_HASH --verify-only
.venv/bin/python diagnostics/resume_snapshot.py artifacts/dev_rollouts/RUN_HASH
```

The launcher checks archive, file, source, configuration and runtime hashes, then executes the saved
implementation against its original artifact directory. It retains normal locks and per-episode
checksums. Source archives live beside the raw artifacts locally; copy them with the artifacts when
moving machines. Source/runtime changes normally create a new run; archived resume is the explicit
way to continue an older implementation without relabelling it as current code.

## Alarm-triggered development interventions

[H4_PROTOCOL.md](H4_PROTOCOL.md) declares the pilot before intervention outcomes. Selection uses
causal running-max consistency and a nominal-dev episode-max quantile; terminal labels are not read.
Alarm/control states match task, condition and timestep exactly. Source nominal/shift pairs are
never reused across selected states. Missing alarms/controls are counted, without substitutes.

```bash
.venv/bin/python scripts/select_h4_dev.py --analysis artifacts/dev_analysis/ANALYSIS_HASH
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_h4_dev.py --selection artifacts/h4_selection_dev/SELECTION_HASH --run artifacts/dev_rollouts/RUN_HASH
.venv/bin/python diagnostics/resume_snapshot.py artifacts/h4_dev/H4_HASH --verify-only
.venv/bin/python diagnostics/resume_snapshot.py artifacts/h4_dev/H4_HASH
.venv/bin/python scripts/analyze_h4_dev.py artifacts/h4_dev/H4_HASH
.venv/bin/python scripts/plot_h4_dev.py artifacts/h4_analysis_dev/ANALYSIS_HASH
```

The runner first requires exact original-tail replay at **each** selected state. Three fresh seed
blocks pair alarm/control states and replan/one-chunk rollback arms; all four outcomes in a block
share noise. Both arms have `original_horizon - checkpoint` forward actions. Branch transactions
have independent completion checksums and resume without loading weights when already complete.
All selected-state gates are completed before any intervention arms. Reset-context reconstruction
and the failed initial pilot are recorded in RESET_AUDIT.md; selected pairs and threshold stayed
byte-identical after the engineering repair.

The analysis reports alarm-vs-control replan and alarm rollback-vs-replan success differences.
Conservative Hoeffding bounds describe Monte Carlo error at the **fixed selected states**, not
generalization to new states/tasks. Their independent sample count is matched sets × seed replicates,
not arms or simulator steps. Development pointwise intervals do not confirm H4 or settle the final
multiplicity rule. Oracle rollback remains a simulation-only intervention.

To validate scene pairing in the replacement development run:

```bash
.venv/bin/python diagnostics/validate_reset_pairs.py artifacts/dev_rollouts/NEW_HASH --require-initial-physics
```

This checks every completed episode, both static model poses and initial physics, and rejects
unavailable required snapshots. The earlier eight-episode isolation smoke predates the initial-
physics array; its validator explicitly reports that field as unavailable.

## Frozen core-study entry points

[CONFIRMATORY_METHODS.md](CONFIRMATORY_METHODS.md) declares the prospective H1–H4 methods, including
H2's descriptive role, fixed-sequence decisions and fixed-state H4 bounds. Test entry points reject
missing/uncommitted protocols or changed scientific source, runtime, model/tokenizer bytes,
simulator/controller source, assets, BDDL or initial states. A local freeze commit and the separate
post-commit `protocol_freeze.json` receipt must exist first. No test-task extraction or outcomes
are used by `prepare_protocol.py`.

The completed study used the following entry points serially. Use the final artifact IDs in STATUS.md
when regenerating reports; data collection is already complete. For an explicitly requested fresh
reproduction under the frozen contract, the stage commands are:

```bash
.venv/bin/python scripts/build_demo_bank.py --partition test --device mps --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json"
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_test.py --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json"
.venv/bin/python scripts/analyze_test.py --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json" --rollouts artifacts/test_rollouts/RUN_HASH --bank artifacts/demo_bank_test/BANK_HASH
.venv/bin/python scripts/select_h4_test.py --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json" --analysis artifacts/test_analysis/ANALYSIS_HASH
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_h4_test.py --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json" --selection artifacts/h4_selection_test/SELECTION_HASH --run artifacts/test_rollouts/RUN_HASH
.venv/bin/python scripts/analyze_h4_test.py artifacts/h4_test/H4_HASH --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json" --analysis artifacts/test_analysis/ANALYSIS_HASH
.venv/bin/python scripts/report_study.py --protocol "$PWD/frozen_protocol.json" --receipt "$PWD/protocol_freeze.json" --analysis artifacts/test_analysis/ANALYSIS_HASH --h4 artifacts/h4_analysis_test/H4_ANALYSIS_HASH --selection artifacts/h4_selection_test/SELECTION_HASH
```

Use the exact printed artifact directories. All 896 test episodes are required, including when an
interim hypothesis is not established. H4 selection uses frozen dev calibration and state/score
columns without terminal labels; no alarms or missing controls are replaced after outcomes.
Isolated counterfactuals recreate their simulator core for every branch. Archive resume also
supports `test_rollouts`, `demo_bank_test` and `h4_test`; verify complete resume **before** downstream
analysis because a resume refreshes the input manifest. H5 remains excluded from the core study.
