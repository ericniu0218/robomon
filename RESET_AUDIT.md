# Reset audit — 2026-11-11

The first alarm-triggered pilot exposed a reproducibility error that the earlier task-0 checks
missed. It stopped at its ninth selected-state gate, before running interventions at that state.
The exact replay criterion was retained; the error was investigated rather than tolerated.

## What failed

At `libero_10/8`, initial state 3, nominal condition, checkpoint 350, reset-plus-action replay first
differs from the saved trajectory at action **184**. The initial difference is about 1e-14; by step
350 the trajectories differ materially. Identical initial qpos/qvel and XML text do not guarantee
the same static fixture pose.

The installed `hf-libero==0.1.4` appends `object_property_initializers` every hard reset, then samples
all accumulated entries in `_reset_internal`. Those extra RNG draws occur before fixture placement.
Two resets with the same seed used **3 versus 4 property samplers**, changing the stove's
`model.body_pos` by up to **0.007955 m**. The flat physics state records qpos/qvel, not this model
parameter. The reset sequence used by the original episode had **9 samplers** instead.

A minimal reproduction uses LeRobot directly, without RoboMon's session wrapper or policy weights
(run in the installed environment):

```python
from libero.libero import benchmark
from lerobot.sim.integrations.libero import LiberoEnv

suite = benchmark.get_benchmark_dict()["libero_10"]()
env = LiberoEnv(suite, task_id=8, task_suite_name="libero_10", obs_type="pixels_agent_pos", init_states=True)
try:
    for _ in range(3):
        env.init_state_id = 3
        env.reset(seed=3)
        rs = env._env.env
        stove = rs.sim.model.body_name2id("flat_stove_1_main")
        print(len(rs.object_property_initializers), rs.sim.model.body_pos[stove])
finally:
    env.close()
```

Observed sampler counts are 3, 4, 5. Stove xy positions are respectively
`(-0.204430, -0.196475)`, `(-0.196475, -0.198183)`, `(-0.198183, -0.209520)`.
The initial physics-vector SHA-256 is identical in all three cases:
`07d8d0f4b29dfef1d799d7f79e1eb5fc26f7d9128df761bef055673052f413bb`.

The complete negative diagnostic is `artifacts/replay_determinism/598d5d93c346/`. It saves model
arrays and state/control/warmstart traces. The failed initial pilot is
`artifacts/h4_dev/acce37642d03/` (8 gates passed, 48 intervention branches, status `failed`). Its
partial outcomes are not analyzed as a completed H4 experiment.

The replacement threshold, all 19 selected alarm/control pairs, and coverage counts stayed
byte-identical to the original selection (results SHA-256
`dc6c6c46dbeddc4b18dbca28b1f985e31a2f45292870f895e9ca7abe7f8576cc`). No state was replaced
because of a branch outcome or replay failure.

## Repair and evidence

- For legacy counterfactuals, reconstruct the candidate original same-task reset sequence. At the
  failing case that sequence is `[0, 0, 1, 1, 2, 2]` before resetting to state 3. Save its pre-reset
  sampler list and restore that list before each subsequent branch reset. The ordinary native
  reset, settle and action paths then run unchanged.
- Two fresh-session reconstructions reproduce all 350 prefix states exactly (`3327cde05b10`).
  Three cached-context attempts reproduce the states, control/warmstart traces and static model
  arrays with identical NPZ hashes (`15a48f355dbf`).
- Every selected-state gate must also match the original **future** policy actions, physics states,
  first embedding and outcome. Declared episode order is only a candidate for legacy reset history:
  interrupted/resumed historical runs may have a different actual history, so acceptance still
  requires the exact gate. All cohort gates now precede intervention branches.
  The replacement pilot passed all **38 selected-state gates** before opening its arms.
- Future `run_dev.py` runs default to **isolated cores per episode**, using the native first-reset
  path. They record actual reset context, static body positions/quaternions and initial physics.
  Branches reject a stored static-model mismatch even if qpos/qvel agree.
- The real-simulator regression passes for repeated reconstructed fixture poses and isolated
  initial physics/camera equality. Eight isolated long-task smoke rollouts completed; all four
  nominal/shift pairs share identical static poses (`ce0a2bc09b96`). Their static-scene validation
  is `artifacts/reset_validation/321b66579466/`. This first smoke predates the new initial-physics
  array, so that validator explicitly reports those four snapshots as unavailable.

## Consequence for the research study

The historical 384-episode audit remains a development dataset. Its nominal/shift success difference
is descriptive and cannot isolate visual corruption where fixture poses differed. The old analysis
must not be pooled with an isolated-reset replacement. Signal/threshold choices remain provisional
until full development calibration is repeated under the final reset regime. No held-out robotics
outcomes have been opened.

The pinned training-source bank need not be rebuilt solely for this environment change: its input
frames and policy embedding definition are unchanged.

```bash
# From robomon/. Run model jobs serially on this 16 GB machine.
.venv/bin/python -m pytest -q -m slow tests/test_reset_context.py
HF_HUB_OFFLINE=1 .venv/bin/python diagnostics/replay_determinism.py artifacts/dev_rollouts/135f2c14e328 --episode-id libero_10/8/3/679937630992262150/nominal --attempts 3 --reconstructed-context

# Required replacement calibration before the held-out freeze (new content-addressed output).
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev.py --episodes 16 --conditions nominal shift --device mps --shared-prefix --reset-mode isolated
.venv/bin/python diagnostics/validate_reset_pairs.py artifacts/dev_rollouts/NEW_HASH --require-initial-physics
```

## Replacement calibration and branch certification

The complete isolated-reset replacement `dev_rollouts/e93d773ba5f3` now has all 384 episodes.
Every one of its 192 nominal/shift pairs has exactly equal initial physics, static body positions
and quaternions (`reset_validation/2effe8e037de`). Shifted success is 102/192 (53.1%); the full
calibration requirement is satisfied. [ISOLATED_DEV_REPORT.md](ISOLATED_DEV_REPORT.md) records the
replacement choices. Every one of the 42 selected development states passes exact original-tail
replay with a fresh core (`h4_dev/3c47effc957a`); archived resume verifies all 42 transactions.
The held-out protocol requires fresh cores before every episode and every branch, plus the same
initial-physics/static-scene and full original-tail gate. Native-history caching remains limited
to historical data. No old and replacement cohorts are pooled.
