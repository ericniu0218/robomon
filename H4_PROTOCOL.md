# H4 development pilot protocol — 2026-11-11

**Development only. The initial revision preceded the first pilot. The reset-context amendment
followed that pilot's failure and preceded the replacement pilot; launch-time protocol text is
retained in each run manifest. This does not freeze or open the held-out study.** The purpose is to exercise alarm selection, exact controls, restoration,
paired branch execution, missing coverage and finite-state analysis end to end.

## Alarm and eligible states

- Use the H1 development candidate: causal running-max consistency, including shadow queries.
- Calibrate one global threshold from the **75th percentile**, with `higher` interpolation, of
  nominal development episodes' maxima over eligible boundaries. Use one maximum per episode,
  regardless of terminal success/failure. This weights episodes equally and is not a false-alarm
  guarantee, a conformal threshold, or an optimization against intervention results.
- Evaluate only live execution boundaries `t = 50, 100, …`, with at least 50 forward actions left
  in the fixed benchmark horizon. Query histories must be complete from step zero in increments
  of 25; only first-query consistency is missing. The score at t uses observations through t.
- Threshold comparison is `score >= threshold`. Each episode supplies only its **first eligible**
  crossing; crossing at a shadow query is acted upon at the next eligible execution boundary.

## Matching and sample size

- Every alarm state requires a live control with `score < threshold` at the **same task, condition
  and timestep**. No nearest-time/task substitute is allowed. Controls may cross the threshold
  later; their future score is not an eligibility criterion.
- No terminal labels enter calibration or matching. Both scripts project only episode/pair/task/
  condition/landmark IDs and consistency out of the development landmark file.
- Deterministically sort candidate alarms by SHA-256 of selection seed 20261111 and episode ID.
  Sort each alarm's controls by a separate hash of seed, alarm ID and control ID. Greedily accept
  the first available control, with **one matched set per task/condition** in this pilot.
- Each original nominal/shift **source pair ID may appear only once in the whole selection**.
  Thus neither episode nor its paired-condition counterpart is reused as alarm/control elsewhere.
  Report all 24 task/condition cells, first alarms, exact-control availability and selected counts.
- Freeze the selected states in a content-addressed selection artifact before branch execution.
  This declared greedy sample is not uniform sampling of every possible alarm state. Missing cells
  limit its scope; no replacement states are chosen after seeing outcomes.

## Branches, random numbers and gates

- **Reset-context amendment (development engineering):** the initial pilot stopped before its
  ninth gate because `libero_10/8` replay diverged at action 184. LIBERO accumulates
  `object_property_initializers` across hard resets. Their RNG draws change the static stove's
  `model.body_pos`, which the saved qpos/qvel vector omits. The failed partial run is retained as
  `artifacts/h4_dev/acce37642d03/`; it is not analyzed as an H4 result.
- Reconstruct the original sequence of prior same-task resets once per selected state. Preserve
  its pre-reset property-sampler list and restore that list before each branch reset. The complete
  native reset, settle and action-prefix paths then run as originally recorded. For legacy trunks,
  declared episode order supplies a **candidate** history, accepted only if the exact gate passes.
  New trunks record actual reset context and static body poses/quaternions explicitly.
- At **every** selected alarm and control state, first run the original-seed, no-rollback control.
  It must exactly reproduce the recorded future actions, physics states, first embedding and
  terminal outcome. A failure aborts the experiment; an arm is never substituted or silently dropped.
  The replacement pilot completes all selected-state gates before opening any intervention arms.
- Restore state by reset plus exact prefix-action replay. Verify every replayed physics state.
  The gate additionally checks the controller/observable/render history through its future trace.
- For each matched set, draw **three** fresh branch seed blocks. Derive each seed from base
  20261111, matched-set ID and replicate index. Use that same seed for both states and both arms
  within a block, and distinct seeds across blocks. This pairs future policy and pixel noise.
- At each state, compare **replan in place** (`rewind=0`) with **oracle rollback by one chunk**
  (`rewind=50`). Each arm receives exactly `benchmark_horizon - t` forward actions, ending early
  on success. Rollback loses 50 actions of physical progress and receives no extra forward budget.
- Pixel and execution seeds use the logical branch clock t, including after rollback. Known-answer
  gates are engineering checks, not extra independent intervention samples.
- Use the trunk's pinned checkpoint, MPS backend, package versions and always-render path. Check
  real session horizons against the declared horizons before execution. Save each branch atomically
  with summary, actions/states/first embedding, completion checksums, source archive and run lock.

## Estimands and uncertainty

The endpoint is success before the equal remaining forward-action budget expires. Weight selected
matched sets equally, with equal numbers of seed replicates. Preserve all four outcomes of each seed
block together:

- **H4a diagnostic:** mean(`control replan success - alarm replan success`). Positive values mean
  the selected alarm states are less salvageable by a fresh replan than their exact controls.
- **H4b diagnostic:** mean(`alarm rollback success - alarm replan success`). Positive values mean
  the simulation-only rollback intervention helps at these selected alarm states.
- Control rollback-minus-replan is retained as an exploratory comparison.

Report counts, rates and the positive/zero/negative paired differences. A block difference is in
[-1, 1]. Independent branch seed blocks give the conservative two-sided Hoeffding radius
`sqrt(2 * log(2 / alpha) / N)` for the mean, clipped to [-1, 1], with `alpha=0.05` **pointwise** in
this development pilot. This follows from the bounded-independent-variables inequality in
[Guntuboyina's STAT 210B notes, Theorem 3.1, p. 13](https://www.stat.berkeley.edu/~aditya/resources/FullNotes210BSpring2018.pdf).
The sample size N is matched sets × replicates, never the number of arms or simulator steps.

These are **Monte Carlo bounds conditional on the fixed selected states**, under independent
pseudorandom seed streams. They do not cover dev selection bias, new initial states, new tasks or
real-robot transfer. Bounds stay wide when all observed paired differences are zero. A bootstrap
over three branches per state could instead spuriously collapse in that case.

Do not use this pilot to confirm H4, claim a deployment policy, or interpret reset/replay compute
as physical recovery time. Before test data, freeze sample-size/selection rules, the target
population, confidence/multiplicity procedures and the remaining H1/H3 protocol. Any change driven
by this pilot must be recorded as development.

The historical development run reused native cores. Its nominal/shift episodes share qpos/qvel
initial-state IDs and policy seeds, but tasks with accumulated property samplers need not share
static fixture poses. Their success-rate difference is therefore descriptive, not an isolated
visual-shift effect. Future `run_dev.py` runs default to isolated cores per episode; development
calibration under that reset regime is required before the held-out freeze. Historical results
are retained and must not be pooled with that replacement calibration.
