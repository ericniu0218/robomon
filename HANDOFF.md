# RoboMon final handoff — 2026-11-12

**The user's authorized core H1–H4 project is complete. No model jobs are active.**
Start with [FINAL_REPORT.md](FINAL_REPORT.md), [SUMMARY.md](SUMMARY.md) and [STATUS.md](STATUS.md).
H2 is descriptive. Optional H5 was excluded before testing, and is outside this completed scope.
Do not restart data collection or retune the monitor without a new, explicit study request.

Project: `/Users/ericniu/sweprojects/bair-plan/robomon`; Python `.venv/bin/python`; Apple M5 MPS,
16 GB RAM. Branch: **robomon-continuation**. Across coding projects, never add `codex/` to a branch.
No push was requested or performed. Leave the unrelated untracked root `bair-plan/` folder alone.
Local commits record the freeze, receipt and completed results. The freeze is
**`5aca84453a411d9c3b8bc824415a0b95d79edd15`**, preceding all held-out work; receipt commit `50634bf`.

## Results and scope

- H1 mean consistency concordance **0.746 [0.713, 0.777]**, supported after matching task, condition
  and exact time. The naive elapsed clock is 0.995 and becomes exactly 0.500 after time matching.
- H3 last support coefficient **1.415 [1.046, 1.728]** per frozen dev SD, supported within the
  declared conditional working model controlling last dispersion and consistency.
- H4a control-minus-alarm fresh-replan success **0.569 [0.409, 0.729]**, supported.
- H4b alarm rollback-minus-replan success **0.108 [−0.052, 0.268]**, not established. Observed alarm
  success rises from 36/288 to 67/288, but its conservative bound spans zero. Do not claim zero effect.

The frozen confirmatory sequence is H1 → H3 → H4a → H4b. H4 bounds cover only Monte Carlo noise at
fixed selected states. The policy trained on all task identities; monitor tuning is held out.
No causal training-data-knowledge, detector-priority or real-robot safety claim. Oracle rollback is
simulation-only, with equal remaining forward-action budgets and no measured physical recovery cost.

## Complete data and artifacts

All artifact paths below are relative to `robomon/artifacts/`:

- Replacement dev: `dev_rollouts/e93d773ba5f3` (384 episodes); exact scene validation:
  `reset_validation/2effe8e037de` (192 pairs); analysis: `dev_analysis/8b2bc61f4df7`.
- Dev bank: `demo_bank/f5972cdb4670` (600 frames / 60 demos). Dev H3 last/max/mean:
  `h3_dev/185d70231b96`, `f9e5c1d80c10`, `9c1786f143bd`.
- Dev alarm: consistency/mean, threshold **0.45370636311812385**; selection
  `h4_selection_dev/5ed3daa419de`; all 42 replay gates pass in `h4_dev/3c47effc957a`.
- Test bank: `demo_bank_test/e593d73b3061` (1,400 frames / 140 demos / 28 tasks).
- All 896 test episodes: `test_rollouts/a1328f372469`; all 448 paired initial physics/static scenes
  equal exactly. Primary analysis: **`test_analysis/af5b72fce0bb`**. Both bootstraps: 2,000 valid draws.
- Test selection: **`h4_selection_test/710295bc600c`**, 96 sets / 192 states from 55/56 cells.
- H4: **`h4_test/4065b2344981`**, all 192 original-tail gates passed before all 1,152 experimental
  arms. Three paired seed blocks per set, 288 independent blocks. No missing/substituted states.
- H4 analysis: **`h4_analysis_test/fcc34b1175a6`**. All arrays, identities, budgets, gates and paired
  seeds verify; all 1,344 records archive-resume without loading weights before analysis.

## Reproducibility and maintenance

All 30 critical scientific source files still match the committed freeze. Runtime, model/tokenizer
bytes, installed simulator/controller sources and simulator assets are also guarded. Only reporting
and documentation changed after freeze; no inferential procedure or sample changed. The 67 fast
tests, fresh real reset/replay regression and 42 dev gates passed before the freeze. The 192 test
gates subsequently passed exactly. Heavy model jobs were serial throughout.

The source/render/input/report/summary/PNG hashes are in `figures/final_provenance.json`. All three
scientific plots were visually reviewed. `scripts/report_study.py` adds no hypothesis tests and
recreates FINAL_REPORT.md, SUMMARY.md and the figures from the explicit final analysis artifacts.
README.md has the complete entry-point commands. Regenerating a report is safe; never silently
change frozen critical code or the protocol.

Raw arrays, source archives, videos and model weights stay local under `.gitignore`; manifests,
results and final figures are tracked. Preserve the raw directories for later verification.
Completed upstream manifests were finalized **before** downstream analysis. Do not run a normal
resume on those completed stages now: it refreshes manifests and invalidates the closed downstream
hash chain. Use `diagnostics/resume_snapshot.py ... --verify-only` for read-only archive checks.

Historical native-reset data and the failed/repaired pilot remain in DEV_REPORT.md, H4_DEV_REPORT.md
and RESET_AUDIT.md. LIBERO leaked fixture samplers across hard resets, changing static poses outside
qpos/qvel. Replacement and test episodes/branches recreate the core every time and check exact
initial physics/static models and complete original tails. Old and replacement cohorts are not pooled.
No required core work remains. Future extensions need their own declared scope and protocol.
