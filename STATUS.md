# RoboMon status — 2026-11-12 (America/Los_Angeles)

**The core H1–H4 study is complete. No model jobs are running.** H2 is descriptive; optional H5 was
excluded before testing and is outside the completed core scope. Read [FINAL_REPORT.md](FINAL_REPORT.md)
and the [one-page summary](SUMMARY.md). Branch: **robomon-continuation**. Work is committed locally;
nothing was pushed during this continuation.

## Frozen results

| Hypothesis | Estimate [two-sided 95% interval/bound] | Fixed-sequence decision |
|---|---|---|
| H1: mean consistency, primary concordance | 0.746 [0.713, 0.777] | Supported |
| H3: last support, conditional coefficient per dev SD | 1.415 [1.046, 1.728] | Supported |
| H4a: control − alarm fresh-replan success | 0.569 [0.409, 0.729] | Supported |
| H4b: alarm rollback − replan success | 0.108 [−0.052, 0.268] | Not established |

Alarm-state branch success is 36/288 (12.5%) under fresh replanning and 67/288 (23.3%) under oracle
rollback. Controls succeed in 200/288 (69.4%) fresh-replan branches. The rollback point difference is
positive, but the prespecified bound spans zero; this does not establish zero benefit. H4 bounds
cover Monte Carlo noise at the fixed selected states, rather than task/state generalization.

## Completion and integrity

- **384 replacement development episodes**, with all **192** nominal/shift initial physics/static
  scene pairs exactly equal. Nominal success 131/192; shifted 102/192 (53.1%), passing calibration.
- Protocol frozen before test work at **`5aca84453a411d9c3b8bc824415a0b95d79edd15`**; separate receipt
  committed at `50634bf`. The receipt, committed/executing scientific code, runtime, 19 model files,
  7 installed simulator/controller sources and 970 simulator input files verify exactly.
- **896 held-out episodes**, all **448** paired initial scenes exactly equal. Nominal success
  307/448 (68.5%); shifted 301/448 (67.2%). The shift remained frozen; no test retuning occurred.
- **96 matched sets / 192 states**, spanning 55/56 task/condition cells. Every one of the **192**
  exact original-tail replay gates passed before all **1,152** intervention branches. Three paired
  seed blocks per set produce 288 independent Monte Carlo blocks. No states were substituted.
- Archive resume verified/skipped all 384 dev episodes, 896 test episodes and 1,344 test gate/branch
  records without loading weights, before their downstream analyses. Both banks also verify/resume.
- H1 and H3 each have **2,000 / 2,000 valid bootstrap draws**. All 30 frozen scientific source files
  remain unchanged. Reporting edits add no tests or selections. The 67-test fast suite, fresh real
  reset/replay regression and 42 development replay gates passed before freeze.
- Three final figures and the 363-word summary are rendered and reviewed. Input, source, report,
  summary and PNG hashes are in `figures/final_provenance.json`.

## Final artifacts

All paths below are under `artifacts/`:

| Stage | Artifact |
|---|---|
| Replacement dev / analysis | `dev_rollouts/e93d773ba5f3`, `dev_analysis/8b2bc61f4df7` |
| Dev bank / paired scenes | `demo_bank/f5972cdb4670`, `reset_validation/2effe8e037de` |
| Dev selection / exact gates | `h4_selection_dev/5ed3daa419de`, `h4_dev/3c47effc957a` |
| Test training-source bank | `demo_bank_test/e593d73b3061` |
| Test episodes / H1–H3 analysis | `test_rollouts/a1328f372469`, `test_analysis/af5b72fce0bb` |
| Frozen H4 selection | `h4_selection_test/710295bc600c` |
| H4 gates and branches / analysis | `h4_test/4065b2344981`, `h4_analysis_test/fcc34b1175a6` |

Raw arrays, archives, videos and model weights remain local under repository policy. Manifests,
results, scientific figures, methods and reports are tracked. Do not refresh completed upstream
manifests after analysis; use read-only verification if checking this closed provenance chain.

## Scope and history

This is an independent audit/replication on one policy and simulator, not a new detector. The policy
was trained on test-task identities; holdout concerns monitor tuning. H3 is a conditional association
beyond two measured linear proxies, not causal training-data knowledge. Oracle rollback is
simulation-only; physical recovery costs and real-robot safety are not measured.

Historical native-reset cohorts are preserved separately in DEV_REPORT.md and H4_DEV_REPORT.md.
RESET_AUDIT.md documents the fixture-sampler bug; ISOLATED_DEV_REPORT.md records its full replacement
calibration. Historical and isolated cohorts are never pooled. No required core-study work remains.
