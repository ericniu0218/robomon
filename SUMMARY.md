# RoboMon — one-page core-study summary

**Question:** how much do simple VLA failure signals predict after controlling elapsed time, task and visual condition, and do selected alarms identify states where oracle rollback helps?

**Design:** SmolVLA / LIBERO; 384 isolated-reset development episodes, then a prospectively frozen 896-episode monitor-test set (28 tasks × two conditions × 16 paired initial states/seeds). All 448 paired initial scenes agree exactly. The policy was trained on these tasks; holdout concerns monitor tuning. H2 is descriptive and optional H5 was excluded before testing.

| Frozen hypothesis | Estimate [two-sided 95% interval/bound] | Sequence decision |
|---|---|---|
| H1: mean consistency, primary matched concordance | 0.746 [0.713, 0.777] | supported |
| H3: last support, conditional coefficient per dev SD | 1.415 [1.046, 1.728] | supported |
| H4a: control − alarm fresh-replan success | 0.569 [0.409, 0.729] | supported |
| H4b: alarm rollback − replan success | 0.108 [-0.052, 0.268] | not established |

The elapsed-time baseline has naive episode AUROC 0.995 and exactly 0.500 after time matching. The primary estimate concerns within-task/condition/time association. The support result concerns a conditional working model with last-value dispersion and consistency; it does not establish causal training-data knowledge. H1 and H3 use 2,000 paired-unit bootstrap draws, with approximate coverage.

Outcome-blind frozen matching selected 96 sets / 192 states. All 192 exact original-tail replay gates preceded 1,152 experimental branches. Three independent paired seed blocks share noise across both states and both arms. Both arms have equal remaining forward-action budgets; rollback restores 50 earlier actions. H4's conservative bounds cover Monte Carlo noise only at the fixed selected states. An interval spanning zero does not establish zero benefit.

**Interpretation:** positive hypotheses are established only along H1 → H3 → H4a → H4b until the first failed or inconclusive test. Later intervals are descriptive. Results concern one checkpoint, simulator, task set, pinned bank and visual condition. Oracle rollback is simulation-only; physical recovery costs and real-robot safety were not measured. Historical reset-history cohorts are excluded from these estimates.

**Reproducibility:** freeze `5aca84453a411d9c3b8bc824415a0b95d79edd15` preceded held-out outcomes. The reset audit caught fixture motion outside saved physics; isolated cores and exact gates address it. [Full report and figures](FINAL_REPORT.md), [prospective methods](CONFIRMATORY_METHODS.md), [freeze receipt](protocol_freeze.json).
