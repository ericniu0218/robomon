# Robotics project: full technical plan

**Title:** *What do VLA failure monitors know beyond elapsed time and task identity? A pre-registered, time- and task-matched audit on SmolVLA / LIBERO*

**Status:** Plan v2 (2026-11-10). Rewritten after a literature check and a feasibility check on the actual machine. Code lives in `robomon/`.
**Budget:** $0, CPU only (4 cores, 16 GB). No GPU is needed for anything below.

---

## 0. In plain words
A robot policy that learned from demonstrations sometimes fails, and it would be useful to have an alarm that goes off before it does. Many recent papers build such alarms ("failure monitors"). This project asks an awkward question about them: **how much of an alarm's apparent skill is real?**

Two shortcuts can make an alarm look good without it knowing anything about the robot's situation:
1. **Elapsed time.** In simulation a failed episode is one that runs out of time, so "this episode has been running for a long time" already predicts failure almost perfectly.
2. **Task identity.** Some tasks are simply harder. An alarm that only recognises "this is the hard task" also scores well.

I run a small open robot model (SmolVLA) on a standard benchmark (LIBERO, 40 tasks), compute several published alarm signals, and score them three ways: the usual way, then comparing only episodes at the same moment in time, then comparing only episodes at the same moment *and* on the same task. I also test whether an alarm is worth acting on, by rewinding the simulator and trying again.

## 1. Why the plan changed (literature check, 2026-11-10)
The original pitch was "do failure monitors transfer to unseen tasks, and can the robot retry when the alarm fires?" Both parts have been published since:

| Paper | What it already did |
|---|---|
| Sentinel / STAC (Agia et al., CoRL 2024) | Action-chunk temporal consistency as a failure signal for diffusion policies |
| FAIL-Detect (Xu et al., 2025) | Failure detection without failure data, conformal thresholds |
| FIPER (NeurIPS 2025) | Embedding out-of-distribution score + action-chunk entropy (ACE) |
| **SAFE (Gu et al., NeurIPS 2025)** | **Detectors trained on seen tasks, tested on unseen tasks**, on LIBERO and SimplerEnv with OpenVLA, π0, π0-FAST |
| Rewind-IL / TIDE (2026) | Training-free chunk-discrepancy detector **plus rewinding to a safe state** |
| Hide-and-Seek (ICML 2026) | Multi-task failure detection with generalisation to unseen tasks |
| VLA-FAIL (2026) | Feature Mahalanobis distance + chunk consistency for flow-matching VLAs |
| **RESCUE (Oct 2026)** | **Counterfactual evaluation of alarms**: restore simulator state, try interventions, on LIBERO with OpenVLA-OFT and π0.5 |

So the project cannot claim a new detector or a new idea. Several 2026 papers do flag an open **evaluation** problem: in simulation every failure is a timeout, and one study found elapsed step count alone reaches AUROC 0.997. That is the gap this project works in. It is an **independent audit and replication**, on a model none of those papers used (SmolVLA), with a stricter scoring rule than I found in any of them (matching on time *and* task). I could only read abstracts and search summaries of these papers, so the positioning must be re-checked against the full texts before any novelty claim is made in an email.

## 2. Feasibility facts (measured on this machine, 2026-11-10)
| Check | Result |
|---|---|
| LeRobot | 0.7.0 installs on Python 3.13 (torch 2.15, transformers 5.22) |
| LIBERO | Works on CPU with software rendering (`MUJOCO_GL=osmesa`, after `apt-get install libosmesa6`). First import asks a question on stdin; answering `N` writes `~/.libero/config.yaml` |
| SmolVLA | `lerobot/smolvla_libero` (official, flow matching, 50-action chunks, 10 flow steps). Needs camera keys renamed (`image`→`camera1`, `image2`→`camera2`). 2/2 successes in a smoke test via `lerobot-eval` |
| Speed | Rendering costs 0.19 s per step, physics alone 0.0175 s. The policy only needs images when it plans, so **rendering on demand** makes an episode take roughly 10–15 s instead of 60 s |
| State restore | LIBERO exposes `get_sim_state` / `set_state`, so counterfactual branching is possible |
| Training data | `lerobot/libero` is public and 1.9 GB, so "distance to the training demonstrations" can be computed |
| PushT diffusion policy | `lerobot/diffusion_pusht` runs (17 s per episode with DDIM-10, about 50% success). Its hub checkpoint needs a manual format migration. Kept as an optional second policy class, not part of the core plan |

## 3. Setup
- **Policy:** `lerobot/smolvla_libero`, unchanged. It executes 50 actions open-loop, then re-plans.
- **Tasks:** LIBERO spatial, object, goal and long (`libero_10`): 40 tasks, fixed initial states.
- **Split by task:** 3 dev tasks and 7 test tasks per suite (seeded). Every design choice is made on dev tasks; test tasks are analysed once. Note that the *policy* was trained on all 40 tasks; "held-out" refers to the monitors' tuning, as in SAFE.
- **Conditions:** `nominal`, and one `visual shift` (image brightness change plus pixel noise, fixed per episode) whose strength is chosen on dev tasks to bring success to about 50–60%. The shift supplies failures on tasks the policy otherwise always solves.
- **Episodes:** 16 per task and condition: 1,280 in total (384 dev, 896 test).
- **Monitoring schedule:** the policy is queried every 25 steps. Every second query is executed; the others are "shadow" queries used only for monitoring. Each query draws K = 8 action chunks.

## 4. Monitor signals (all training-free, each credited)
| Signal | Definition | Source idea |
|---|---|---|
| `dispersion` | Spread of the K sampled chunks over the next 25 actions | ACE / FIPER |
| `consistency` | Distance between the new chunks and the previous query's chunks on their 25-step overlap | STAC / Sentinel, TIDE |
| `support` | Distance from the current observation embedding to the nearest embeddings of the **training demonstrations** of the same task | FAIL-Detect, FIPER, VLA-FAIL (embedding OOD) |
| `elapsed` | Steps so far | baseline |

Each signal's history up to a landmark is reduced to one number (running max, running mean, or last value; chosen per signal on dev and frozen).

## 5. Scoring: three levels of matching
A **landmark** is a query time *t*. At a landmark, only episodes still running are compared, and the label is whether the episode eventually fails.
1. **Naive (episode-level):** one score per episode (max over the episode), AUROC over all episodes. This is the confounded number.
2. **Time-matched:** AUROC computed within each landmark, pooled over landmarks. Elapsed time scores exactly 0.5 here by construction.
3. **Time- and task-matched (primary):** pairs are only compared within the same task, condition and landmark (a stratified concordance). An alarm that only knows which task is hard scores 0.5.

Confidence intervals come from a bootstrap that resamples whole episodes within each task.

## 6. Hypotheses (to be frozen in `robomon/preregistration.md` before any test-task analysis)
| ID | Hypothesis | Type | Test |
|---|---|---|---|
| **H1 (primary)** | On test tasks, the dev-selected best training-free signal predicts eventual failure under time- and task-matching | Confirmatory | Stratified AUROC, 95% CI above 0.5 |
| H2 | The audit: how far each signal's score falls from naive to time-matched to time- and task-matched | Descriptive | All three numbers for every signal, with CIs |
| H3 | Training-data support adds information beyond the policy's own uncertainty | Confirmatory | Conditional logistic regression within strata; coefficient of `support` above 0 controlling for `dispersion` and `consistency` |
| H4 | Alarms are worth acting on: (a) states where the alarm fires have a lower chance of success than matched states where it doesn't; (b) rewinding one chunk and re-planning raises that chance compared with re-planning in place | Confirmatory | Counterfactual branching from restored simulator states, 3 branches per arm |
| H5 | Task substitution: in `libero_goal` failures the robot completes a *different* task from the same scene | Exploratory, stretch | Evaluate the other tasks' goal conditions |

H3 mirrors the NLP project's H5 (training-data statistics add to model confidence). H5 mirrors its main result (wrong answers are familiar ones).

## 7. Engineering
- **Package `robomon/`:** render-on-demand LIBERO session with state save/restore; SmolVLA wrapper that returns K chunks plus an observation embedding; monitor signals; landmark statistics; counterfactual branching.
- **Reproducibility:** fixed seeds per episode; every stage writes Parquet plus a manifest; inputs are located by config hash, never by file time (a bug class found in the NLP project); rollouts are sharded per task and condition so they resume after container restarts.
- **Tests:** render-on-demand equals always-render; restoring a state and replaying actions reproduces the trajectory; the stratified AUROC returns 0.5 for a pure time signal and a pure task signal, and recovers a planted within-stratum effect; signal definitions on hand-built examples.
- **Stack match for the target labs:** PyTorch, LeRobot, SmolVLA, LIBERO / robosuite / MuJoCo, LeRobotDataset.

## 8. Compute budget
| Stage | Size | Estimated CPU time |
|---|---|---|
| Dev pilot (choose shift strength, aggregations) | 12 tasks | ~1.5 h |
| Test rollouts | 28 tasks × 2 conditions × 16 episodes | ~3.5 h |
| Training-demo embeddings | ~50 frames per task-episode sample | ~1 h |
| Counterfactual branching | ~160 states × 2 arms × 3 branches | ~2.5 h |

## 9. Timeline and gates
| Step | Deliverable | Gate |
|---|---|---|
| 1 | Runner with render-on-demand and K-sample inference; equivalence and determinism tests | Success rate with my runner matches `lerobot-eval` on a few tasks |
| 2 | Signals and statistics with tests; dev pilot | Dev failure rate between 20% and 80% overall, otherwise adjust the shift on dev |
| 3 | Freeze `preregistration.md`; run test rollouts | — |
| 4 | H1–H3 on test tasks, figures | If H1 is null, report it: "training-free signals carry no information beyond time and task on this model" is a publishable-quality audit result |
| 5 | H4 branching | State restore must pass its test first |
| 6 | Report and one-page summary | — |

## 10. Threats to validity
| Threat | Mitigation |
|---|---|
| All failures are timeouts (length confound) | Landmark analysis; naive number reported only as the confounded reference |
| Task difficulty confound | Stratification by task and condition |
| Monitoring changes the policy | Shadow queries never change the executed actions; verified against a no-monitor run |
| Shift strength tuned to results | Chosen on dev tasks only, frozen in the pre-registration |
| Few failures per task | Pooled stratified concordance; strata without both outcomes contribute nothing and are counted in the report |
| Simulator restore is inexact | Replay test; branch only at chunk boundaries |
| One model, one benchmark, simulation only | Stated as a limitation; PushT diffusion policy as an optional second policy class |
| Software rendering differs from GPU rendering | The nominal success rate is compared with the published SmolVLA numbers |
