# Checked positioning — 2026-11-11

This is a replication/audit project. The original handoff was based largely on abstracts; it is
not sufficient evidence for novelty. The following notes identify specific full-text sections
checked during the continuation. They are not an exhaustive literature review.

- **[SAFE](https://arxiv.org/html/2506.09937v1), §§3–5:** learns failure probes from VLA representations
  and evaluates task transfer. Its supervised feature probe is different from RoboMon's training-free
  distances. RoboMon's task holdout refers to monitor selection; the underlying policy was trained
  on these tasks. Do not claim the unseen-task monitoring problem is new.
- **[RESCUE](https://arxiv.org/html/2610.13516v1), §§III, IV-D/E:** already studies restored-state
  interventions and timeout-length confounding. Its reported clock-only discrimination and
  early-exposure controls directly motivate replication. Its detection metrics include episode
  maxima, an early-exposure restriction and a fixed early cutoff. A counterfactual result requires
  complete replay fidelity and a declared remaining action budget. RoboMon's exact landmark/task
  comparisons differ in implementation, but that does not establish priority. H4 must account for
  recovery costs and cannot infer intervention value from AUROC alone.
- **[VLA-FAIL](https://arxiv.org/html/2606.21386v1), §§3.1–3.3:** uses token-wise last-layer
  Mahalanobis distance and velocity-normalized, smoothed action consistency. RoboMon uses pooled
  prefix-input embeddings and nearest-demo distance, plus RMS disagreement of mean normalized
  action chunks. These are deliberately simple inspired baselines, not implementations of LLMD or ACC.
- **[Flow uncertainty](https://arxiv.org/html/2607.27933v1), Appendices B/F:** includes SmolVLA and
  discusses unsupported late horizons in time-indexed conformal bands. The original claim that
  SmolVLA was an unused model is incorrect. The earlier 0.997 clock-AUROC attribution was not found
  in the checked version and is withdrawn; the verified RESCUE result supports the broader concern.

Open positioning work: verify the original Sentinel/STAC definitions, FIPER and FAIL-Detect in
their full texts before describing this as a reproduction of any named detector. Current README
definitions describe the actual code, and are the claims this project can defend.
