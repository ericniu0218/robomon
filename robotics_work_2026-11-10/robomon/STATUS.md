# robomon status (2026-11-10)

Plan: `../projects/robotics_plan.md`.

## Working
- LeRobot 0.7.0, LIBERO (CPU, `MUJOCO_GL=osmesa`) and `lerobot/smolvla_libero` run on a 4-core CPU.
- `robomon/envs/libero_session.py`: LIBERO session with physics-only steps and state save/restore.
- `robomon/policy/smolvla.py`: SmolVLA wrapper returning K sampled action chunks and an embedding.
- State restore test passes (restoring and replaying actions reproduces the simulator state).

## Open issues (found by the integration tests, not yet fixed)
1. **On-demand images differ from the standard pipeline's images.** Physics is identical, but robosuite samples
   cameras part-way through a control step, so an image rendered after the step is slightly "newer".
   Planned fix: keep the camera observables enabled and render during the one step before each policy query.
2. **Shared-prefix sampling does not match the reference path** (chunks differ by up to 0.027; embedding differs).
   Use `shared_prefix=False` until this is resolved.
3. Load the policy before creating a LIBERO environment; the other order segfaults.

## Not started
Monitor signals, landmark statistics, rollouts, pre-registration, analysis.
