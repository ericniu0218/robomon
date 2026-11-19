#!/usr/bin/env python3
"""Compare whole development episodes with LeRobot select_action + always-render env.step.

This uses LeRobot's control path with explicit matched noise, not its vectorized CLI evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from robomon.envs.libero_session import LiberoSession
from robomon.policy.smolvla import CHECKPOINT_REVISION, VLM_REVISION, SmolVLARunner
from robomon.rollout import EpisodeSpec, derived_seed, run_episode
from robomon.splits import SUITES
from robomon.utils.io import runtime_fingerprint, source_hash, stage_dir, write_json, write_manifest


def reference_episode(runner, session, spec):
    from lerobot.envs.utils import prepare_canonical_observation
    from lerobot.sim.integrations.libero import canonical_observation
    from lerobot.utils.constants import ACTION

    runner.policy.reset()
    obs = session.reset(spec.init_state_id)
    session._set_rendering(True)
    actions, states, success = [], [], False
    started = time.perf_counter()
    for step in range(session.max_steps):
        canonical = canonical_observation(obs)
        batch = prepare_canonical_observation({key: np.asarray(value)[None] for key, value in canonical.items()})
        batch["task"] = [session.task_description]
        batch = runner.pre(runner.env_pre(batch))
        noise = None
        if step % runner.chunk_size == 0:
            generator = torch.Generator().manual_seed(derived_seed(spec.seed, "execution", step))
            noise = torch.randn((1, runner.chunk_size, int(runner.cfg.max_action_dim)), generator=generator)
        with torch.no_grad():
            action = runner.policy.select_action(batch, noise=noise)
            action = runner.env_post({ACTION: runner.post(action)})[ACTION][0].cpu().numpy()
        obs, _, done, truncated, info = session.env.step(action.astype(np.float64))
        actions.append(action.copy())
        states.append(session.get_state())
        success = bool(info["is_success"])
        if success or done or truncated:
            break
    session._set_rendering(False)
    return np.asarray(actions), np.asarray(states), success, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--render-mode", choices=["always", "scheduled"], default="always")
    args = parser.parse_args()
    config = {"stage": "runner_validation", "schema": 1, "args": vars(args),
              "vlm_revision": VLM_REVISION, "runtime": runtime_fingerprint(),
              "checkpoint_revision": CHECKPOINT_REVISION, "source_hash": source_hash()}
    output = stage_dir("runner_validation", config)
    write_manifest(output, "runner_validation", config)
    print(f"OUTPUT {output}", flush=True)
    policy, comparisons = SmolVLARunner(threads=args.threads, device=args.device), []
    for suite in args.suites:
        session = LiberoSession(suite, 0, render_mode=args.render_mode)
        try:
            for initial in range(args.episodes):
                spec = EpisodeSpec(suite, 0, initial, derived_seed(20261110, f"validation/{suite}/{initial}"))
                ours = run_episode(policy, session, spec, monitor=False)
                actions, states, success, seconds = reference_episode(policy, session, spec)
                same_length = actions.shape == ours.arrays["actions"].shape
                action_diff = float(np.max(np.abs(actions - ours.arrays["actions"]))) if same_length else None
                state_diff = float(np.max(np.abs(states - ours.arrays["states"]))) if same_length else None
                passed = (same_length and success == ours.summary["success"]
                          and action_diff < 1e-5 and state_diff < 1e-5)
                row = {**asdict(spec), "passed": passed, "success": success,
                       "ours_success": ours.summary["success"], "reference_steps": len(actions),
                       "ours_steps": ours.summary["steps"], "max_action_difference": action_diff,
                       "max_state_difference": state_diff, "reference_seconds": seconds,
                       "ours_seconds": ours.summary["wall_seconds"]}
                row["render_mode"] = args.render_mode
                comparisons.append(row)
                write_json({"comparisons": comparisons, "all_passed": all(r["passed"] for r in comparisons),
                            "complete": len(comparisons) == len(args.suites) * args.episodes},
                           output / "results.json")
                print(row, flush=True)
                if not passed:
                    raise RuntimeError("Runner equivalence gate failed; see results.json")
        finally:
            session.close()


if __name__ == "__main__":
    main()
