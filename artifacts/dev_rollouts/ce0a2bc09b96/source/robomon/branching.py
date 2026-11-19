"""Counterfactual engine using deterministic prefix replay to restore full simulator/controller history.

Replaying a recorded prefix reconstructs clocks, controllers, observables and rendering history,
which the existing 92-float physics state does not capture. Oracle rollback is a simulation-only
intervention; its reset/replay compute time is not a physical robot recovery cost.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from robomon.envs.libero_session import perturb_images
from robomon.rollout import EpisodeSpec, derived_seed


def replay_prefix(
    session,
    spec: EpisodeSpec,
    actions: np.ndarray,
    states: np.ndarray,
    checkpoint: int,
    reset_history: tuple[int, ...] | None = None,
) -> dict:
    if not 0 <= checkpoint < len(actions) or len(actions) != len(states):
        raise ValueError(
            "A branch requires a nonterminal checkpoint inside a recorded episode"
        )
    obs = (
        session.reset(spec.init_state_id)
        if reset_history is None
        else session.reset_for_replay(spec.init_state_id, reset_history)
    )
    for step in range(checkpoint):
        result = session.step(actions[step], render=step + 1 == checkpoint)
        if not np.array_equal(session.get_state(), states[step]):
            raise RuntimeError(
                f"Prefix replay differs at step {step + 1}; counterfactual is invalid"
            )
        if result.success or result.done:
            raise RuntimeError("Replay terminated before the selected checkpoint")
    return session.observe() if checkpoint else obs


def run_branch(
    policy,
    session,
    spec: EpisodeSpec,
    actions: np.ndarray,
    states: np.ndarray,
    *,
    checkpoint: int,
    rewind: int = 0,
    seed: int | None = None,
    reset_history: tuple[int, ...] | None = None,
) -> dict:
    """Compare replan-in-place to oracle rollback with the SAME remaining forward-action budget.

    Both arms start their logical clock at checkpoint and get horizon-checkpoint forward actions.
    A rewind arm reconstructs the earlier physical state, without receiving extra forward actions.
    Equal `seed` values across arms implement common random numbers. `seed=spec.seed, rewind=0`
    is the known-answer control that must reproduce the original tail at chunk-boundary checkpoints.
    """
    if checkpoint % policy.chunk_size or rewind < 0 or rewind % policy.chunk_size:
        raise ValueError(
            "Branching and rollback must occur at complete execution-chunk boundaries"
        )
    if not 0 <= checkpoint < min(len(actions), session.max_steps):
        raise ValueError("Checkpoint is terminal or outside the benchmark horizon")
    if getattr(session, "render_mode", "always") != "always":
        raise ValueError(
            "Counterfactual replay requires the certified always-render path"
        )
    started = time.perf_counter()
    restored = max(0, checkpoint - rewind)
    obs = replay_prefix(session, spec, actions, states, restored, reset_history)
    seed = spec.seed if seed is None else seed
    budget = session.max_steps - checkpoint
    branch_actions, branch_states = [], []
    execution = None
    first_embedding = None
    success = False
    for offset in range(budget):
        logical_step = checkpoint + offset
        if offset % policy.chunk_size == 0:
            if offset:
                obs = session.observe()
            shifted = perturb_images(
                obs,
                spec.brightness,
                spec.noise_sigma,
                np.random.default_rng(derived_seed(seed, "pixels", logical_step)),
            )
            query = policy.query(
                shifted,
                session.task_description,
                1,
                torch.Generator().manual_seed(
                    derived_seed(seed, "execution", logical_step)
                ),
                shared_prefix=False,
            )
            execution = query.exec_actions
            if first_embedding is None:
                first_embedding = query.embedding
        action = execution[offset % policy.chunk_size]
        result = session.step(action, render=(offset + 1) % policy.chunk_size == 0)
        branch_actions.append(action.copy())
        branch_states.append(session.get_state())
        if result.success:
            success = True
            break
        if result.done:
            raise RuntimeError(
                "Unexpected simulator termination during a counterfactual branch"
            )
    return {
        "success": success,
        "actions": np.asarray(branch_actions),
        "states": np.asarray(branch_states),
        "first_embedding": first_embedding,
        "forward_steps": len(branch_actions),
        "budget": budget,
        "checkpoint": checkpoint,
        "restored_step": restored,
        "rewind": rewind,
        "seed": seed,
        "wall_seconds": time.perf_counter() - started,
    }
