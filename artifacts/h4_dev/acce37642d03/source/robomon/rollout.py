"""Deterministic development rollouts with passive monitors and separate random streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from robomon.envs.libero_session import perturb_images
from robomon.monitors.signals import consistency, dispersion


def derived_seed(seed: int, stream: str, step: int = 0) -> int:
    payload = f"{seed}/{stream}/{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class EpisodeSpec:
    suite: str
    task_id: int
    init_state_id: int
    seed: int
    condition: str = "nominal"
    brightness: float = 1.0
    noise_sigma: float = 0.0

    def __post_init__(self):
        if self.condition not in ("nominal", "shift"):
            raise ValueError("condition must be nominal or shift")
        if self.condition == "nominal" and (
            self.brightness != 1 or self.noise_sigma != 0
        ):
            raise ValueError("nominal observations must be unshifted")
        if (
            not np.isfinite([self.brightness, self.noise_sigma]).all()
            or min(self.brightness, self.noise_sigma) < 0
        ):
            raise ValueError("shift parameters must be finite and nonnegative")

    @property
    def task(self) -> str:
        return f"{self.suite}/{self.task_id}"

    @property
    def pair_id(self) -> str:
        return f"{self.task}/{self.init_state_id}/{self.seed}"

    @property
    def episode_id(self) -> str:
        return f"{self.pair_id}/{self.condition}"


@dataclass
class Rollout:
    summary: dict
    events: pd.DataFrame
    arrays: dict[str, np.ndarray]


def run_episode(
    policy,
    session,
    spec: EpisodeSpec,
    *,
    monitor: bool = True,
    k: int = 8,
    query_interval: int = 25,
    shared_prefix: bool = False,
) -> Rollout:
    """Execute ordinary K=1 reference-policy chunks; monitoring never chooses executed actions.

    Execution noise, monitor noise, and image noise are keyed by (seed, purpose, environment step).
    Thus adding a shadow query cannot consume randomness destined for a later executed action or
    image. K=1 reference execution also avoids changes from batch-size-dependent low-precision
    numerics. Monitoring draws K independent samples in addition to the execution query.
    """
    if monitor and k < 2:
        raise ValueError("monitoring dispersion requires at least two samples")
    if query_interval <= 0 or policy.chunk_size % query_interval:
        raise ValueError("query_interval must divide the action chunk")
    if monitor and query_interval >= policy.chunk_size:
        raise ValueError("consistency requires overlapping chunks")
    started = time.perf_counter()
    obs = session.reset(spec.init_state_id)
    previous, executing = None, None
    rows, chunks, embeddings, states, actions, query_steps = [], [], [], [], [], []
    success, elapsed, query_seconds = False, 0, 0.0
    for step in range(session.max_steps):
        execute_now = step % policy.chunk_size == 0
        monitor_now = monitor and step % query_interval == 0
        if execute_now or monitor_now:
            if step:
                obs = session.observe()
            shifted = perturb_images(
                obs,
                spec.brightness,
                spec.noise_sigma,
                np.random.default_rng(derived_seed(spec.seed, "pixels", step)),
            )
            if execute_now:
                execution = policy.query(
                    shifted,
                    session.task_description,
                    1,
                    torch.Generator().manual_seed(
                        derived_seed(spec.seed, "execution", step)
                    ),
                    shared_prefix=False,
                )
                executing = execution.exec_actions.copy()
            if monitor_now:
                then = time.perf_counter()
                query = policy.query(
                    shifted,
                    session.task_description,
                    k,
                    torch.Generator().manual_seed(
                        derived_seed(spec.seed, "monitor", step)
                    ),
                    shared_prefix=shared_prefix,
                )
                query_seconds += time.perf_counter() - then
                rows.append(
                    {
                        "episode_id": spec.episode_id,
                        "pair_id": spec.pair_id,
                        "task": spec.task,
                        "condition": spec.condition,
                        "landmark": step,
                        "elapsed": float(step),
                        "dispersion": dispersion(query.chunks_norm, query_interval),
                        "consistency": (
                            consistency(previous, query.chunks_norm, query_interval)
                            if previous is not None
                            else float("nan")
                        ),
                    }
                )
                previous = query.chunks_norm.copy()
                chunks.append(previous)
                embeddings.append(query.embedding.copy())
                query_steps.append(step)
        action = executing[step % policy.chunk_size]
        # Sample at the standard timestamp on the step immediately before the next query.
        next_step = step + 1
        render = next_step % policy.chunk_size == 0 or (
            monitor and next_step % query_interval == 0
        )
        result = session.step(action, render=render and next_step < session.max_steps)
        elapsed += 1
        actions.append(action.copy())
        states.append(session.get_state())
        if result.success:
            success = True
            break
        if result.done and elapsed < session.max_steps:
            raise RuntimeError(
                "Simulator ended before the benchmark horizon without success"
            )
    events = pd.DataFrame(rows)
    if rows:
        events["failure"] = int(not success)
    summary = {
        **asdict(spec),
        "episode_id": spec.episode_id,
        "pair_id": spec.pair_id,
        "task": spec.task,
        "success": success,
        "steps": elapsed,
        "termination": "success" if success else "timeout",
        "monitor": monitor,
        "k": k if monitor else 0,
        "shared_prefix": shared_prefix,
        "query_interval": query_interval,
        "queries": len(rows),
        "monitor_query_seconds": query_seconds,
        "wall_seconds": time.perf_counter() - started,
    }
    return Rollout(
        summary,
        events,
        {
            "actions": np.asarray(actions),
            "states": np.asarray(states),
            "query_steps": np.asarray(query_steps, dtype=int),
            "chunks_norm": np.asarray(chunks),
            "embeddings": np.asarray(embeddings),
        },
    )
