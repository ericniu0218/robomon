#!/usr/bin/env python3
"""Run only development tasks; held-out testing is deliberately unavailable until preregistration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robomon.envs.libero_session import LiberoSession
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
    SmolVLARunner,
)
from robomon.rollout import EpisodeSpec, derived_seed, run_episode
from robomon.splits import SUITES, task_split
from robomon.utils.artifacts import completed_episode, save_episode
from robomon.utils.io import (
    config_hash,
    runtime_fingerprint,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument(
        "--task-ids", nargs="+", type=int, help="Optional subset; must all be dev tasks"
    )
    parser.add_argument(
        "--episodes", type=int, default=2, help="Initial state indices 0..N-1"
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=["nominal", "shift"], default=["nominal"]
    )
    parser.add_argument("--brightness", type=float, default=0.6)
    parser.add_argument("--noise-sigma", type=float, default=10.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument(
        "--reset-mode",
        choices=["native", "isolated"],
        default="isolated",
        help="isolated recreates the core before each episode, avoiding accumulated fixture sampler history",
    )
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument(
        "--shared-prefix",
        action="store_true",
        help="Opt-in; validate on the current backend first",
    )
    parser.add_argument(
        "--render-mode",
        choices=["always", "scheduled"],
        default="always",
        help="scheduled is experimental and fails strict macOS equivalence on object task 0",
    )
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50 or args.k < 2:
        parser.error("episodes must be in [1,50], k >= 2")
    split, specs = task_split(), []
    for suite in args.suites:
        tasks = (
            args.task_ids
            if args.task_ids is not None
            else split["suites"][suite]["dev"]
        )
        if not set(tasks) <= set(split["suites"][suite]["dev"]):
            parser.error(
                f"Only dev tasks allowed for {suite}: {split['suites'][suite]['dev']}"
            )
        for task in sorted(set(tasks)):
            for initial in range(args.episodes):
                seed = derived_seed(args.seed, f"{suite}/{task}/{initial}")
                for condition in sorted(set(args.conditions)):
                    specs.append(
                        EpisodeSpec(
                            suite,
                            task,
                            initial,
                            seed,
                            condition,
                            args.brightness if condition == "shift" else 1.0,
                            args.noise_sigma if condition == "shift" else 0.0,
                        )
                    )
    config = {
        "stage": "dev_rollouts",
        "schema": 3,
        "args": vars(args),
        "split": split,
        "embedding_definition": EMBEDDING_DEFINITION,
        "checkpoint": "lerobot/smolvla_libero",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "runtime": runtime_fingerprint(),
        "source_hash": source_hash(),
        "episodes": [asdict(spec) for spec in specs],
    }
    output = stage_dir("dev_rollouts", config)
    print(f"OUTPUT {output}", flush=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_manifest(output, "dev_rollouts", config)
        policy, session, current_task = None, None, None
        summaries = []
        try:
            for index, spec in enumerate(specs):
                directory = output / "episodes" / config_hash(asdict(spec))
                if completed_episode(directory, config):
                    summary = json.loads((directory / "summary.json").read_text())
                    print(f"RESUME {spec.episode_id}", flush=True)
                else:
                    if policy is None:
                        policy = SmolVLARunner(threads=args.threads, device=args.device)
                    if current_task != spec.task:
                        if session is not None:
                            session.close()
                        session = LiberoSession(
                            spec.suite,
                            spec.task_id,
                            render_mode=args.render_mode,
                            reset_mode=args.reset_mode,
                        )
                        current_task = spec.task
                    result = run_episode(
                        policy,
                        session,
                        spec,
                        k=args.k,
                        shared_prefix=args.shared_prefix,
                    )
                    save_episode(result, directory, config)
                    summary = result.summary
                    print(
                        f"{index + 1}/{len(specs)} {spec.episode_id} success={summary['success']} "
                        f"steps={summary['steps']} seconds={summary['wall_seconds']:.1f}",
                        flush=True,
                    )
                summaries.append(summary)
                write_json(
                    {
                        "status": "complete"
                        if len(summaries) == len(specs)
                        else "partial",
                        "config_hash": config_hash(config),
                        "completed": len(summaries),
                        "planned": len(specs),
                        "episodes": summaries,
                    },
                    output / "results.json",
                )
        finally:
            if session is not None:
                session.close()


if __name__ == "__main__":
    main()
