#!/usr/bin/env python3
"""Run exactly the held-out cohort in a checksum-verified, locally committed frozen protocol."""

import argparse
import fcntl
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robomon.envs.libero_session import LiberoSession
from robomon.policy.smolvla import SmolVLARunner
from robomon.rollout import EpisodeSpec, run_episode
from robomon.study import load_frozen_protocol
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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    args.protocol, args.receipt = (
        str(Path(args.protocol).resolve()),
        str(Path(args.receipt).resolve()),
    )
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    specs = [EpisodeSpec(**row) for row in protocol["test_episodes"]]
    execution = protocol["execution"]
    config = {
        "stage": "test_rollouts",
        "schema": 1,
        "args": vars(args).copy(),
        "cli_args": vars(args).copy(),
        "execution": execution,
        "checkpoint_revision": protocol["checkpoint_revision"],
        "vlm_revision": protocol["vlm_revision"],
        "embedding_definition": protocol["embedding_definition"],
        "runtime": runtime_fingerprint(),
        "source_hash": source_hash(),
        "protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "episodes": [asdict(spec) for spec in specs],
    }
    # Scorers share the explicit backend/reset contract with development run manifests.
    config["args"].update(execution)
    output = stage_dir("test_rollouts", config)
    print(f"OUTPUT {output}", flush=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_manifest(output, "test_rollouts", config)
        policy, session, current_task, summaries = None, None, None, []
        try:
            for index, spec in enumerate(specs):
                directory = output / "episodes" / config_hash(asdict(spec))
                if completed_episode(directory, config):
                    summary = json.loads((directory / "summary.json").read_text())
                    print(f"RESUME {spec.episode_id}", flush=True)
                else:
                    if policy is None:
                        policy = SmolVLARunner(
                            threads=execution["threads"], device=execution["device"]
                        )
                    if current_task != spec.task:
                        if session is not None:
                            session.close()
                        session = LiberoSession(
                            spec.suite,
                            spec.task_id,
                            render_mode=execution["render_mode"],
                            reset_mode=execution["reset_mode"],
                        )
                        current_task = spec.task
                    result = run_episode(
                        policy,
                        session,
                        spec,
                        k=execution["k"],
                        shared_prefix=execution["shared_prefix"],
                    )
                    save_episode(result, directory, config)
                    summary = result.summary
                    print(
                        f"{index + 1}/{len(specs)} {spec.episode_id} success={summary['success']} steps={summary['steps']} seconds={summary['wall_seconds']:.1f}",
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
