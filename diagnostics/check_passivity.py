"""Validate saved dev rollouts against fresh no-monitor episodes without repeating K=8 work.

Usage: .venv/bin/python diagnostics/check_passivity.py artifacts/dev_rollouts/RUN_HASH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from robomon.envs.libero_session import LiberoSession
from robomon.policy.smolvla import SmolVLARunner
from robomon.rollout import EpisodeSpec, run_episode
from robomon.utils.artifacts import completed_episode
from robomon.utils.io import (
    config_hash,
    file_sha256,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = manifest["config"]
    if (
        config_hash(config) != manifest["config_hash"]
        or config["source_hash"] != source_hash()
    ):
        raise ValueError("Input config or source differs; do not mix implementations")
    specs = [EpisodeSpec(**spec) for spec in config["episodes"]]
    directories = [
        args.run / "episodes" / config_hash(spec) for spec in config["episodes"]
    ]
    for directory in directories:
        if not completed_episode(directory, config):
            raise ValueError(f"Input run is not complete: {directory}")
    validation_config = {
        "stage": "passivity",
        "input_config_hash": config_hash(config),
        "source_hash": source_hash(),
        "diagnostic_sha": file_sha256(Path(__file__)),
        "completion_hashes": [file_sha256(d / "complete.json") for d in directories],
    }
    output = stage_dir("passivity", validation_config)
    write_manifest(output, "passivity", validation_config)
    print(f"OUTPUT {output}", flush=True)
    policy = SmolVLARunner(
        threads=config["args"]["threads"], device=config["args"].get("device", "cpu")
    )
    comparisons = []
    for spec, directory in zip(specs, directories, strict=True):
        session = LiberoSession(
            spec.suite, spec.task_id, render_mode=config["args"]["render_mode"]
        )
        try:
            without = run_episode(policy, session, spec, monitor=False)
        finally:
            session.close()
        with np.load(directory / "arrays.npz", allow_pickle=False) as monitored:
            same_actions = np.array_equal(
                without.arrays["actions"], monitored["actions"]
            )
            same_states = np.array_equal(without.arrays["states"], monitored["states"])
        summary = json.loads((directory / "summary.json").read_text())
        same_outcome = without.summary["success"] == summary["success"]
        row = {
            "episode_id": spec.episode_id,
            "identical_actions": same_actions,
            "identical_states": same_states,
            "same_success": same_outcome,
            "passed": same_actions and same_states and same_outcome,
            "without_monitor_seconds": without.summary["wall_seconds"],
            "with_monitor_seconds": summary["wall_seconds"],
            "steps": without.summary["steps"],
        }
        comparisons.append(row)
        write_json(
            {
                "comparisons": comparisons,
                "all_passed": all(r["passed"] for r in comparisons),
                "complete": len(comparisons) == len(specs),
            },
            output / "results.json",
        )
        print(row, flush=True)
        if not row["passed"]:
            raise RuntimeError(
                "Monitoring changed the trajectory; inspect the comparison artifacts"
            )


if __name__ == "__main__":
    main()
