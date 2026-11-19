"""Check isolated nominal/shift scenes, recording missing initial-physics snapshots explicitly."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from robomon.data.demos import dev_task_keys
from robomon.rollout import EpisodeSpec
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
    parser.add_argument("--require-initial-physics", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    trunk = manifest["config"]
    if (
        trunk.get("stage") != "dev_rollouts"
        or config_hash(trunk) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified development rollout manifest")
    if trunk["args"].get("reset_mode") != "isolated":
        raise ValueError("Paired-scene validation requires isolated episode resets")
    results = json.loads((args.run / "results.json").read_text())
    if results["status"] != "complete" or results["completed"] != len(
        trunk["episodes"]
    ):
        raise ValueError("Incomplete rollout cohort")
    pairs, hashes = {}, {}
    for episode in trunk["episodes"]:
        spec = EpisodeSpec(**episode)
        if spec.task not in dev_task_keys():
            raise ValueError("Development validation cannot inspect held-out tasks")
        directory = args.run / "episodes" / config_hash(episode)
        if not completed_episode(directory, trunk):
            raise ValueError("Incomplete episode")
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("reset_context") != {
            "mode": "isolated",
            "prior_init_states": [],
        }:
            raise ValueError("Wrong reset context")
        hashes[spec.episode_id] = file_sha256(directory / "complete.json")
        with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
            pair = pairs.setdefault(spec.pair_id, {})
            if spec.condition in pair:
                raise ValueError("Repeated source pair/condition")
            pair[spec.condition] = {
                name: saved[name]
                for name in ("reset_state", "reset_body_pos", "reset_body_quat")
                if name in saved
            }
    comparisons = []
    for pair_id, conditions in sorted(pairs.items()):
        if set(conditions) != {"nominal", "shift"}:
            raise ValueError("Each source pair must have both conditions")
        left, right = conditions["nominal"], conditions["shift"]
        row = {"pair_id": pair_id}
        for key in ("reset_body_pos", "reset_body_quat", "reset_state"):
            row[key + "_equal"] = (
                bool(np.array_equal(left[key], right[key]))
                if key in left and key in right
                else None
            )
        comparisons.append(row)
    if not comparisons:
        raise ValueError("No paired episodes")
    static_passed = all(
        row["reset_body_pos_equal"] is True and row["reset_body_quat_equal"] is True
        for row in comparisons
    )
    initial_passed = all(row["reset_state_equal"] is True for row in comparisons)
    missing_initial = sum(row["reset_state_equal"] is None for row in comparisons)
    valid = (
        static_passed
        and not any(row["reset_state_equal"] is False for row in comparisons)
        and (initial_passed or not args.require_initial_physics)
    )
    config = {
        "stage": "reset_validation",
        "scope": "development engineering",
        "source_hash": source_hash(),
        "diagnostic_sha": file_sha256(Path(__file__)),
        "args": {**vars(args), "run": str(args.run)},
        "run_manifest_sha": file_sha256(args.run / "manifest.json"),
        "episode_completion_hashes": hashes,
    }
    output = stage_dir("reset_validation", config)
    write_manifest(output, "reset_validation", config)
    report = {
        "complete": True,
        "requested_checks_passed": valid,
        "pairs": len(comparisons),
        "all_static_scenes_equal": static_passed,
        "all_initial_physics_equal": initial_passed if missing_initial == 0 else None,
        "pairs_missing_initial_physics_snapshots": missing_initial,
        "comparisons": comparisons,
    }
    write_json(report, output / "results.json")
    print(f"OUTPUT {output}")
    print({key: value for key, value in report.items() if key != "comparisons"})
    if not valid:
        raise RuntimeError(
            "Paired reset validation failed or required snapshots are unavailable"
        )


if __name__ == "__main__":
    main()
