"""Complete paired-scene validation, shared by development and committed held-out entry points."""

import json
from pathlib import Path

import numpy as np

from robomon.rollout import EpisodeSpec
from robomon.utils.artifacts import completed_episode
from robomon.utils.io import config_hash, file_sha256


def validate_pairs(
    run: Path, *, allowed_tasks: list[str], stage: str, require_initial: bool = True
) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if config.get("stage") != stage or config_hash(config) != manifest["config_hash"]:
        raise ValueError("Wrong rollout stage or damaged manifest")
    results = json.loads((run / "results.json").read_text())
    if results["status"] != "complete" or results["completed"] != len(
        config["episodes"]
    ):
        raise ValueError("Incomplete cohort")
    if config["args"].get("reset_mode") != "isolated":
        raise ValueError("Scientific scene validation requires isolated episode resets")
    pairs, hashes = {}, {}
    for row in config["episodes"]:
        spec = EpisodeSpec(**row)
        if spec.task not in allowed_tasks:
            raise ValueError("Outcome access outside the allowed task partition")
        directory = run / "episodes" / config_hash(row)
        if not completed_episode(directory, config):
            raise ValueError("Incomplete episode")
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("reset_context") != {
            "mode": "isolated",
            "prior_init_states": [],
        }:
            raise ValueError("Nonisolated reset context")
        hashes[spec.episode_id] = file_sha256(directory / "complete.json")
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            group = pairs.setdefault(spec.pair_id, {})
            if spec.condition in group:
                raise ValueError("Repeated source pair/condition")
            group[spec.condition] = {
                key: arrays[key]
                for key in ("reset_state", "reset_body_pos", "reset_body_quat")
                if key in arrays
            }
    comparisons = []
    for key, group in sorted(pairs.items()):
        if set(group) != {"nominal", "shift"}:
            raise ValueError("Missing paired condition")
        comparison = {"pair_id": key}
        for field in ("reset_body_pos", "reset_body_quat", "reset_state"):
            if field not in group["nominal"] or field not in group["shift"]:
                if require_initial or field != "reset_state":
                    raise ValueError(f"Missing required scene snapshot: {field}")
                comparison[field] = None
            else:
                comparison[field] = bool(
                    np.array_equal(group["nominal"][field], group["shift"][field])
                )
                if not comparison[field]:
                    raise ValueError(f"Nominal/shift scene differs: {key} {field}")
        comparisons.append(comparison)
    if not comparisons:
        raise ValueError("Empty cohort")
    return {
        "passed": True,
        "pairs": len(comparisons),
        "initial_physics_required": require_initial,
        "comparisons": comparisons,
        "episode_completion_hashes": hashes,
    }
