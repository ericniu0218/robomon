"""Known-answer replay gates on completed DEVELOPMENT episodes; optional oracle-rollback probes.

Run: .venv/bin/python diagnostics/validate_branches.py artifacts/dev_rollouts/RUN_HASH
This is branch-engine validation, not an alarm-triggered H4 evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from robomon.branching import run_branch
from robomon.envs.libero_session import LiberoSession
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
    SmolVLARunner,
)
from robomon.rollout import EpisodeSpec, derived_seed
from robomon.splits import SUITES
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
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--intervention-samples", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    trunk = manifest["config"]
    if (
        trunk["stage"] != "dev_rollouts"
        or config_hash(trunk) != manifest["config_hash"]
    ):
        raise ValueError("Expected a checked development manifest")
    for key, value in {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
    }.items():
        if trunk.get(key) != value:
            raise ValueError(f"Incompatible {key}")
    selected = [
        EpisodeSpec(**row)
        for row in trunk["episodes"]
        if row["suite"] in SUITES and row["task_id"] == 0 and row["init_state_id"] == 0
    ]
    config = {
        "stage": "branch_validation",
        "scope": "development engineering only",
        "input_manifest_sha": file_sha256(args.run / "manifest.json"),
        "source_hash": source_hash(),
        "diagnostic_sha": file_sha256(Path(__file__)),
        "checkpoint": args.checkpoint,
        "intervention_samples": args.intervention_samples,
        "budget": "original_horizon-checkpoint",
        "selected": [asdict(spec) for spec in selected],
    }
    output = stage_dir("branch_validation", config)
    write_manifest(output, "branch_validation", config)
    print(f"OUTPUT {output}", flush=True)
    policy = SmolVLARunner(threads=4, device=trunk["args"]["device"])
    comparisons = []
    for spec in selected:
        directory = args.run / "episodes" / config_hash(asdict(spec))
        if not completed_episode(directory, trunk):
            raise ValueError(f"Missing episode: {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        if summary["steps"] <= args.checkpoint:
            comparisons.append(
                {"episode_id": spec.episode_id, "skipped": "checkpoint is terminal"}
            )
            continue
        with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
            actions, states = saved["actions"], saved["states"]
            mask = saved["query_steps"] == args.checkpoint
            if mask.sum() != 1:
                raise ValueError("Missing unique checkpoint observation embedding")
            embedding = saved["embeddings"][mask][0]
        session = LiberoSession(spec.suite, spec.task_id)
        try:
            branch = run_branch(
                policy, session, spec, actions, states, checkpoint=args.checkpoint
            )
            row = {
                "episode_id": spec.episode_id,
                "same_actions": np.array_equal(
                    branch["actions"], actions[args.checkpoint :]
                ),
                "same_states": np.array_equal(
                    branch["states"], states[args.checkpoint :]
                ),
                "same_embedding": np.array_equal(branch["first_embedding"], embedding),
                "same_success": branch["success"] == summary["success"],
                "probes": [],
            }
            row["passed"] = all(
                row[k]
                for k in (
                    "same_actions",
                    "same_states",
                    "same_embedding",
                    "same_success",
                )
            )
            if not row["passed"]:
                comparisons.append(row)
                write_json(
                    {"comparisons": comparisons, "all_passed": False},
                    output / "results.json",
                )
                raise RuntimeError("Known-answer counterfactual replay failed")
            for replicate in range(args.intervention_samples):
                seed = derived_seed(spec.seed, f"branch/{args.checkpoint}/{replicate}")
                for rewind in (0, policy.chunk_size):
                    probe = run_branch(
                        policy,
                        session,
                        spec,
                        actions,
                        states,
                        checkpoint=args.checkpoint,
                        rewind=rewind,
                        seed=seed,
                    )
                    row["probes"].append(
                        {
                            k: probe[k]
                            for k in (
                                "success",
                                "forward_steps",
                                "budget",
                                "restored_step",
                                "rewind",
                                "seed",
                                "wall_seconds",
                            )
                        }
                    )
            comparisons.append(row)
            write_json(
                {
                    "scope": config["scope"],
                    "comparisons": comparisons,
                    "all_passed": all(r.get("passed", True) for r in comparisons),
                    "complete": len(comparisons) == len(selected),
                },
                output / "results.json",
            )
            print({k: v for k, v in row.items() if k != "probes"}, flush=True)
        finally:
            session.close()


if __name__ == "__main__":
    main()
