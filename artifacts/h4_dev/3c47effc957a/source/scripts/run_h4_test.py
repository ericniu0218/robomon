#!/usr/bin/env python3
"""Frozen held-out H4 execution; all original-tail gates precede every intervention arm."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from robomon.branching import run_branch
from robomon.envs.libero_session import LiberoSession
from robomon.interventions import (
    STATE_COLUMNS,
    boundary_histories,
    match_alarm_states,
)
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
    SmolVLARunner,
)
from robomon.rollout import EpisodeSpec, derived_seed
from robomon.study import load_frozen_protocol, task_keys
from robomon.utils.artifacts import (
    completed_array_result,
    completed_episode,
    save_array_result,
)
from robomon.utils.io import (
    config_hash,
    file_sha256,
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
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.protocol, args.receipt = (
        str(Path(args.protocol).resolve()),
        str(Path(args.receipt).resolve()),
    )
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    allowed = task_keys("test")
    if (args.replicates, args.seed, args.threads) != (
        protocol["h4"]["replicates"],
        protocol["h4"]["seed"],
        4,
    ):
        raise ValueError("H4 settings differ from the frozen protocol")
    manifest = json.loads((args.run / "manifest.json").read_text())
    trunk = manifest["config"]
    if (
        trunk.get("stage") != "test_rollouts"
        or config_hash(trunk) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified held-out rollout manifest")
    if (
        trunk["protocol_sha"] != protocol["_protocol_sha256"]
        or trunk["episodes"] != protocol["test_episodes"]
    ):
        raise ValueError("Unexpected held-out trunk cohort/protocol")
    for key, value in {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
    }.items():
        if trunk.get(key) != value:
            raise ValueError(f"Incompatible {key}")
    runtime = runtime_fingerprint()
    if (
        runtime["python"] != trunk["runtime"]["python"]
        or runtime["platform"] != trunk["runtime"]["platform"]
    ):
        raise ValueError("Restore the trunk runtime before branching")
    for key, value in trunk["runtime"]["packages"].items():
        if runtime["packages"].get(key) != value:
            raise ValueError(f"Trunk runtime package mismatch: {key}")
    if trunk["args"]["render_mode"] != "always":
        raise ValueError("Counterfactual replay requires always-render trunks")
    selection_manifest = json.loads((args.selection / "manifest.json").read_text())
    selection_config = selection_manifest["config"]
    if (
        selection_config.get("stage") != "h4_selection_test"
        or config_hash(selection_config) != selection_manifest["config_hash"]
    ):
        raise ValueError("Expected a verified held-out state selection")
    if selection_config["frozen_protocol_sha"] != protocol["_protocol_sha256"]:
        raise ValueError("Selection is not from this frozen protocol")
    if selection_config["rollout_config"] != trunk:
        raise ValueError("Selection and trunk rollouts disagree")
    if (
        hashlib.sha256(selection_config["protocol_text"].encode()).hexdigest()
        != selection_config["protocol_sha"]
    ):
        raise ValueError("Selection method hash mismatch")
    analysis = Path(selection_config["args"]["analysis"])
    for name, key in [
        ("manifest.json", "analysis_manifest_sha"),
        ("scored_landmarks.parquet", "landmarks_sha"),
    ]:
        if file_sha256(analysis / name) != selection_config[key]:
            raise ValueError(f"Selection input changed: {name}")
    rows = pd.read_parquet(
        analysis / "scored_landmarks.parquet",
        columns=STATE_COLUMNS + [protocol["h4"]["signal"]],
    )
    boundaries = boundary_histories(
        rows,
        signal=protocol["h4"]["signal"],
        reduction=protocol["h4"]["reduction"],
        allowed_tasks=allowed,
    )
    alarm = protocol["h4"]["alarm"]
    selection = match_alarm_states(
        boundaries,
        alarm["threshold"],
        max_pairs_per_cell=protocol["h4"]["pairs_per_cell"],
        seed=protocol["h4"]["seed"],
        allowed_tasks=allowed,
    )
    stored = json.loads((args.selection / "results.json").read_text())
    if (
        stored != {"scope": selection_config["scope"], "alarm": alarm, **selection}
        or alarm != selection_config["alarm"]
    ):
        raise ValueError("Selected states do not reproduce the outcome-blind algorithm")
    # Empty matching is an inconclusive H4 cohort, never a reason to choose substitute states.
    specs = {
        EpisodeSpec(**row).episode_id: EpisodeSpec(**row) for row in trunk["episodes"]
    }
    if any(spec.task not in allowed for spec in specs.values()):
        raise ValueError("Held-out branching cannot inspect development outcomes")
    histories, previous_task, history = {}, None, []
    for episode in trunk["episodes"]:
        spec = EpisodeSpec(**episode)
        if spec.task != previous_task:
            history = []
        histories[spec.episode_id] = history.copy()
        history.append(spec.init_state_id)
        previous_task = spec.task
    jobs = []
    for match in selection["matches"]:
        for role in ("alarm", "control"):
            state = match[role]
            spec = specs[state["episode_id"]]
            trunk_directory = args.run / "episodes" / config_hash(asdict(spec))
            if not completed_episode(trunk_directory, trunk):
                raise ValueError("Selected state has no complete trunk")
            saved_summary = json.loads((trunk_directory / "summary.json").read_text())
            context = saved_summary.get("reset_context")
            if context != {"mode": "isolated", "prior_init_states": []}:
                raise ValueError(
                    "Held-out branches require recorded isolated reset context"
                )
            reset_history = (
                histories[spec.episode_id]
                if context is None
                else context["prior_init_states"]
            )
            for replicate in range(-1, args.replicates):
                for rewind in (0,) if replicate == -1 else (0, 50):
                    jobs.append(
                        {
                            "match_id": match["match_id"],
                            "role": role,
                            **state,
                            "kind": "gate" if replicate == -1 else "intervention",
                            "replicate": replicate,
                            "rewind": rewind,
                            "branch_seed": specs[state["episode_id"]].seed
                            if replicate == -1
                            else derived_seed(
                                args.seed, f"h4/{match['match_id']}/{replicate}"
                            ),
                            "reset_history": reset_history,
                            "reset_history_source": "recorded episode context"
                            if context is not None
                            else "legacy declared-order candidate; requires exact gate",
                        }
                    )
    # Certify the entire selected cohort before opening any intervention arms. A later failed
    # gate must not leave an outcome-conditioned subset that could be mistaken for the pilot.
    jobs = [job for job in jobs if job["kind"] == "gate"] + [
        job for job in jobs if job["kind"] == "intervention"
    ]
    config = {
        "stage": "h4_test",
        "scope": "frozen held-out fixed selected-state Monte Carlo experiment",
        "schema": 1,
        "frozen_protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "source_hash": source_hash(),
        "runtime": runtime,
        "args": {**vars(args), "selection": str(args.selection), "run": str(args.run)},
        "selection_manifest_sha": file_sha256(args.selection / "manifest.json"),
        "selection_results_sha": file_sha256(args.selection / "results.json"),
        "trunk_manifest_sha": file_sha256(args.run / "manifest.json"),
        "trunk_config": trunk,
        "jobs": jobs,
        "budget": "both arms get original_horizon-checkpoint forward actions",
        "seeds": "one independent seed per matched-set/replicate, common across both states and both arms",
        "rewind": 50,
        "pilot_protocol": selection_config["protocol_text"],
        "pilot_protocol_sha": selection_config["protocol_sha"],
    }
    output = stage_dir("h4_test", config)
    print(f"OUTPUT {output}", flush=True)
    summaries = []

    def progress(status):
        write_json(
            {
                "scope": config["scope"],
                "status": status,
                "config_hash": config_hash(config),
                "matched_sets": len(selection["matches"]),
                "planned": len(jobs),
                "completed": len(summaries),
                "planned_interventions": sum(
                    job["kind"] == "intervention" for job in jobs
                ),
                "completed_interventions": sum(
                    row["kind"] == "intervention" for row in summaries
                ),
                "records": summaries,
            },
            output / "results.json",
        )

    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_manifest(output, "h4_test", config)
        policy, session, current_task = None, None, None
        current_episode, arrays, trunk_summary = None, None, None
        try:
            for index, job in enumerate(jobs):
                directory = output / "branches" / config_hash(job)
                if completed_array_result(directory, config):
                    summary = json.loads((directory / "summary.json").read_text())
                    if any(summary.get(key) != value for key, value in job.items()):
                        raise ValueError("Completed branch identity mismatch")
                    print(f"RESUME {index + 1}/{len(jobs)}", flush=True)
                else:
                    spec = specs[job["episode_id"]]
                    if job["episode_id"] != current_episode:
                        trunk_dir = args.run / "episodes" / config_hash(asdict(spec))
                        if not completed_episode(trunk_dir, trunk):
                            raise ValueError("Missing complete trunk")
                        trunk_summary = json.loads(
                            (trunk_dir / "summary.json").read_text()
                        )
                        with np.load(
                            trunk_dir / "arrays.npz", allow_pickle=False
                        ) as saved:
                            arrays = {
                                key: saved[key]
                                for key in (
                                    "actions",
                                    "states",
                                    "embeddings",
                                    "query_steps",
                                )
                            }
                            reset_model = {
                                name: saved[f"reset_{name}"]
                                for name in ("body_pos", "body_quat")
                                if f"reset_{name}" in saved
                            }
                            reset_physics = saved["reset_state"]
                        if trunk_summary["steps"] <= job["landmark"]:
                            raise ValueError("Selected state is terminal")
                        current_episode = job["episode_id"]
                    if policy is None:
                        policy = SmolVLARunner(
                            threads=args.threads, device=trunk["args"]["device"]
                        )
                    if current_task != spec.task:
                        if session is not None:
                            session.close()
                        session = LiberoSession(
                            spec.suite, spec.task_id, reset_mode="isolated"
                        )
                        current_task = spec.task
                    if session.max_steps != job["horizon"]:
                        raise ValueError(
                            "Declared horizon differs from the real benchmark"
                        )
                    result = run_branch(
                        policy,
                        session,
                        spec,
                        arrays["actions"],
                        arrays["states"],
                        checkpoint=job["landmark"],
                        rewind=job["rewind"],
                        seed=job["branch_seed"],
                        reset_history=tuple(job["reset_history"]),
                        reset_model=reset_model,
                        reset_physics=reset_physics,
                    )
                    checks = {}
                    if job["kind"] == "gate":
                        mask = arrays["query_steps"] == job["landmark"]
                        checks = {
                            "same_actions": np.array_equal(
                                result["actions"], arrays["actions"][job["landmark"] :]
                            ),
                            "same_states": np.array_equal(
                                result["states"], arrays["states"][job["landmark"] :]
                            ),
                            "same_embedding": bool(
                                mask.sum() == 1
                                and np.array_equal(
                                    result["first_embedding"],
                                    arrays["embeddings"][mask][0],
                                )
                            ),
                            "same_success": result["success"]
                            == trunk_summary["success"],
                        }
                        checks["passed"] = all(checks.values())
                    array_keys = ("actions", "states", "first_embedding")
                    summary = {
                        **job,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in array_keys
                        },
                        **checks,
                    }
                    save_array_result(
                        summary,
                        {key: result[key] for key in array_keys},
                        directory,
                        config,
                    )
                    print(
                        f"{index + 1}/{len(jobs)} {spec.task} {spec.condition} {job['role']} {job['kind']} rewind={job['rewind']} success={result['success']} seconds={result['wall_seconds']:.2f}",
                        flush=True,
                    )
                summaries.append(summary)
                progress("running")
                if job["kind"] == "gate" and not summary["passed"]:
                    raise RuntimeError(
                        "Known-answer gate failed; interventions at this state are invalid"
                    )
            progress("complete")
        except Exception:
            progress("failed")
            raise
        finally:
            if session is not None:
                session.close()


if __name__ == "__main__":
    main()
