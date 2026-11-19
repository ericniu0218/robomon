"""Find reset/model/state discrepancies without relaxing the branch replay acceptance criterion.

Replays saved actions only: no model weights, no intervention outcomes, no replacement states.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from robomon.data.demos import dev_task_keys
from robomon.envs.libero_session import LiberoSession
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


def difference(left, right):
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    changed = ~((left == right) | (np.isnan(left) & np.isnan(right)))
    finite = np.isfinite(left) & np.isfinite(right)
    return (
        None
        if not changed.any()
        else {
            "max_abs": float(np.max(np.abs(left[finite] - right[finite])))
            if finite.any()
            else None,
            "nonfinite_mismatches": int((changed & ~finite).sum()),
            "elements": int(changed.sum()),
            "first_index": list(map(int, np.argwhere(changed)[0])),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--declared-reset-history", action="store_true")
    parser.add_argument("--reconstructed-context", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    trunk = manifest["config"]
    if (
        trunk.get("stage") != "dev_rollouts"
        or config_hash(trunk) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified development run")
    matches = [
        EpisodeSpec(**row)
        for row in trunk["episodes"]
        if EpisodeSpec(**row).episode_id == args.episode_id
    ]
    if len(matches) != 1 or matches[0].task not in dev_task_keys():
        raise ValueError("One development episode required")
    spec = matches[0]
    prior_resets = []
    for row in trunk["episodes"]:
        candidate = EpisodeSpec(**row)
        if candidate.episode_id == spec.episode_id:
            break
        if candidate.task == spec.task:
            prior_resets.append(candidate.init_state_id)
    directory = args.run / "episodes" / config_hash(asdict(spec))
    if not completed_episode(directory, trunk):
        raise ValueError("Incomplete trunk")
    with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
        actions, expected = saved["actions"], saved["states"]
    if not 0 < args.steps <= len(actions) or args.attempts < 2:
        raise ValueError("Need >=2 attempts and a prefix inside the recorded episode")
    config = {
        "stage": "replay_determinism",
        "scope": "development engineering; not H4 inference",
        "args": {**vars(args), "run": str(args.run)},
        "source_hash": source_hash(),
        "diagnostic_sha": file_sha256(Path(__file__)),
        "trunk_manifest_sha": file_sha256(args.run / "manifest.json"),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    output = stage_dir("replay_determinism", config)
    write_manifest(output, "replay_determinism", config)
    print(f"OUTPUT {output}", flush=True)
    baseline_model, baseline_traces, baseline_reset = None, None, None
    attempts = []
    session = LiberoSession(spec.suite, spec.task_id)
    try:
        for attempt in range(args.attempts):
            if args.declared_reset_history:
                session.close()
                session = LiberoSession(spec.suite, spec.task_id)
                for initial in prior_resets:
                    session.reset(initial)
            if args.reconstructed_context:
                session.reset_for_replay(spec.init_state_id, tuple(prior_resets))
            else:
                session.reset(spec.init_state_id)
            model = session.rs.sim.model._model
            fields = {}
            for name in dir(model):
                if name.startswith("_"):
                    continue
                value = getattr(model, name)
                if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                    fields[name] = value.copy()
            reset_state = session.get_state()
            traces = {
                key: []
                for key in (
                    "state",
                    "ctrl",
                    "qacc_warmstart",
                    "qacc",
                    "qfrc_constraint",
                )
            }
            for step, action in enumerate(actions[: args.steps]):
                session.step(action, render=(step + 1) % 25 == 0)
                traces["state"].append(session.get_state())
                for name in traces.keys() - {"state"}:
                    traces[name].append(
                        np.asarray(getattr(session.rs.sim.data, name)).copy()
                    )
            traces = {key: np.asarray(value) for key, value in traces.items()}
            np.savez_compressed(
                output / f"attempt_{attempt}.npz",
                reset=reset_state,
                **traces,
                **{f"model__{key}": value for key, value in fields.items()},
            )
            differences = (
                {}
                if baseline_model is None
                else {
                    key: changed
                    for key, value in fields.items()
                    if (changed := difference(value, baseline_model[key])) is not None
                }
            )
            trace_differences = (
                {}
                if baseline_traces is None
                else {
                    key: changed
                    for key, value in traces.items()
                    if (changed := difference(value, baseline_traces[key])) is not None
                }
            )
            row = {
                "attempt": attempt,
                "object_property_initializers": len(
                    session.rs.object_property_initializers
                ),
                "body_names": [model.body(i).name for i in range(model.nbody)],
                "replayed_prior_resets": prior_resets
                if args.declared_reset_history or args.reconstructed_context
                else None,
                "xml_sha": hashlib.sha256(
                    session.rs.model.get_xml().encode()
                ).hexdigest(),
                "reset_state_vs_first_attempt": None
                if baseline_reset is None
                else difference(reset_state, baseline_reset),
                "compiled_model_differences_vs_first_attempt": differences,
                "traces_vs_first_attempt": trace_differences,
                "states_vs_recorded_trunk": difference(
                    traces["state"], expected[: args.steps]
                ),
                "arrays_sha": file_sha256(output / f"attempt_{attempt}.npz"),
            }
            if baseline_model is None:
                baseline_model, baseline_traces, baseline_reset = (
                    fields,
                    traces,
                    reset_state,
                )
            attempts.append(row)
            write_json(
                {
                    "scope": config["scope"],
                    "attempts": attempts,
                    "complete": len(attempts) == args.attempts,
                },
                output / "results.json",
            )
            print(row, flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
