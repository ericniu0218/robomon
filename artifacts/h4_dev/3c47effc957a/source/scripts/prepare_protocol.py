#!/usr/bin/env python3
"""Prepare the prospective protocol from a complete isolated-reset DEVELOPMENT calibration.

A separate local Git commit and receipt are still required before any held-out execution.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from robomon.analysis.development import SIGNALS, add_history_reductions
from robomon.interventions import boundary_histories, calibrate_alarm
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
)
from robomon.splits import task_split
from robomon.study import (
    asset_hashes,
    critical_source_hashes,
    dependency_hashes,
    episode_plan,
    model_hashes,
    task_keys,
)
from robomon.study_data import validate_pairs
from robomon.utils.io import (
    ROOT,
    config_hash,
    file_sha256,
    runtime_fingerprint,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.analysis / "manifest.json").read_text())
    config = manifest["config"]
    if (
        config.get("stage") != "dev_analysis"
        or config_hash(config) != manifest["config_hash"]
    ):
        raise ValueError("Only verified development analysis can select the protocol")
    if config["rollout_manifest_sha"] != file_sha256(
        args.rollouts / "manifest.json"
    ) or config["rollout_results_sha"] != file_sha256(args.rollouts / "results.json"):
        raise ValueError("Development inputs changed")
    report = json.loads((args.analysis / "results.json").read_text())
    if not report["coverage"]["eligible_for_frozen_signal_selection"]:
        raise ValueError("Development sample/strata/uncertainty gates did not pass")
    paired = validate_pairs(
        args.rollouts, allowed_tasks=task_keys("dev"), stage="dev_rollouts"
    )
    raw_config = json.loads((args.rollouts / "manifest.json").read_text())["config"]
    brightness, sigma = (
        raw_config["args"]["brightness"],
        raw_config["args"]["noise_sigma"],
    )

    def canonical(rows):
        return sorted(json.dumps(row, sort_keys=True) for row in rows)

    if (
        canonical(raw_config["episodes"])
        != canonical(episode_plan("dev", brightness, sigma))
        or paired["pairs"] != 192
    ):
        raise ValueError("Expected the exact 384-episode isolated development cohort")
    raw_results = json.loads((args.rollouts / "results.json").read_text())["episodes"]
    shifted = [row for row in raw_results if row["condition"] == "shift"]
    shifted_success = sum(row["success"] for row in shifted) / len(shifted)
    if not 0.5 <= shifted_success <= 0.6:
        raise ValueError(f"Shift calibration outside 50-60% success: {shifted_success}")
    candidates = [
        row
        for row in report["audit"]
        if row["matching"] == "time_task_condition" and row["signal"] != "elapsed"
    ]
    candidates.sort(key=lambda row: (-row["estimate"], row["signal"], row["reduction"]))
    winner = candidates[0]
    chosen = {
        signal: next(
            {"signal": row["signal"], "reduction": row["reduction"]}
            for row in candidates
            if row["signal"] == signal
        )
        for signal in ("consistency", "dispersion", "support")
    }
    rows = pd.read_parquet(args.analysis / "scored_landmarks.parquet")
    common = add_history_reductions(rows).dropna(subset=list(SIGNALS))
    features = [f"{name}__last" for name in ("support", "dispersion", "consistency")]
    scales = {name: float(common[name].std(ddof=0)) for name in features}
    if not all(math.isfinite(value) and value > 0 for value in scales.values()):
        raise ValueError("Development normalization needs finite positive scales")
    boundaries = boundary_histories(
        rows, signal=winner["signal"], reduction=winner["reduction"]
    )
    alarm = calibrate_alarm(
        boundaries, 0.75, signal=winner["signal"], reduction=winner["reduction"]
    )
    protocol = {
        "schema": 1,
        "status": "frozen",
        "method_text": (ROOT / "CONFIRMATORY_METHODS.md").read_text(),
        "scope": "core H1-H4 study; H2 descriptive; H5 optional stretch excluded",
        "split": task_split(),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
        "execution": {
            "device": "mps",
            "threads": 4,
            "k": 8,
            "shared_prefix": True,
            "render_mode": "always",
            "reset_mode": "isolated",
        },
        "visual_shift": {"brightness": brightness, "noise_sigma": sigma},
        "test_episodes": episode_plan("test", brightness, sigma),
        "statistics": {
            "bootstrap": 2000,
            "seed": 20261111,
            "cluster": "pair_id",
            "minimum_valid_bootstrap_fraction": 0.9,
            "interval_level": 0.95,
            "sequence": ["H1", "H3", "H4a", "H4b"],
            "decision": "lower two-sided 95% bound strictly above null; no later confirmatory claims after sequence closes",
            "inference_scope": "selected held-out tasks and episode population; bootstrap coverage approximate; H4 noise bounds conditional on fixed selected states",
        },
        "h1": {
            "signal": winner["signal"],
            "reduction": winner["reduction"],
            "selection": "maximum dev primary point estimate; lexical signal/reduction tie break",
        },
        "per_signal_reductions": chosen,
        "h3": {
            "reduction": "last",
            "features": features,
            "scales": scales,
            "interval": "basic paired bootstrap",
            "method": "pairwise conditional composite logistic likelihood",
        },
        "h4": {
            "signal": winner["signal"],
            "reduction": winner["reduction"],
            "alarm": alarm,
            "pairs_per_cell": 2,
            "replicates": 3,
            "seed": 20261111,
            "sampling": "exact task/condition/time, first eligible crossing, no previously alarmed controls, no source pair reuse across conditions",
            "budget": "both arms receive horizon-checkpoint forward actions; oracle rollback by 50; no physical recovery-cost claim",
            "interval": "pointwise 95% Hoeffding over independent paired seed blocks, fixed selected states",
        },
        "training_source": {
            "repo": "lerobot/libero",
            "revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
            "historical_training_bytes_known": False,
            "episodes_per_task": 5,
            "frames_per_episode": 10,
            "seed": 20261111,
        },
        "development": {
            "run": str(args.rollouts.resolve()),
            "analysis": str(args.analysis.resolve()),
            "analysis_results_sha": file_sha256(args.analysis / "results.json"),
            "landmarks_sha": file_sha256(args.analysis / "scored_landmarks.parquet"),
            "paired_scenes_verified": paired["pairs"],
            "shift_success": shifted_success,
        },
        "runtime": runtime_fingerprint(),
        "dependency_sources": dependency_hashes(),
        "simulator_assets": asset_hashes(),
        "model_snapshots": model_hashes(),
        "critical_sources": critical_source_hashes(),
    }
    output = ROOT / "frozen_protocol.json"
    write_json(protocol, output)
    print(f"PROTOCOL {output}")
    print(
        f"H1 {protocol['h1']} H4_THRESHOLD {alarm['threshold']} SHIFT_SUCCESS {shifted_success}"
    )
    print(
        "A local freeze commit and receipt are required before any test entry point can run."
    )


if __name__ == "__main__":
    main()
