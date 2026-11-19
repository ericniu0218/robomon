#!/usr/bin/env python3
"""Select held-out states with the already frozen DEV threshold; never tune on test labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from robomon.interventions import STATE_COLUMNS, boundary_histories, match_alarm_states
from robomon.study import load_frozen_protocol, task_keys
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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    manifest = json.loads((args.analysis / "manifest.json").read_text())
    upstream = manifest["config"]
    if (
        upstream.get("stage") != "test_analysis"
        or config_hash(upstream) != manifest["config_hash"]
        or upstream["protocol_sha"] != protocol["_protocol_sha256"]
    ):
        raise ValueError("Expected the frozen held-out analysis")
    h4 = protocol["h4"]
    rows = pd.read_parquet(
        args.analysis / "scored_landmarks.parquet",
        columns=STATE_COLUMNS + [h4["signal"]],
    )
    allowed = task_keys("test")
    boundaries = boundary_histories(
        rows, signal=h4["signal"], reduction=h4["reduction"], allowed_tasks=allowed
    )
    selection = match_alarm_states(
        boundaries,
        h4["alarm"]["threshold"],
        max_pairs_per_cell=h4["pairs_per_cell"],
        seed=h4["seed"],
        allowed_tasks=allowed,
    )
    config = {
        "stage": "h4_selection_test",
        "scope": "frozen held-out outcome-blind state selection",
        "source_hash": source_hash(),
        "frozen_protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "args": {"analysis": str(args.analysis.resolve())},
        "analysis_manifest_sha": file_sha256(args.analysis / "manifest.json"),
        "landmarks_sha": file_sha256(args.analysis / "scored_landmarks.parquet"),
        "rollout_config": upstream["rollout_config"],
        "alarm": h4["alarm"],
        "protocol_text": protocol["method_text"],
        "protocol_sha": hashlib.sha256(protocol["method_text"].encode()).hexdigest(),
    }
    output = stage_dir("h4_selection_test", config)
    write_manifest(output, "h4_selection_test", config)
    write_json(
        {"scope": config["scope"], "alarm": h4["alarm"], **selection},
        output / "results.json",
    )
    print(f"OUTPUT {output}")
    print(
        f"MATCHED_SETS {len(selection['matches'])} STATES {2 * len(selection['matches'])}"
    )


if __name__ == "__main__":
    main()
