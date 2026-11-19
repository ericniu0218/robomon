#!/usr/bin/env python3
"""Calibrate a development alarm and select controls without reading terminal labels."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from robomon.interventions import (
    STATE_COLUMNS,
    boundary_histories,
    calibrate_alarm,
    match_alarm_states,
)
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
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--pairs-per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    upstream = json.loads((args.analysis / "manifest.json").read_text())
    if (
        upstream["config"].get("stage") != "dev_analysis"
        or config_hash(upstream["config"]) != upstream["config_hash"]
    ):
        raise ValueError("Expected a verified development analysis")
    # Column projection makes terminal labels unavailable to calibration and matching.
    rows = pd.read_parquet(
        args.analysis / "scored_landmarks.parquet",
        columns=STATE_COLUMNS + ["consistency"],
    )
    boundaries = boundary_histories(rows)
    alarm = calibrate_alarm(boundaries, args.quantile)
    selection = match_alarm_states(
        boundaries,
        alarm["threshold"],
        max_pairs_per_cell=args.pairs_per_cell,
        seed=args.seed,
    )
    config = {
        "stage": "h4_selection_dev",
        "scope": "development only; no confirmatory H4 claim",
        "source_hash": source_hash(),
        "args": {**vars(args), "analysis": str(args.analysis)},
        "analysis_manifest_sha": file_sha256(args.analysis / "manifest.json"),
        "landmarks_sha": file_sha256(args.analysis / "scored_landmarks.parquet"),
        "rollout_config": upstream["config"]["rollout_config"],
        "alarm": alarm,
        "matching": "first eligible crossing; exact task/condition/landmark; disjoint source pairs; hash-ordered greedy",
        "protocol_text": (
            Path(__file__).resolve().parents[1] / "H4_PROTOCOL.md"
        ).read_text(),
        "protocol_sha": file_sha256(
            Path(__file__).resolve().parents[1] / "H4_PROTOCOL.md"
        ),
    }
    output = stage_dir("h4_selection_dev", config)
    write_manifest(output, "h4_selection_dev", config)
    write_json(
        {"scope": config["scope"], "alarm": alarm, **selection}, output / "results.json"
    )
    print(f"OUTPUT {output}")
    print(
        f"THRESHOLD {alarm['threshold']} MATCHED_SETS {len(selection['matches'])} STATES {2 * len(selection['matches'])}"
    )
    for row in selection["coverage"]:
        print(row)


if __name__ == "__main__":
    main()
