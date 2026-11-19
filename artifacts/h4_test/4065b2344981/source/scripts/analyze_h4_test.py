#!/usr/bin/env python3
"""Verify frozen held-out branches and produce the prespecified hierarchical decisions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from robomon.analysis.interventions import summarize_branches
from robomon.study import load_frozen_protocol, sequence_decisions
from robomon.utils.artifacts import completed_array_result
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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    manifest = json.loads((args.run / "manifest.json").read_text())
    upstream = manifest["config"]
    results = json.loads((args.run / "results.json").read_text())
    if (
        upstream.get("stage") != "h4_test"
        or config_hash(upstream) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified held-out branch run")
    if results["status"] != "complete" or results["completed"] != len(upstream["jobs"]):
        raise ValueError("H4 held-out run is incomplete")
    if upstream["frozen_protocol_sha"] != protocol["_protocol_sha256"]:
        raise ValueError("H4 run protocol mismatch")
    records, hashes = [], {}
    for job in upstream["jobs"]:
        key = config_hash(job)
        directory = args.run / "branches" / key
        if not completed_array_result(directory, upstream):
            raise ValueError("Incomplete branch")
        row = json.loads((directory / "summary.json").read_text())
        if any(row.get(key) != value for key, value in job.items()):
            raise ValueError("Branch identity mismatch")
        if job["kind"] == "gate" and not row.get("passed", False):
            raise ValueError("Known-answer replay gate did not pass")
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            actions, states, embedding = (
                arrays["actions"],
                arrays["states"],
                arrays["first_embedding"],
            )
            if (
                actions.shape != (row["forward_steps"], 7)
                or states.ndim != 2
                or len(states) != row["forward_steps"]
                or embedding.shape != (960,)
            ):
                raise ValueError(
                    "Branch arrays disagree with the recorded forward steps"
                )
            if (
                not all(
                    np.isfinite(value).all() for value in (actions, states, embedding)
                )
                or not (np.diff(states[:, 0]) > 0).all()
            ):
                raise ValueError(
                    "Nonfinite or nonmonotone simulator trace; do not count it as a valid branch"
                )
        hashes[key] = file_sha256(directory / "complete.json")
        records.append(row)
    if records != results["records"]:
        raise ValueError("Run results disagree with completed branch summaries")
    if records:
        frame = pd.DataFrame(records)
        report = summarize_branches(
            frame[frame.kind == "intervention"], upstream["args"]["replicates"], 0.05
        )
        report["gates_passed"] = int(frame.kind.eq("gate").sum())
    else:
        report = {
            "matched_sets": 0,
            "states": 0,
            "branches": 0,
            "gates_passed": 0,
            "rates": {},
            "contrasts": {
                name: {
                    "status": "inconclusive",
                    "reason": "no exact matched states",
                    "ci_low": None,
                    "ci_high": None,
                    "estimate": None,
                }
                for name in (
                    "h4a_control_minus_alarm_replan",
                    "h4b_alarm_rollback_minus_replan",
                    "control_rollback_minus_replan_exploratory",
                )
            },
        }
    primary_manifest = json.loads((args.analysis / "manifest.json").read_text())
    primary_config = primary_manifest["config"]
    if (
        primary_config.get("stage") != "test_analysis"
        or config_hash(primary_config) != primary_manifest["config_hash"]
        or primary_config["protocol_sha"] != protocol["_protocol_sha256"]
    ):
        raise ValueError("Primary analysis protocol mismatch")
    selection_path = Path(upstream["args"]["selection"])
    if (
        file_sha256(selection_path / "manifest.json")
        != upstream["selection_manifest_sha"]
    ):
        raise ValueError("Selection input changed")
    selected_config = json.loads((selection_path / "manifest.json").read_text())[
        "config"
    ]
    if (
        file_sha256(args.analysis / "manifest.json")
        != selected_config["analysis_manifest_sha"]
    ):
        raise ValueError("H4 and H1/H3 analyses refer to different cohorts")
    config = {
        "stage": "h4_analysis_test",
        "scope": "frozen held-out H4",
        "source_hash": source_hash(),
        "run_manifest_sha": file_sha256(args.run / "manifest.json"),
        "run_results_sha": file_sha256(args.run / "results.json"),
        "branch_completion_hashes": hashes,
        "alpha": 0.05,
        "interval": "pointwise fixed-state Monte Carlo Hoeffding bounds",
        "protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "primary_manifest_sha": file_sha256(args.analysis / "manifest.json"),
        "primary_results_sha": file_sha256(args.analysis / "results.json"),
    }
    output = stage_dir("h4_analysis_test", config)
    write_manifest(output, "h4_analysis_test", config)
    primary = json.loads((args.analysis / "results.json").read_text())
    h3 = primary["h3"]
    support = (
        None
        if not h3.get("coefficients")
        else {"status": h3["status"], **h3["coefficients"]["support__last"]}
    )
    contrasts = report["contrasts"]
    report["status"] = "held-out; frozen protocol"
    report["sequence_decisions"] = sequence_decisions(
        {
            "H1": primary["h1"],
            "H3": support,
            "H4a": {"status": "ok", **contrasts["h4a_control_minus_alarm_replan"]},
            "H4b": {"status": "ok", **contrasts["h4b_alarm_rollback_minus_replan"]},
        }
    )
    report["freeze_commit"] = protocol["_freeze_commit"]
    write_json(report, output / "results.json")
    print(f"OUTPUT {output}")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "matched_sets",
                    "states",
                    "branches",
                    "gates_passed",
                    "rates",
                    "contrasts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
