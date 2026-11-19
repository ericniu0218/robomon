#!/usr/bin/env python3
"""Verify every saved H4 development branch, then report finite selected-state contrasts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from robomon.analysis.interventions import summarize_branches
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
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    upstream = manifest["config"]
    results = json.loads((args.run / "results.json").read_text())
    if (
        upstream.get("stage") != "h4_dev"
        or config_hash(upstream) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified development branch run")
    if results["status"] != "complete" or results["completed"] != len(upstream["jobs"]):
        raise ValueError("H4 development run is incomplete")
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
    frame = pd.DataFrame(records)
    report = summarize_branches(
        frame[frame.kind == "intervention"], upstream["args"]["replicates"], args.alpha
    )
    report["gates_passed"] = int(frame.kind.eq("gate").sum())
    config = {
        "stage": "h4_analysis_dev",
        "scope": "development only",
        "source_hash": source_hash(),
        "run_manifest_sha": file_sha256(args.run / "manifest.json"),
        "run_results_sha": file_sha256(args.run / "results.json"),
        "branch_completion_hashes": hashes,
        "alpha": args.alpha,
        "interval": "pointwise fixed-state Monte Carlo Hoeffding bounds",
    }
    output = stage_dir("h4_analysis_dev", config)
    write_manifest(output, "h4_analysis_dev", config)
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
