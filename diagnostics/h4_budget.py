"""Prospective H4 Monte Carlo precision arithmetic; NOT a power or state-generalization analysis.

Run before choosing a held-out budget. No intervention outcomes are read by this diagnostic.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robomon.utils.io import (
    file_sha256,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-sets", nargs="+", type=int, default=[19, 56, 80])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-radius", type=float, default=0.1)
    args = parser.parse_args()
    if (
        not 0 < args.alpha < 1
        or not 0 < args.target_radius < 1
        or args.replicates < 1
        or min(args.matched_sets) < 1
    ):
        raise ValueError(
            "Positive counts, alpha in (0,1), and target radius in (0,1) required"
        )
    coefficient = 2 * math.log(2 / args.alpha)
    blocks_required = math.ceil(coefficient / args.target_radius**2)
    scenarios = []
    for matched_sets in args.matched_sets:
        needed_replicates = math.ceil(blocks_required / matched_sets)
        scenarios.append(
            {
                "matched_sets": matched_sets,
                "selected_states": 2 * matched_sets,
                "replicates": args.replicates,
                "independent_seed_blocks": matched_sets * args.replicates,
                "intervention_branches": 4 * matched_sets * args.replicates,
                "known_answer_gates": 2 * matched_sets,
                "hoeffding_radius": math.sqrt(
                    coefficient / (matched_sets * args.replicates)
                ),
                "replicates_for_target_radius": needed_replicates,
                "intervention_branches_for_target_radius": 4
                * matched_sets
                * needed_replicates,
            }
        )
    config = {
        "stage": "h4_precision",
        "source_hash": source_hash(),
        "diagnostic_sha": file_sha256(Path(__file__)),
        "args": vars(args),
    }
    output = stage_dir("h4_precision", config)
    write_manifest(output, "h4_precision", config)
    write_json(
        {
            "scope": "prospective fixed-state Monte Carlo precision; no outcomes read",
            "alpha": args.alpha,
            "target_radius": args.target_radius,
            "minimum_independent_blocks_for_target": blocks_required,
            "scenarios": scenarios,
            "limitations": "Not power; no new-state/task uncertainty; assumes every planned exact match is available; no sample-size adaptation to outcomes.",
        },
        output / "results.json",
    )
    print(f"OUTPUT {output}")
    for row in scenarios:
        print(row)


if __name__ == "__main__":
    main()
