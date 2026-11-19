#!/usr/bin/env python3
"""Execute the frozen H1/H2/H3 analysis once, retaining all prespecified descriptive results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from robomon.analysis.conditional import fit_pairwise_conditional
from robomon.analysis.development import (
    SIGNALS,
    add_history_reductions,
    audit_tables,
    json_safe,
    score_run,
)
from robomon.analysis.landmarks import bootstrap_concordance
from robomon.monitors.bank import SupportBank
from robomon.study import load_frozen_protocol, sequence_decisions, task_keys
from robomon.study_data import validate_pairs
from robomon.utils.io import (
    file_sha256,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
    write_parquet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    allowed = task_keys("test")
    bank = SupportBank(args.bank)
    if (
        set(bank.by_task) != set(allowed)
        or bank.config.get("protocol_sha") != protocol["_protocol_sha256"]
    ):
        raise ValueError("Held-out bank does not belong to this frozen protocol")
    paired = validate_pairs(args.rollouts, allowed_tasks=allowed, stage="test_rollouts")
    rows, summaries, upstream = score_run(
        args.rollouts, bank, allowed_tasks=allowed, expected_stage="test_rollouts"
    )
    if (
        upstream["episodes"] != protocol["test_episodes"]
        or upstream["protocol_sha"] != protocol["_protocol_sha256"]
    ):
        raise ValueError("Unexpected held-out cohort/protocol")
    n_boot, seed = protocol["statistics"]["bootstrap"], protocol["statistics"]["seed"]
    enriched = add_history_reductions(rows)
    common = enriched.dropna(subset=list(SIGNALS))
    h1_choice = protocol["h1"]
    h1 = bootstrap_concordance(
        common,
        f"{h1_choice['signal']}__{h1_choice['reduction']}",
        n_boot=n_boot,
        seed=seed,
        cluster="pair_id",
        cluster_universe=rows,
    )
    h1["status"] = (
        "ok"
        if h1["pairs"] > 0
        and h1["bootstrap_valid"] >= 0.9 * n_boot
        and np.isfinite(h1["estimate"])
        else "inconclusive"
    )
    h1.update(signal=h1_choice["signal"], reduction=h1_choice["reduction"])
    features = tuple(
        f"{name}__last" for name in ("support", "dispersion", "consistency")
    )
    try:
        h3 = fit_pairwise_conditional(
            common,
            features,
            scales=protocol["h3"]["scales"],
            n_boot=n_boot,
            seed=seed,
            cluster_universe=rows,
        )
    except ValueError as error:
        if str(error) != "No comparable outcome pairs":
            raise
        h3 = {"status": "inconclusive", "reason": str(error), "coefficients": None}
    h3_support = (
        None
        if not h3.get("coefficients")
        else {"status": h3["status"], **h3["coefficients"]["support__last"]}
    )
    config = {
        "stage": "test_analysis",
        "scope": "frozen held-out H1/H2/H3",
        "source_hash": source_hash(),
        "protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "rollout_manifest_sha": file_sha256(args.rollouts / "manifest.json"),
        "rollout_results_sha": file_sha256(args.rollouts / "results.json"),
        "bank_completion_sha": file_sha256(args.bank / "complete.json"),
        "rollout_config": upstream,
        "n_boot": n_boot,
        "seed": seed,
    }
    output = stage_dir("test_analysis", config)
    write_manifest(output, "test_analysis", config)
    write_parquet(rows, output / "scored_landmarks.parquet")
    write_json(paired, output / "paired_scene_validation.json")
    audit = audit_tables(rows, n_boot=n_boot, seed=seed)
    coverage = [
        {
            "task": task,
            "condition": condition,
            "episodes": len(group),
            "successes": int(group.success.sum()),
            "mean_steps": float(group.steps.mean()),
        }
        for (task, condition), group in summaries.groupby(
            ["task", "condition"], sort=True
        )
    ]
    report = {
        "scope": "held-out; frozen protocol",
        "h1": h1,
        "h3": h3,
        "sequence_decisions_pending_h4": sequence_decisions(
            {"H1": h1, "H3": h3_support}
        ),
        "audit": audit,
        "coverage": coverage,
        "episodes": len(summaries),
        "matched_rows": len(common),
        "paired_scenes_verified": paired["pairs"],
        "freeze_commit": protocol["_freeze_commit"],
    }
    write_json(json_safe(report), output / "results.json")
    print(f"OUTPUT {output}", flush=True)
    print(
        json.dumps(
            json_safe(
                {
                    "h1": h1,
                    "h3": h3,
                    "sequence": report["sequence_decisions_pending_h4"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
