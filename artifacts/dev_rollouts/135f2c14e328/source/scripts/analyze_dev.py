#!/usr/bin/env python3
"""Audit one explicitly selected complete development run; never select inputs by modification time."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robomon.analysis.development import audit_tables, coverage_summary, json_safe, score_completed_run
from robomon.monitors.bank import SupportBank
from robomon.utils.io import file_sha256, source_hash, stage_dir, write_json, write_manifest, write_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--bootstrap", default=1000, type=int)
    parser.add_argument("--seed", default=20261111, type=int)
    args = parser.parse_args()
    bank = SupportBank(args.bank)
    rows, summaries, upstream = score_completed_run(args.rollouts, bank)
    config = {"stage": "dev_analysis", "schema": 1, "source_hash": source_hash(),
              "rollout_manifest_sha": file_sha256(args.rollouts / "manifest.json"),
              "rollout_results_sha": file_sha256(args.rollouts / "results.json"),
              "bank_completion_sha": file_sha256(args.bank / "complete.json"),
              "rollout_config": upstream, "bank_config": bank.config,
              "n_boot": args.bootstrap, "seed": args.seed,
              "common_landmarks": "all four raw signals nonmissing", "bootstrap_cluster": "pair_id"}
    output = stage_dir("dev_analysis", config)
    write_manifest(output, "dev_analysis", config)
    write_parquet(rows, output / "scored_landmarks.parquet")
    results = {"scope": "development only", "coverage": coverage_summary(summaries),
               "audit": audit_tables(rows, n_boot=args.bootstrap, seed=args.seed)}
    write_json(json_safe(results), output / "results.json")
    print(f"OUTPUT {output}")
    print(results["coverage"])


if __name__ == "__main__":
    main()
