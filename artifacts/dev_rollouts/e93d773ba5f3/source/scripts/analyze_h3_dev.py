#!/usr/bin/env python3
"""Development diagnostics for the H3 pairwise conditional composite-logistic estimator."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from robomon.analysis.conditional import fit_pairwise_conditional
from robomon.analysis.development import add_history_reductions, json_safe
from robomon.utils.io import (
    file_sha256,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--reduction", choices=["max", "mean", "last"], default="last")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    upstream = json.loads((args.analysis / "results.json").read_text())
    if upstream["scope"] != "development only":
        raise ValueError("This script is only for development analysis")
    rows = pd.read_parquet(args.analysis / "scored_landmarks.parquet")
    rows = add_history_reductions(rows).dropna(
        subset=["support", "dispersion", "consistency"]
    )
    features = tuple(
        f"{signal}__{args.reduction}"
        for signal in ("support", "dispersion", "consistency")
    )
    config = {
        "stage": "h3_dev",
        "scope": "development only",
        "source_hash": source_hash(),
        "landmarks_sha": file_sha256(args.analysis / "scored_landmarks.parquet"),
        "analysis_manifest_sha": file_sha256(args.analysis / "manifest.json"),
        "features": features,
        "n_boot": args.bootstrap,
        "method": "pairwise_conditional_composite_logistic",
        "interval": "basic",
    }
    output = stage_dir("h3_dev", config)
    write_manifest(output, "h3_dev", config)
    result = fit_pairwise_conditional(rows, features, n_boot=args.bootstrap)
    write_json(json_safe(result), output / "results.json")
    print(f"OUTPUT {output}")
    print(result)


if __name__ == "__main__":
    main()
