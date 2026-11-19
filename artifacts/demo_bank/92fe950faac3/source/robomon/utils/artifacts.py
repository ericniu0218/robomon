"""Episode transactions: completion is published last and verified by content hashes on resume."""
import json
from pathlib import Path

import numpy as np

from robomon.rollout import Rollout
from robomon.utils.io import config_hash, file_sha256, write_json, write_parquet


def save_episode(result: Rollout, directory: Path, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_parquet(result.events, directory / "events.parquet")
    with (directory / "arrays.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **result.arrays)
    (directory / "arrays.npz.tmp").replace(directory / "arrays.npz")
    write_json(result.summary, directory / "summary.json")
    hashes = {name: file_sha256(directory / name) for name in ("events.parquet", "arrays.npz", "summary.json")}
    write_json({"config_hash": config_hash(config), "files": hashes}, directory / "complete.json")


def completed_episode(directory: Path, config: dict) -> bool:
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    if data["config_hash"] != config_hash(config):
        raise ValueError(f"Episode config mismatch: {directory}")
    for name in ("events.parquet", "arrays.npz", "summary.json"):
        path = directory / name
        if not path.exists() or data["files"].get(name) != file_sha256(path):
            raise ValueError(f"Completed episode is damaged: {path}; preserve it for inspection")
    return True
