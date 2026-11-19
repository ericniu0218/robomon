"""Checked, task-specific demonstration bank. No fallback to another task is allowed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robomon.monitors.signals import support_distance
from robomon.data.demos import DATASET_REPO, DATASET_REVISION
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
)
from robomon.utils.io import config_hash, file_sha256


def verify_files(directory: Path, config: dict, filenames: tuple[str, ...]) -> bool:
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    stored = json.loads(marker.read_text())
    if stored["config_hash"] != config_hash(config):
        raise ValueError(f"Bank configuration mismatch: {directory}")
    for name in filenames:
        path = directory / name
        if not path.exists() or stored["files"].get(name) != file_sha256(path):
            raise ValueError(f"Damaged bank artifact: {path}")
    return True


BANK_FILES = ("embeddings.npy", "frames.parquet", "results.json")
SHARD_FILES = ("embeddings.npy", "frames.parquet")


class SupportBank:
    def __init__(self, path: Path):
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.config = manifest["config"]
        if manifest["config_hash"] != config_hash(self.config):
            raise ValueError("Bank manifest config hash mismatch")
        if not verify_files(self.path, self.config, BANK_FILES):
            raise ValueError("Bank extraction is incomplete")
        expected = {
            "checkpoint_revision": CHECKPOINT_REVISION,
            "vlm_revision": VLM_REVISION,
            "embedding_definition": EMBEDDING_DEFINITION,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
        }
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f"Incompatible bank {key}")
        self.frames = pd.read_parquet(self.path / "frames.parquet").reset_index(
            drop=True
        )
        self.embeddings = np.load(self.path / "embeddings.npy", allow_pickle=False)
        if (
            self.embeddings.shape != (len(self.frames), 960)
            or not np.isfinite(self.embeddings).all()
            or np.any(np.linalg.norm(self.embeddings, axis=1) == 0)
        ):
            raise ValueError("Invalid bank embedding matrix")
        if self.frames.duplicated(["episode_index", "frame_index"]).any():
            raise ValueError("Repeated demonstration frame")
        self.by_task = {
            task: self.embeddings[group.index.to_numpy()]
            for task, group in self.frames.groupby("task")
        }

    def score(self, embedding: np.ndarray, task: str) -> float:
        if task not in self.by_task:
            raise ValueError(f"Bank has no demonstrations for {task}")
        return support_distance(embedding, self.by_task[task])
