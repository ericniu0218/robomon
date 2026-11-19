"""Validate the LIBERO training-source catalog before selecting demonstration frames.

The policy's training config names this repository but does not pin its historical revision.
We pin the revision used for THIS audit; that is not proof of byte-identical historical training data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from robomon.splits import SUITES, task_split
from robomon.utils.io import file_sha256

DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def catalog_snapshot() -> Path:
    from huggingface_hub import constants, snapshot_download
    return Path(snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
                                  allow_patterns=["meta/*", "data/*"], max_workers=4,
                                  local_files_only=constants.HF_HUB_OFFLINE))


def prefetch_demonstrations(episodes: list[int]) -> None:
    """Populate the official LeRobot revision cache before constructing a filtered dataset.

    LeRobot 0.7's try_load can raise ValueError (rather than trigger download) if a partially cached
    dataset contains none of the requested episodes. Explicitly prefetching avoids that cold-cache
    trap and uses the metadata's physical file/video mapping without modifying package code.
    """
    from huggingface_hub import constants, snapshot_download
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    metadata = LeRobotDatasetMetadata(DATASET_REPO, revision=DATASET_REVISION)
    files = {str(metadata.get_data_file_path(ep)) for ep in episodes}
    files.update(str(metadata.get_video_file_path(ep, key)) for ep in episodes for key in metadata.video_keys)
    snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
                      cache_dir=HF_LEROBOT_HUB_CACHE, allow_patterns=["meta/*", *sorted(files)],
                      max_workers=4, local_files_only=constants.HF_HUB_OFFLINE)


def validate_catalog(frames: pd.DataFrame, episodes: pd.DataFrame, tasks: pd.DataFrame) -> None:
    """Detect index, task, or episode metadata errors instead of decoding the wrong video frame."""
    cols = ["index", "episode_index", "frame_index", "task_index", "timestamp"]
    if frames.empty or frames[cols].isna().any().any():
        raise ValueError("Catalog must have nonmissing frame identifiers and timestamps")
    for col in cols[:-1]:
        if not np.equal(frames[col], frames[col].astype(np.int64)).all():
            raise ValueError(f"Noninteger catalog identifier: {col}")
    if not np.array_equal(np.sort(frames["index"].to_numpy()), np.arange(len(frames))):
        raise ValueError("Global frame indices must be unique and contiguous")
    if episodes.episode_index.duplicated().any() or tasks.task_index.duplicated().any():
        raise ValueError("Duplicate episode or task metadata")
    if set(frames.episode_index) != set(episodes.episode_index):
        raise ValueError("Episode metadata does not cover the frame catalog")
    if not set(frames.task_index) <= set(tasks.task_index):
        raise ValueError("Unknown dataset task index")
    metadata = episodes.set_index("episode_index")
    for episode_id, group in frames.groupby("episode_index", sort=True):
        group = group.sort_values("index")
        row = metadata.loc[episode_id]
        if group.task_index.nunique() != 1:
            raise ValueError(f"Multiple instructions inside episode {episode_id}")
        if len(group) != int(row.length) or not np.array_equal(group.frame_index, np.arange(len(group))):
            raise ValueError(f"Invalid frame sequence in episode {episode_id}")
        if not np.array_equal(group["index"], np.arange(int(row.dataset_from_index), int(row.dataset_to_index))):
            raise ValueError(f"Wrong global offset for episode {episode_id}")
        if not np.allclose(group.timestamp, group.frame_index / 10., atol=1e-4, rtol=0):
            raise ValueError(f"Invalid 10 Hz timestamps in episode {episode_id}")


def read_catalog(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames_paths = sorted((root / "data").rglob("*.parquet"))
    episode_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    columns = ["index", "episode_index", "frame_index", "task_index", "timestamp"]
    frames = pd.concat([pd.read_parquet(p, columns=columns) for p in frames_paths], ignore_index=True)
    episodes = pd.concat([pd.read_parquet(p) for p in episode_paths], ignore_index=True)
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    validate_catalog(frames, episodes, tasks)
    paths = [*frames_paths, *episode_paths, root / "meta/tasks.parquet", root / "meta/info.json"]
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    return frames, tasks, hashes


def task_mapping(tasks: pd.DataFrame) -> dict:
    """Match benchmark instruction strings; never assume dataset and suite task IDs coincide."""
    from libero.libero import benchmark
    normalized = [str(x).strip().lower() for x in tasks.index]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Ambiguous normalized instruction strings")
    lookup = dict(zip(normalized, tasks.task_index.astype(int), strict=True))
    result = {}
    for name in SUITES:
        suite = benchmark.get_benchmark_dict()[name]()
        for task_id in range(10):
            instruction = suite.get_task(task_id).language
            key = instruction.strip().lower()
            if key not in lookup:
                raise ValueError(f"No training-source instruction for {name}/{task_id}: {instruction}")
            result[f"{name}/{task_id}"] = {"dataset_task_index": lookup[key], "instruction": instruction}
    if len({row["dataset_task_index"] for row in result.values()}) != 40:
        raise ValueError("Benchmark to dataset mapping is not one-to-one")
    return result


def dev_task_keys() -> list[str]:
    return sorted(f"{suite}/{i}" for suite, split in task_split()["suites"].items() for i in split["dev"])


def select_demonstrations(frames: pd.DataFrame, mapping: dict, task_keys: list[str], *,
                          episodes_per_task: int = 5, frames_per_episode: int = 10,
                          seed: int = 20261111) -> list[dict]:
    """Sample episodes uniformly by task, then evenly spaced frames including endpoints.

    Uses no rollout outcomes, support scores, or policy predictions. A per-task seed makes selection
    invariant to task order or selecting additional tasks later.
    """
    if episodes_per_task < 1 or frames_per_episode < 2:
        raise ValueError("Select at least one episode and two frames per episode")
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("Duplicate requested task")
    selected = []
    for task in sorted(task_keys):
        info = mapping[task]
        data = frames[frames.task_index == info["dataset_task_index"]]
        candidates = np.sort(data.episode_index.unique())
        if len(candidates) < episodes_per_task:
            raise ValueError(f"Not enough demonstration episodes for {task}")
        task_seed = int.from_bytes(hashlib.sha256(f"{seed}/{task}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(task_seed)
        for episode_id in sorted(rng.choice(candidates, episodes_per_task, replace=False)):
            episode = data[data.episode_index == episode_id].sort_values("frame_index")
            offsets = np.linspace(0, len(episode)-1, min(frames_per_episode, len(episode)), dtype=int)
            chosen = episode.iloc[offsets]
            selected.append({"task": task, **info, "episode_index": int(episode_id),
                             "frame_indices": chosen.frame_index.astype(int).tolist(),
                             "global_indices": chosen["index"].astype(int).tolist()})
    return selected
