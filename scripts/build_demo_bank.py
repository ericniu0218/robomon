#!/usr/bin/env python3
"""Build a resumable, outcome-blind support bank from pinned training-source demonstrations."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from robomon.data.demos import (
    DATASET_REPO,
    DATASET_REVISION,
    catalog_snapshot,
    prefetch_demonstrations,
    read_catalog,
    select_demonstrations,
    task_mapping,
)
from robomon.monitors.bank import BANK_FILES, SHARD_FILES, SupportBank, verify_files
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
    SmolVLARunner,
)
from robomon.study import load_frozen_protocol, task_keys
from robomon.utils.io import (
    config_hash,
    file_sha256,
    runtime_fingerprint,
    source_hash,
    stage_dir,
    write_json,
    write_manifest,
    write_parquet,
)


def complete(directory: Path, config: dict, names: tuple[str, ...]) -> None:
    write_json(
        {
            "config_hash": config_hash(config),
            "files": {name: file_sha256(directory / name) for name in names},
        },
        directory / "complete.json",
    )


def save_arrays(embeddings, records, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "embeddings.npy.tmp").open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    (directory / "embeddings.npy.tmp").replace(directory / "embeddings.npy")
    write_parquet(pd.DataFrame(records), directory / "frames.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional subset of dev task keys, e.g. libero_spatial/0",
    )
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--frames-per-episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--partition", choices=["dev", "test"], default="dev")
    parser.add_argument("--protocol")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    protocol = None
    if args.partition == "test":
        if not args.protocol or not args.receipt:
            parser.error("Held-out banks require a committed frozen protocol")
        args.protocol, args.receipt = (
            str(Path(args.protocol).resolve()),
            str(Path(args.receipt).resolve()),
        )
        protocol = load_frozen_protocol(args.protocol, args.receipt)
        if (
            args.episodes_per_task,
            args.frames_per_episode,
            args.seed,
            args.device,
            args.threads,
        ) != (5, 10, 20261111, "mps", 4):
            parser.error("Held-out bank settings must match the frozen protocol")
    allowed = task_keys(args.partition)
    keys = sorted(args.tasks or allowed)
    if not set(keys) <= set(allowed) or (args.partition == "test" and keys != allowed):
        parser.error("Requested tasks do not match the declared partition")
    root = catalog_snapshot()
    frames, tasks, catalog_hashes = read_catalog(root)
    mapping = task_mapping(tasks)
    selection = select_demonstrations(
        frames,
        mapping,
        keys,
        episodes_per_task=args.episodes_per_task,
        frames_per_episode=args.frames_per_episode,
        seed=args.seed,
    )
    config = {
        "stage": "demo_bank" if args.partition == "dev" else "demo_bank_test",
        "schema": 1,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_training_revision_known": False,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
        "video_backend": "pyav",
        "device": args.device,
        "runtime": runtime_fingerprint(),
        "source_hash": source_hash(),
        "args": vars(args),
        "catalog_files": catalog_hashes,
        "selection": selection,
    }
    if protocol is not None:
        config.update(
            protocol_sha=protocol["_protocol_sha256"],
            freeze_commit=protocol["_freeze_commit"],
        )
    output = stage_dir(config["stage"], config)
    print(f"OUTPUT {output}", flush=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify_files(output, config, BANK_FILES):
            SupportBank(output)
            print("RESUME complete bank; no model loaded", flush=True)
            return
        write_manifest(output, config["stage"], config)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        pending = [
            row
            for row in selection
            if not verify_files(
                output / "shards" / str(row["episode_index"]),
                {"parent": config_hash(config), "selection": row},
                (*SHARD_FILES, "inputs.json"),
            )
        ]
        input_hashes = {}
        if pending:
            # Download before loading weights; run ONE heavy model process at a time on 16 GB Macs.
            prefetch_demonstrations([row["episode_index"] for row in pending])
            dataset = LeRobotDataset(
                DATASET_REPO,
                episodes=[row["episode_index"] for row in pending],
                revision=DATASET_REVISION,
                video_backend="pyav",
            )
            ids = dataset.hf_dataset.select_columns("index")[:]["index"]
            relative = {int(global_id): i for i, global_id in enumerate(ids)}
            policy = SmolVLARunner(threads=args.threads, device=args.device)
            for number, row in enumerate(pending):
                start = time.perf_counter()
                records, embeddings = [], []
                for frame_id, global_id in zip(
                    row["frame_indices"], row["global_indices"], strict=True
                ):
                    sample = dataset[relative[global_id]]
                    actual = (
                        int(sample["index"]),
                        int(sample["episode_index"]),
                        int(sample["frame_index"]),
                        int(sample["task_index"]),
                    )
                    expected = (
                        global_id,
                        row["episode_index"],
                        frame_id,
                        row["dataset_task_index"],
                    )
                    if (
                        actual != expected
                        or sample["task"].strip().lower()
                        != row["instruction"].strip().lower()
                    ):
                        raise ValueError(
                            f"Decoded frame does not match selection: {actual} != {expected}"
                        )
                    embedding = policy.embed_demo_frame(sample)
                    if embedding.shape != (960,) or not np.isfinite(embedding).all():
                        raise ValueError("Invalid extracted embedding")
                    embeddings.append(embedding)
                    records.append(
                        {
                            "task": row["task"],
                            "dataset_task_index": row["dataset_task_index"],
                            "episode_index": row["episode_index"],
                            "frame_index": frame_id,
                            "index": global_id,
                            "timestamp": float(sample["timestamp"]),
                        }
                    )
                shard = output / "shards" / str(row["episode_index"])
                save_arrays(embeddings, records, shard)
                files = [dataset.meta.get_data_file_path(row["episode_index"])]
                files += [
                    dataset.meta.get_video_file_path(row["episode_index"], cam)
                    for cam in dataset.meta.video_keys
                ]
                for file in files:
                    name = str(file)
                    if name not in input_hashes:
                        input_hashes[name] = file_sha256(dataset.root / file)
                write_json(
                    {str(f): input_hashes[str(f)] for f in files}, shard / "inputs.json"
                )
                # Include input video hashes in the shard transaction as well.
                complete(
                    shard,
                    {"parent": config_hash(config), "selection": row},
                    (*SHARD_FILES, "inputs.json"),
                )
                print(
                    f"{number + 1}/{len(pending)} {row['task']} demo={row['episode_index']} "
                    f"frames={len(records)} seconds={time.perf_counter() - start:.2f}",
                    flush=True,
                )
        all_embeddings, all_frames = [], []
        for row in selection:
            shard = output / "shards" / str(row["episode_index"])
            if not verify_files(
                shard,
                {"parent": config_hash(config), "selection": row},
                (*SHARD_FILES, "inputs.json"),
            ):
                raise ValueError(f"Incomplete shard: {shard}")
            input_hashes.update(json.loads((shard / "inputs.json").read_text()))
            all_embeddings.append(np.load(shard / "embeddings.npy", allow_pickle=False))
            all_frames.append(pd.read_parquet(shard / "frames.parquet"))
        joined_frames = pd.concat(all_frames, ignore_index=True)
        save_arrays(
            np.concatenate(all_embeddings), joined_frames.to_dict("records"), output
        )
        write_json(
            {
                "status": "complete",
                "frames": len(joined_frames),
                "episodes": len(selection),
                "tasks": {
                    str(k): int(v)
                    for k, v in joined_frames.groupby("task").size().items()
                },
                "input_files": input_hashes,
            },
            output / "results.json",
        )
        complete(output, config, BANK_FILES)
        SupportBank(output)
        print(
            f"COMPLETE {len(joined_frames)} embeddings, {len(selection)} demonstrations",
            flush=True,
        )


if __name__ == "__main__":
    main()
