import pandas as pd
import pytest
import torch

from robomon.data.demos import select_demonstrations, validate_catalog
from robomon.policy.smolvla import SmolVLARunner


def catalog():
    frames = pd.DataFrame(
        [
            dict(
                index=ep * 6 + t,
                episode_index=ep,
                frame_index=t,
                task_index=ep // 4,
                timestamp=t / 10,
            )
            for ep in range(8)
            for t in range(6)
        ]
    )
    episodes = pd.DataFrame(
        [
            dict(
                episode_index=ep,
                length=6,
                dataset_from_index=ep * 6,
                dataset_to_index=(ep + 1) * 6,
            )
            for ep in range(8)
        ]
    )
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=["task a", "task b"])
    return frames, episodes, tasks


def test_catalog_catches_wrong_episode_offsets_and_tasks():
    frames, episodes, tasks = catalog()
    validate_catalog(frames, episodes, tasks)
    bad = episodes.copy()
    bad.loc[1, "dataset_from_index"] += 1
    with pytest.raises(ValueError, match="offset"):
        validate_catalog(frames, bad, tasks)
    bad_frames = frames.copy()
    bad_frames.loc[0, "task_index"] = 1
    with pytest.raises(ValueError, match="Multiple instructions"):
        validate_catalog(bad_frames, episodes, tasks)
    bad_frames = frames.copy()
    bad_frames.loc[5, "index"] = 4
    with pytest.raises(ValueError, match="unique"):
        validate_catalog(bad_frames, episodes, tasks)


def test_sampling_is_task_local_unique_and_includes_endpoints():
    frames, _, _ = catalog()
    mapping = {
        "a": {"dataset_task_index": 0, "instruction": "task a"},
        "b": {"dataset_task_index": 1, "instruction": "task b"},
    }
    both = select_demonstrations(
        frames, mapping, ["b", "a"], episodes_per_task=2, frames_per_episode=3
    )
    a = select_demonstrations(
        frames, mapping, ["a"], episodes_per_task=2, frames_per_episode=3
    )
    assert [r for r in both if r["task"] == "a"] == a
    assert len({r["episode_index"] for r in both}) == 4
    assert all(r["frame_indices"] == [0, 2, 5] for r in both)
    assert all(
        r["global_indices"] == [r["episode_index"] * 6 + t for t in r["frame_indices"]]
        for r in both
    )


def test_dataset_preprocessing_does_not_rotate_or_renormalize_canonical_pixels():
    runner = SmolVLARunner.__new__(SmolVLARunner)
    runner.pre = runner.env_pre = lambda x: x
    image = torch.zeros(3, 256, 256)
    image[:, 0, 0] = 1
    sample = {
        "observation.images.image": image,
        "observation.images.image2": image.clone(),
        "observation.state": torch.arange(8, dtype=torch.float32),
        "task": "task a",
    }
    batch = runner._prepare_demo_frame(sample)
    assert torch.equal(batch["observation.images.image"][0], image)
    assert batch["task"] == ["task a"]
    batch["observation.images.image"].zero_()
    assert sample["observation.images.image"][:, 0, 0].sum() == 3
    sample["observation.images.image"] = torch.zeros(256, 256, 3)
    with pytest.raises(ValueError, match="Invalid canonical"):
        runner._prepare_demo_frame(sample)
