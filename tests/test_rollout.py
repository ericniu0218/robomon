from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robomon.rollout import EpisodeSpec, run_episode
from robomon.splits import task_split


class ToySession:
    max_steps = 120
    task_description = "test"

    def reset(self, init_state_id):
        self.x, self.t, self.ready = 0.0, 0, True
        return self.observe()

    def observe(self):
        assert self.ready
        return {
            "pixels": {"a": np.full((2, 2, 3), self.t, dtype=np.uint8)},
            "x": self.x,
        }

    def step(self, action, *, render=False):
        self.x += float(action[0])
        self.t += 1
        self.ready = render
        return SimpleNamespace(success=False, done=False)

    def get_state(self):
        return np.array([self.t, self.x])


class ToyPolicy:
    chunk_size = 50

    def query(self, obs, task, k, generator, *, shared_prefix=False):
        noise = torch.randn(k, 50, 7, generator=generator).numpy()
        # Image noise affects the action, so RNG-stream contamination cannot pass unnoticed.
        noise += obs["pixels"]["a"].mean() / 255 + obs["x"] / 1000
        return SimpleNamespace(
            chunks_norm=noise, exec_actions=noise[0], embedding=np.ones(4)
        )


@pytest.mark.parametrize("condition", ["nominal", "shift"])
def test_monitor_is_passive_and_reproducible(condition):
    spec = EpisodeSpec(
        "libero_spatial",
        0,
        0,
        13,
        condition,
        brightness=0.6 if condition == "shift" else 1.0,
        noise_sigma=10 if condition == "shift" else 0.0,
    )
    off = run_episode(ToyPolicy(), ToySession(), spec, monitor=False)
    on = run_episode(ToyPolicy(), ToySession(), spec)
    again = run_episode(ToyPolicy(), ToySession(), spec, k=3)
    for result in (on, again):
        assert np.array_equal(off.arrays["actions"], result.arrays["actions"])
        assert np.array_equal(off.arrays["states"], result.arrays["states"])
    assert list(on.events.landmark) == [0, 25, 50, 75, 100]
    assert on.events.failure.eq(1).all()
    assert np.isnan(on.events.consistency.iloc[0])
    assert on.summary["termination"] == "timeout"


def test_split_reserves_exposed_task_zero_and_is_disjoint():
    split = task_split()
    assert split == task_split()
    for suite in split["suites"].values():
        assert len(suite["dev"]) == 3 and len(suite["test"]) == 7
        assert 0 in suite["dev"]
        assert set(suite["dev"]).isdisjoint(suite["test"])
        assert sorted(suite["dev"] + suite["test"]) == list(range(10))
