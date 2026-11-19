"""Real tests for consistent dataset/online embeddings; run serially with the simulator/model."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


def test_demonstration_and_online_embedding_are_identical(policy, session):
    from lerobot.envs.utils import prepare_canonical_observation
    from lerobot.sim.integrations.libero import canonical_observation

    obs = session.reset(0)
    canonical = canonical_observation(obs)
    tensors = prepare_canonical_observation(
        {k: np.asarray(v)[None] for k, v in canonical.items()}
    )
    frame = {k: v[0] for k, v in tensors.items()}
    frame["task"] = session.task_description
    before = {k: v.clone() for k, v in frame.items() if isinstance(v, torch.Tensor)}
    embedded = policy.embed_demo_frame(frame)
    query = policy.query(
        obs, session.task_description, 1, torch.Generator().manual_seed(11)
    )
    assert np.array_equal(embedded, query.embedding)
    for key, value in before.items():
        assert torch.equal(frame[key], value)
