"""Slow integration tests (real SmolVLA + LIBERO on CPU). Run with: pytest -m slow"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


def test_shared_prefix_matches_reference(policy, session):
    obs = session.reset(0)
    a = policy.query(obs, session.task_description, 4, torch.Generator().manual_seed(7), shared_prefix=True)
    b = policy.query(obs, session.task_description, 4, torch.Generator().manual_seed(7), shared_prefix=False)
    assert a.chunks_norm.shape == (4, policy.chunk_size, policy.action_dim)
    assert np.abs(a.chunks_norm - b.chunks_norm).max() < 1e-3
    assert np.abs(a.exec_actions - b.exec_actions).max() < 1e-3
    assert np.abs(a.embedding - b.embedding).max() < 1e-4
    # different seeds give different samples; same seed is reproducible
    c = policy.query(obs, session.task_description, 4, torch.Generator().manual_seed(8))
    assert np.abs(a.chunks_norm - c.chunks_norm).max() > 1e-3
    d = policy.query(obs, session.task_description, 4, torch.Generator().manual_seed(7))
    assert np.allclose(a.chunks_norm, d.chunks_norm, atol=1e-5)


def test_render_on_demand_matches_always_render(session):
    """Images produced on demand after physics-only steps equal images from the always-render path."""
    rng = np.random.default_rng(0)
    actions = np.clip(rng.normal(0, 0.3, size=(12, 7)), -1, 1)
    session.reset(1)
    for a in actions:
        session.step(a)
    on_demand = session.observe()
    session.reset(1)
    session._set_rendering(True)
    for a in actions:  # lerobot's own step renders every time
        obs, *_ = session.env.step(a.astype(np.float64))
    for cam in obs["pixels"]:
        assert np.array_equal(obs["pixels"][cam], on_demand["pixels"][cam])
    assert np.allclose(obs["robot_state"]["eef"]["pos"], on_demand["robot_state"]["eef"]["pos"], atol=1e-6)


def test_state_restore_reproduces_trajectory(session):
    """Save state, run actions, restore, re-run the same actions: same end-effector path and sim state."""
    rng = np.random.default_rng(1)
    session.reset(2)
    for a in np.clip(rng.normal(0, 0.3, size=(10, 7)), -1, 1):
        session.step(a)
    saved = session.get_state()
    tail = np.clip(rng.normal(0, 0.3, size=(25, 7)), -1, 1)
    for a in tail:
        session.step(a)
    end1 = session.get_state()
    session.set_state(saved)
    for a in tail:
        session.step(a)
    end2 = session.get_state()
    assert np.abs(end1 - end2).max() < 1e-3, np.abs(end1 - end2).max()
