"""Slow integration tests (real SmolVLA + LIBERO on CPU). Run with: pytest -m slow"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("k", [1, 4, 8])
def test_shared_prefix_matches_reference(policy, session, k):
    obs = session.reset(0)
    a = policy.query(
        obs,
        session.task_description,
        k,
        torch.Generator().manual_seed(7),
        shared_prefix=True,
    )
    b = policy.query(
        obs,
        session.task_description,
        k,
        torch.Generator().manual_seed(7),
        shared_prefix=False,
    )
    assert a.chunks_norm.shape == (k, policy.chunk_size, policy.action_dim)
    assert np.abs(a.chunks_norm - b.chunks_norm).max() < 1e-3
    assert np.abs(a.exec_actions - b.exec_actions).max() < 1e-3
    assert np.abs(a.embedding - b.embedding).max() < 1e-4
    # different seeds give different samples; same seed is reproducible
    c = policy.query(
        obs,
        session.task_description,
        k,
        torch.Generator().manual_seed(8),
        shared_prefix=True,
    )
    assert np.abs(a.chunks_norm - c.chunks_norm).max() > 1e-3
    d = policy.query(
        obs,
        session.task_description,
        k,
        torch.Generator().manual_seed(7),
        shared_prefix=True,
    )
    assert np.allclose(a.chunks_norm, d.chunks_norm, atol=1e-5)


@pytest.mark.parametrize("n_steps", [1, 12, 25, 50])
def test_render_on_demand_matches_always_render(session, n_steps):
    """Scheduled images and state equal the always-render path at several clock phases."""
    rng = np.random.default_rng(0)
    actions = np.clip(rng.normal(0, 0.3, size=(n_steps, 7)), -1, 1)
    session.reset(1)
    for i, a in enumerate(actions):
        session.step(a, render=(i == n_steps - 1))
    on_demand = session.observe()
    session.reset(1)
    session._set_rendering(True)
    for a in actions:  # lerobot's own step renders every time
        obs, *_ = session.env.step(a.astype(np.float64))
    for cam in obs["pixels"]:
        assert np.array_equal(obs["pixels"][cam], on_demand["pixels"][cam])
    assert np.allclose(
        obs["robot_state"]["eef"]["pos"],
        on_demand["robot_state"]["eef"]["pos"],
        atol=1e-6,
    )


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


def test_observe_requires_a_scheduled_render(session):
    session.reset(0)
    session.step(np.zeros(7))
    with pytest.raises(RuntimeError, match="render=True"):
        session.observe()
    session.step(np.zeros(7), render=True)
    assert "pixels" in session.observe()


def test_skipped_steps_really_skip_rendering_after_repeated_resets(
    session, monkeypatch
):
    for _ in range(2):
        session.reset(0)
        calls = []
        original = session.rs.sim.render

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(session.rs.sim, "render", counted)
            for _ in range(5):
                session.step(np.zeros(7))
            assert not calls
            session.step(np.zeros(7), render=True)
            assert len(calls) == len(session._cams)


@pytest.mark.parametrize("condition", ["nominal", "shift"])
def test_monitor_does_not_change_real_rollout(policy, condition):
    from robomon.envs.libero_session import LiberoSession
    from robomon.rollout import EpisodeSpec, run_episode

    spec = EpisodeSpec(
        "libero_spatial",
        0,
        0,
        17,
        condition,
        brightness=0.6 if condition == "shift" else 1.0,
        noise_sigma=10.0 if condition == "shift" else 0.0,
    )
    session = LiberoSession("libero_spatial", 0)
    try:
        baseline = run_episode(policy, session, spec, monitor=False)
        monitored = run_episode(policy, session, spec, k=8, shared_prefix=True)
        assert baseline.summary["success"] == monitored.summary["success"]
        assert np.array_equal(baseline.arrays["actions"], monitored.arrays["actions"])
        assert np.array_equal(baseline.arrays["states"], monitored.arrays["states"])
    finally:
        session.close()
