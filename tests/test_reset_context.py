"""A stateful external-reset model reproduces LIBERO's fixture-sampler accumulation bug."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_rollout import ToyPolicy, ToySession

from robomon.branching import replay_prefix
from robomon.envs.libero_session import LiberoSession
from robomon.rollout import EpisodeSpec, run_episode


class StatefulNativeEnv:
    def __init__(self):
        self._env = None

    def close(self):
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = SimpleNamespace(
                env=SimpleNamespace(
                    object_property_initializers=[0, 0],
                    _observables={},
                    sim=SimpleNamespace(
                        model=SimpleNamespace(body_pos=np.zeros((1, 3)))
                    ),
                )
            )

    def reset(self, seed):
        self._ensure_env()
        rs = self._env.env
        rs.object_property_initializers.append(0)
        # One extra RNG draw per leaked initializer changes the static fixture position.
        rs.sim.model.body_pos[0, 0] = np.random.default_rng(seed).random(
            len(rs.object_property_initializers)
        )[-1]
        return {"fixture": rs.sim.model.body_pos.copy()}, {}


def session(mode="native"):
    value = LiberoSession.__new__(LiberoSession)
    value.env = StatefulNativeEnv()
    value.reset_mode, value.render_mode, value.n_init_states = mode, "always", 50
    value._camera_sensors, value._cams = {}, []
    value._observation, value._rendering = None, True
    value.reset_history, value.last_reset_context = [], None
    value._replay_reset_key, value._replay_initializers = None, ()
    return value


def test_native_seed_does_not_restore_fixture_but_reset_context_does():
    original = session()
    first = original.reset(0)
    second = original.reset(0)
    assert not np.array_equal(first["fixture"], second["fixture"])
    original.reset(1)
    expected = original.reset(1)
    context = original.last_reset_context
    assert context["prior_init_states"] == [0, 0, 1]
    replay = session()
    for _ in range(3):
        observed = replay.reset_for_replay(1, tuple(context["prior_init_states"]))
        assert np.array_equal(observed["fixture"], expected["fixture"])
        assert replay.last_reset_context == context
    # A different declared reset history must rebuild the context, not reuse the old one.
    assert not np.array_equal(
        replay.reset_for_replay(1, ())["fixture"], expected["fixture"]
    )


def test_isolated_resets_have_identical_fixture_and_empty_prior_context():
    isolated = session("isolated")
    observations = [isolated.reset(3) for _ in range(3)]
    assert all(
        np.array_equal(row["fixture"], observations[0]["fixture"])
        for row in observations
    )
    assert isolated.last_reset_context == {"mode": "isolated", "prior_init_states": []}
    assert np.array_equal(
        isolated.reset_for_replay(3, ())["fixture"], observations[0]["fixture"]
    )
    with pytest.raises(ValueError, match="empty prior"):
        isolated.reset_for_replay(3, (0,))


def test_replay_rejects_static_model_mismatch_even_when_physics_state_matches():
    spec = EpisodeSpec("libero_spatial", 0, 0, 17)
    trunk = run_episode(ToyPolicy(), ToySession(), spec, monitor=False)
    restored = ToySession()
    restored.rs = SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(body_pos=np.zeros((1, 3))))
    )
    with pytest.raises(RuntimeError, match="Reset model differs"):
        replay_prefix(
            restored,
            spec,
            trunk.arrays["actions"],
            trunk.arrays["states"],
            50,
            reset_model={"body_pos": np.ones((1, 3))},
        )


def test_replay_rejects_initial_physics_mismatch_before_any_action():
    spec = EpisodeSpec("libero_spatial", 0, 0, 17)
    trunk = run_episode(ToyPolicy(), ToySession(), spec, monitor=False)
    with pytest.raises(RuntimeError, match="Initial physics differs"):
        replay_prefix(
            ToySession(),
            spec,
            trunk.arrays["actions"],
            trunk.arrays["states"],
            50,
            reset_physics=np.array([0.0, 1.0]),
        )


def test_rollout_saves_initial_physics_and_static_scene_without_aliasing():
    class RecordedSession(ToySession):
        def reset(self, initial):
            obs = super().reset(initial)
            self.last_reset_context = {"mode": "isolated", "prior_init_states": []}
            self.rs = SimpleNamespace(
                sim=SimpleNamespace(
                    model=SimpleNamespace(
                        body_pos=np.ones((1, 3)),
                        body_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                    )
                )
            )
            return obs

    environment = RecordedSession()
    result = run_episode(
        ToyPolicy(), environment, EpisodeSpec("libero_spatial", 0, 0, 17)
    )
    assert result.summary["reset_context"] == {
        "mode": "isolated",
        "prior_init_states": [],
    }
    assert np.array_equal(result.arrays["reset_state"], [0.0, 0.0])
    environment.rs.sim.model.body_pos[:] = 10
    assert np.array_equal(result.arrays["reset_body_pos"], np.ones((1, 3)))


@pytest.mark.slow
def test_real_stove_fixture_reconstructs_history_and_isolates_episode_resets():
    prior = (0, 0, 1, 1, 2, 2)
    original = LiberoSession("libero_10", 8)
    try:
        for initial in prior:
            original.reset(initial)
        original.reset(3)
        expected = {
            key: getattr(original.rs.sim.model, key).copy()
            for key in ("body_pos", "body_quat")
        }
    finally:
        original.close()
    reconstructed = LiberoSession("libero_10", 8)
    try:
        for _ in range(2):
            reconstructed.reset_for_replay(3, prior)
            for key, value in expected.items():
                assert np.array_equal(getattr(reconstructed.rs.sim.model, key), value)
    finally:
        reconstructed.close()
    isolated = LiberoSession("libero_10", 8, reset_mode="isolated")
    try:
        first = isolated.reset(3)
        poses = isolated.rs.sim.model.body_pos.copy()
        physics = isolated.get_state()
        second = isolated.reset_for_replay(3, ())
        assert np.array_equal(isolated.rs.sim.model.body_pos, poses)
        assert np.array_equal(isolated.get_state(), physics)
        assert isolated.last_reset_context == {
            "mode": "isolated",
            "prior_init_states": [],
        }
        for camera in first["pixels"]:
            assert np.array_equal(first["pixels"][camera], second["pixels"][camera])
    finally:
        isolated.close()
