import numpy as np
import pytest

from robomon.branching import replay_prefix, run_branch
from robomon.rollout import EpisodeSpec, run_episode
from test_rollout import ToyPolicy, ToySession


def test_known_answer_tail_matches_and_rewind_does_not_receive_extra_steps():
    spec = EpisodeSpec("libero_spatial", 0, 0, 17, "shift", 0.6, 10.0)
    trunk = run_episode(ToyPolicy(), ToySession(), spec, monitor=True)
    continued = run_branch(
        ToyPolicy(),
        ToySession(),
        spec,
        trunk.arrays["actions"],
        trunk.arrays["states"],
        checkpoint=50,
    )
    assert np.array_equal(continued["actions"], trunk.arrays["actions"][50:])
    assert np.array_equal(continued["states"], trunk.arrays["states"][50:])
    rollback = run_branch(
        ToyPolicy(),
        ToySession(),
        spec,
        trunk.arrays["actions"],
        trunk.arrays["states"],
        checkpoint=50,
        rewind=50,
        seed=101,
    )
    assert rollback["restored_step"] == 0 and continued["restored_step"] == 50
    assert rollback["budget"] == continued["budget"] == 70
    assert rollback["forward_steps"] == 70


def test_counterfactual_rejects_inexact_prefix_and_terminal_checkpoint():
    spec = EpisodeSpec("libero_spatial", 0, 0, 17)
    trunk = run_episode(ToyPolicy(), ToySession(), spec, monitor=False)
    damaged = trunk.arrays["states"].copy()
    damaged[10, 1] += 1
    with pytest.raises(RuntimeError, match="differs"):
        replay_prefix(ToySession(), spec, trunk.arrays["actions"], damaged, 50)
    with pytest.raises(ValueError, match="terminal"):
        replay_prefix(ToySession(), spec, trunk.arrays["actions"], damaged, 120)
    with pytest.raises(ValueError, match="boundaries"):
        run_branch(
            ToyPolicy(),
            ToySession(),
            spec,
            trunk.arrays["actions"],
            trunk.arrays["states"],
            checkpoint=25,
        )


def test_libero_replay_capability_requires_explicit_reset_context():
    spec = EpisodeSpec("libero_spatial", 0, 0, 17)
    trunk = run_episode(ToyPolicy(), ToySession(), spec, monitor=False)
    restored = ToySession()
    restored.reset_for_replay = lambda initial, history: restored.reset(initial)
    with pytest.raises(ValueError, match="Explicit reset history"):
        replay_prefix(
            restored, spec, trunk.arrays["actions"], trunk.arrays["states"], 50
        )
    replay_prefix(
        restored,
        spec,
        trunk.arrays["actions"],
        trunk.arrays["states"],
        50,
        reset_history=(),
    )
    assert np.array_equal(restored.get_state(), trunk.arrays["states"][49])
