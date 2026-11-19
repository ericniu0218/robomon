import numpy as np
import pytest

from robomon.monitors.signals import (
    consistency,
    dispersion,
    history_score,
    support_distance,
)


def test_dispersion_known_variance_and_horizon():
    chunks = np.zeros((2, 50, 7))
    chunks[0, :25] = -1
    chunks[1, :25] = 1
    chunks[:, 25:] = 1000
    assert dispersion(chunks) == pytest.approx(np.sqrt(2))
    assert dispersion(np.ones((8, 50, 7))) == 0
    assert dispersion(chunks + 17) == pytest.approx(dispersion(chunks))
    with pytest.raises(ValueError):
        dispersion(chunks[:1])


def test_consistency_aligns_overlap_and_does_not_pair_independent_samples():
    old = np.repeat(np.arange(50)[None, :, None], 2, axis=0)
    new = old + 25
    assert consistency(old, new) == 0
    new[:, :25] += 3
    assert consistency(old, new) == 3
    new[:, 25:] = 10000
    assert consistency(old, new[::-1]) == 3
    with pytest.raises(ValueError):
        consistency(old, new, offset=50)


def test_support_uses_nearest_demo_and_is_scale_invariant():
    bank = np.array([[1, 0], [0, 1]])
    assert support_distance(np.array([10, 0]), bank) == 0
    assert support_distance(np.array([-1, 0]), bank) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        support_distance(np.zeros(2), bank)
    with pytest.raises(ValueError):
        support_distance(np.ones(2), bank[:0])


def test_history_is_causal_and_missing_consistency_is_not_zero():
    history = np.array([np.nan, 3, 1, 8])
    assert np.isnan(history_score(history[:1], "last"))
    assert history_score(history[:3], "max") == 3
    assert history_score(history[:3], "mean") == 2
    assert history_score(history[:3], "last") == 1
