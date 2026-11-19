"""Training-free signals. Larger values mean greater hypothesized failure risk.

These are operational definitions for this audit, not reproductions of a paper's full detector.
Action distances use the checkpoint's normalized action coordinates, including the gripper.
"""
from __future__ import annotations

import numpy as np


def _chunks(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("chunks must be a finite, nonempty (samples, horizon, actions) array")
    return x


def dispersion(chunks: np.ndarray, horizon: int = 25) -> float:
    """RMS sample standard deviation over the next horizon actions (unbiased variance, K >= 2)."""
    x = _chunks(chunks)
    if x.shape[0] < 2 or not 1 <= horizon <= x.shape[1]:
        raise ValueError("dispersion requires K >= 2 and a horizon inside the chunk")
    return float(np.sqrt(x[:, :horizon].var(axis=0, ddof=1).mean()))


def consistency(previous: np.ndarray, current: np.ndarray, offset: int = 25) -> float:
    """RMS difference of sample means on the shared future interval.

    The previous query occurred `offset` environment steps before the current query. Samples are
    independent, so sample indices must not be treated as paired trajectories. For 50-step chunks
    queried every 25 steps this compares previous[:, 25:50] to current[:, 0:25].
    """
    a, b = _chunks(previous), _chunks(current)
    if a.shape[2] != b.shape[2] or not 0 < offset < a.shape[1]:
        raise ValueError("action dimensions must agree and offset must leave a nonempty overlap")
    overlap = min(a.shape[1] - offset, b.shape[1])
    delta = a[:, offset:offset + overlap].mean(axis=0) - b[:, :overlap].mean(axis=0)
    return float(np.sqrt(np.mean(delta ** 2)))


def support_distance(embedding: np.ndarray, demonstrations: np.ndarray) -> float:
    """Nearest-demo Euclidean distance after L2 normalization; bank must be for the SAME task.

    The caller must supply training demonstrations, never rollout/test observation embeddings.
    Normalization gives a distance in [0, 2] and removes embedding-norm scale as a nuisance.
    """
    q, bank = np.asarray(embedding, dtype=np.float64), np.asarray(demonstrations, dtype=np.float64)
    if q.ndim != 1 or bank.ndim != 2 or bank.shape[0] == 0 or bank.shape[1] != q.size:
        raise ValueError("expected one embedding and a nonempty bank with matching dimensions")
    if not np.isfinite(q).all() or not np.isfinite(bank).all():
        raise ValueError("embeddings must be finite")
    qnorm, norms = np.linalg.norm(q), np.linalg.norm(bank, axis=1)
    if qnorm == 0 or np.any(norms == 0):
        raise ValueError("zero embeddings cannot be normalized")
    return float(np.linalg.norm(bank / norms[:, None] - q / qnorm, axis=1).min())


def history_score(values: np.ndarray, reduction: str) -> float:
    """Causal history reduction; missing first-query consistency is ignored, never filled with 0."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or np.isinf(x).any():
        raise ValueError("history must be a vector without infinities")
    if reduction not in ("max", "mean", "last"):
        raise ValueError("reduction must be max, mean, or last")
    x = x[~np.isnan(x)]
    if not x.size:
        return float("nan")
    return float({"max": np.max, "mean": np.mean, "last": lambda a: a[-1]}[reduction](x))
