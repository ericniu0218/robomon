"""Pair-weighted concordance at prespecified landmarks with episode-cluster uncertainty.

Failure=1. Ties count as half; strata with only one outcome contain no comparable pairs and are
reported, not assigned 0.5. The primary estimator pools all failure/success pairs within the same
(task, condition, landmark). It is not an unweighted average of stratum AUROCs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Concordance:
    estimate: float
    pairs: float
    informative_strata: int
    total_strata: int
    rows: int
    missing_scores: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(frame: pd.DataFrame, score: str, strata: tuple[str, ...]) -> pd.DataFrame:
    required = set(strata) | {"episode_id", "task", "condition", "landmark", "failure", score}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    keys = ["episode_id", "task", "condition", "landmark", "failure"]
    if frame[keys].isna().any().any() or not frame.failure.isin([0, 1]).all():
        raise ValueError("keys and binary failure labels must be nonmissing")
    if frame.duplicated(["episode_id", "landmark"]).any():
        raise ValueError("duplicate episode/landmark: repeated records would inflate precision")
    if (frame.groupby("episode_id")[["task", "condition", "failure"]].nunique() > 1).any().any():
        raise ValueError("each episode must have one task, condition, and terminal label")
    if np.isinf(frame[score].to_numpy(dtype=float)).any():
        raise ValueError("infinite scores are invalid")
    return frame.loc[frame[score].notna()].copy()


def _groups(frame: pd.DataFrame, strata: tuple[str, ...]):
    return frame.groupby(list(strata), sort=True, observed=True) if strata else [((), frame)]


def _pair_table(frame: pd.DataFrame, score: str, strata: tuple[str, ...], episode_indices: dict):
    left, right, wins = [], [], []
    informative = 0
    for _, group in _groups(frame, strata):
        failed, passed = group[group.failure == 1], group[group.failure == 0]
        if failed.empty or passed.empty:
            continue
        informative += 1
        a, b = failed[score].to_numpy(), passed[score].to_numpy()
        wins.extend(((a[:, None] > b) + 0.5 * (a[:, None] == b)).ravel())
        left.extend(np.repeat([episode_indices[e] for e in failed.episode_id], len(passed)))
        right.extend(np.tile([episode_indices[e] for e in passed.episode_id], len(failed)))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int), np.asarray(wins), informative


def concordance(frame: pd.DataFrame, score: str,
                strata: tuple[str, ...] = ("task", "condition", "landmark")) -> Concordance:
    data = _validate(frame, score, strata)
    ids = {e: i for i, e in enumerate(data.episode_id.unique())}
    _, _, wins, n = _pair_table(data, score, strata, ids)
    total = len(data.groupby(list(strata), observed=True)) if strata else int(len(data) > 0)
    return Concordance(float(wins.mean()) if wins.size else float("nan"), float(wins.size),
                       n, total, len(data), len(frame) - len(data))


def bootstrap_concordance(frame: pd.DataFrame, score: str,
                          strata: tuple[str, ...] = ("task", "condition", "landmark"), *,
                          n_boot: int = 2000, seed: int = 0,
                          cluster: str = "episode_id") -> dict:
    """Percentile CI, resampling entire trajectories within each task.

    `cluster='episode_id'` implements the plan literally. For nominal/shift episodes paired by
    initial state and policy seed, pass a shared `pair_id`: all conditions and landmarks then get
    the same multiplicity. Missing scores remain missing; no-outcome resamples are counted.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    data = _validate(frame, score, strata)
    if cluster not in frame or frame[cluster].isna().any():
        raise ValueError("cluster IDs must be present and nonmissing")
    if (frame.groupby(cluster).task.nunique() > 1).any():
        raise ValueError("a cluster cannot cross tasks")
    if (frame.groupby("episode_id")[cluster].nunique() > 1).any():
        raise ValueError("an episode cannot cross clusters")
    episodes = frame[["episode_id", "task", cluster]].loc[:, lambda x: ~x.columns.duplicated()]
    episodes = episodes.drop_duplicates("episode_id").reset_index(drop=True)
    indices = {e: i for i, e in enumerate(episodes.episode_id)}
    left, right, wins, _ = _pair_table(data, score, strata, indices)
    cluster_ids = list(episodes[cluster].unique())
    cindex = {c: i for i, c in enumerate(cluster_ids)}
    episode_clusters = np.array([cindex[c] for c in episodes[cluster]])
    blocks = [np.array([cindex[c] for c in group[cluster].unique()])
              for _, group in episodes.groupby("task", sort=True)]
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(n_boot):
        counts = np.zeros(len(cluster_ids), dtype=int)
        for block in blocks:
            counts += np.bincount(rng.choice(block, size=len(block), replace=True), minlength=len(counts))
        weights = counts[episode_clusters]
        pair_weights = weights[left] * weights[right]
        denominator = pair_weights.sum()
        if denominator:
            estimates.append(float(np.sum(pair_weights * wins) / denominator))
    bounds = np.quantile(estimates, [0.025, 0.975]) if estimates else [float("nan")] * 2
    return {**concordance(frame, score, strata).to_dict(), "ci_low": float(bounds[0]),
            "ci_high": float(bounds[1]), "bootstrap_valid": len(estimates),
            "bootstrap_degenerate": n_boot - len(estimates), "n_boot": n_boot,
            "seed": seed, "cluster": cluster}


def episode_max(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    """Naive analysis input: one full-episode maximum, including episode length confounding."""
    data = _validate(frame, score, ())
    keys = ["episode_id", "task", "condition", "failure"]
    if "pair_id" in data:
        if data.pair_id.isna().any() or (data.groupby("episode_id").pair_id.nunique() != 1).any():
            raise ValueError("Every episode must have one nonmissing pair_id")
        keys.append("pair_id")
    result = data.groupby(keys, as_index=False)[score].max()
    result["landmark"] = 0
    return result
