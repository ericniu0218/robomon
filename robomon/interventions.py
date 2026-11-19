"""Outcome-blind alarm calibration and disjoint, exactly matched states.

Only STATE_COLUMNS and one explicitly chosen raw signal enter selection. Terminal outcomes,
other signals, future branch outcomes and file ordering cannot choose the states.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from robomon.data.demos import dev_task_keys
from robomon.monitors.signals import history_score
from robomon.utils.io import config_hash

# LeRobot LIBERO benchmark horizons, checked against the installed integration. The branch
# runner also checks the real session's horizon before it executes any selected state.
HORIZONS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
STATE_COLUMNS = ["episode_id", "pair_id", "task", "condition", "landmark"]


def boundary_histories(
    rows: pd.DataFrame,
    *,
    signal: str = "consistency",
    reduction: str = "max",
    allowed_tasks: list[str] | None = None,
) -> pd.DataFrame:
    """Causal frozen score at live chunk boundaries with >=50 actions remaining.

    Histories include the intervening shadow queries. Only the first query may have missing
    consistency; no zero imputation or later observation is used for a boundary's score.
    """
    if signal not in ("consistency", "dispersion", "support") or reduction not in (
        "max",
        "mean",
        "last",
    ):
        raise ValueError("Unknown alarm signal or reduction")
    allowed_tasks = dev_task_keys() if allowed_tasks is None else allowed_tasks
    data = rows[STATE_COLUMNS + [signal]].copy()
    if data[STATE_COLUMNS].isna().any().any() or data.empty:
        raise ValueError("Nonempty, nonmissing state identities required")
    if not data.task.isin(allowed_tasks).all():
        raise ValueError("Intervention selection cannot inspect tasks outside the allowed partition")
    if not data.condition.isin(["nominal", "shift"]).all():
        raise ValueError("Unknown condition")
    if data.duplicated(["episode_id", "landmark"]).any():
        raise ValueError("Repeated episode/landmark")
    if (
        (data.groupby("episode_id")[["task", "condition", "pair_id"]].nunique() != 1)
        .any()
        .any()
    ):
        raise ValueError("An episode must have one task, condition and pair ID")
    if (data.groupby("pair_id").task.nunique() != 1).any() or (
        data.groupby(["pair_id", "condition"]).episode_id.nunique() != 1
    ).any():
        raise ValueError("A source pair must have one episode per task/condition")
    if not np.isfinite(data.landmark).all() or (data.landmark < 0).any():
        raise ValueError("Landmarks must be finite, nonnegative integers")
    data = data.sort_values(["episode_id", "landmark"]).reset_index(drop=True)
    parts = []
    for _, episode in data.groupby("episode_id", sort=True):
        if not np.array_equal(episode.landmark, np.arange(len(episode)) * 25):
            raise ValueError(
                "Expected complete 25-action query histories starting at zero"
            )
        values = episode[signal].to_numpy(float)
        if (
            (signal == "consistency" and not np.isnan(values[0]))
            or not np.isfinite(values[1:]).all()
            or (signal != "consistency" and not np.isfinite(values[0]))
        ):
            raise ValueError("Only first-query consistency may be missing")
        episode = episode.copy()
        episode["score"] = [
            history_score(values[: i + 1], reduction) for i in range(len(values))
        ]
        episode["horizon"] = HORIZONS[episode.task.iloc[0].split("/")[0]]
        if (episode.landmark >= episode.horizon).any():
            raise ValueError("A query cannot be at or beyond the benchmark horizon")
        parts.append(episode)
    data = pd.concat(parts, ignore_index=True)
    return data.loc[
        (data.landmark >= 50)
        & (data.landmark % 50 == 0)
        & (data.landmark <= data.horizon - 50),
        STATE_COLUMNS + ["score", "horizon"],
    ].reset_index(drop=True)


def calibrate_alarm(
    boundaries: pd.DataFrame,
    quantile: float = 0.75,
    *,
    signal: str = "consistency",
    reduction: str = "max",
) -> dict:
    """One nominal-episode maximum per eligible episode, regardless of its terminal outcome."""
    if not 0 < quantile < 1:
        raise ValueError("Alarm quantile must be strictly inside (0, 1)")
    maxima = (
        boundaries[boundaries.condition == "nominal"].groupby("episode_id").score.max()
    )
    if maxima.empty or not np.isfinite(maxima).all():
        raise ValueError("Finite nominal episode maxima required for calibration")
    threshold = float(maxima.quantile(quantile, interpolation="higher"))
    return {
        "signal": signal,
        "reduction": "causal running maximum including shadow queries"
        if reduction == "max"
        else reduction,
        "quantile": quantile,
        "interpolation": "higher",
        "population": "one maximum per eligible nominal dev episode, all outcomes",
        "episodes": len(maxima),
        "threshold": threshold,
        "comparison": ">=",
        "nominal_episode_alarm_fraction": float((maxima >= threshold).mean()),
    }


def _priority(seed: int, *values) -> str:
    return hashlib.sha256(
        (str(seed) + "/" + "/".join(map(str, values))).encode()
    ).hexdigest()


def match_alarm_states(
    boundaries: pd.DataFrame,
    threshold: float,
    *,
    max_pairs_per_cell: int = 1,
    seed: int = 20261111,
    allowed_tasks: list[str] | None = None,
) -> dict:
    """Greedy hashed-order matching, without replacement across source nominal/shift pair IDs.

    Each alarm episode contributes only its first eligible crossing. Controls are still live,
    below threshold at the same task/condition/landmark, and cannot be reused in either condition.
    This is a declared finite selected-state population, not uniform sampling of all robot states.
    """
    if not np.isfinite(threshold) or max_pairs_per_cell < 1:
        raise ValueError("Finite threshold and positive cell cap required")
    data = boundaries[STATE_COLUMNS + ["score", "horizon"]].copy()
    allowed_tasks = dev_task_keys() if allowed_tasks is None else allowed_tasks
    if not data.task.isin(allowed_tasks).all():
        raise ValueError("Intervention selection cannot inspect tasks outside the allowed partition")
    if (
        data.duplicated(["episode_id", "landmark"]).any()
        or not np.isfinite(data.score).all()
    ):
        raise ValueError("Unique finite boundary scores required")
    data = data.sort_values(["episode_id", "landmark"])
    data["alarm_seen"] = data.score.ge(threshold).groupby(data.episode_id).cummax()
    first = (
        data[data.score >= threshold]
        .drop_duplicates("episode_id", keep="first")
        .drop(columns="alarm_seen")
    )
    candidates, matchable = [], {}
    for alarm in first.to_dict("records"):
        controls = (
            data[
                data.task.eq(alarm["task"])
                & data.condition.eq(alarm["condition"])
                & data.landmark.eq(alarm["landmark"])
                & ~data.alarm_seen
                & data.pair_id.ne(alarm["pair_id"])
            ]
            .drop(columns="alarm_seen")
            .to_dict("records")
        )
        cell = (alarm["task"], alarm["condition"])
        matchable[cell] = matchable.get(cell, 0) + bool(controls)
        controls.sort(
            key=lambda row: _priority(
                seed, "control", alarm["episode_id"], row["episode_id"]
            )
        )
        if controls:
            candidates.append((alarm, controls))
    candidates.sort(key=lambda item: _priority(seed, "alarm", item[0]["episode_id"]))
    used, counts, selected = set(), {}, []
    for alarm, controls in candidates:
        cell = (alarm["task"], alarm["condition"])
        if counts.get(cell, 0) >= max_pairs_per_cell or alarm["pair_id"] in used:
            continue
        control = next((row for row in controls if row["pair_id"] not in used), None)
        if control is None:
            continue
        identity = {
            "alarm": alarm["episode_id"],
            "control": control["episode_id"],
            "landmark": alarm["landmark"],
        }
        selected.append(
            {"match_id": config_hash(identity), "alarm": alarm, "control": control}
        )
        used.update([alarm["pair_id"], control["pair_id"]])
        counts[cell] = counts.get(cell, 0) + 1
    coverage = []
    for task, condition in [
        (task, condition)
        for task in sorted(allowed_tasks)
        for condition in ("nominal", "shift")
    ]:
        group = data[data.task.eq(task) & data.condition.eq(condition)]
        alarms = first[first.task.eq(task) & first.condition.eq(condition)]
        coverage.append(
            {
                "task": task,
                "condition": condition,
                "eligible_episodes": int(group.episode_id.nunique()),
                "first_alarms": len(alarms),
                "alarms_with_exact_controls": matchable.get((task, condition), 0),
                "selected_pairs": counts.get((task, condition), 0),
            }
        )
    return {
        "seed": seed,
        "max_pairs_per_cell": max_pairs_per_cell,
        "no_source_pair_reuse_across_conditions": True,
        "coverage": coverage,
        "matches": sorted(
            selected,
            key=lambda row: (
                row["alarm"]["task"],
                row["alarm"]["condition"],
                row["match_id"],
            ),
        ),
    }
