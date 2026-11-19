"""Paired branch contrasts and conservative Monte Carlo bounds for FIXED selected states."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bounded_mean(values: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-sided Hoeffding bound for independent seed-block differences in [-1, 1].

    P(|mean - E mean| >= r) <= 2 exp(-n r^2 / 2), so r=sqrt(2 log(2/alpha)/n).
    Blocks need not have the same distribution. This quantifies branch-noise Monte Carlo error
    conditional on the selected states; it does NOT quantify selection or task generalization.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError("Nonempty finite differences in [-1, 1] required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie inside (0, 1)")
    estimate = float(x.mean())
    radius = float(np.sqrt(2 * np.log(2 / alpha) / len(x)))
    return {
        "estimate": estimate,
        "ci_low": max(-1.0, estimate - radius),
        "ci_high": min(1.0, estimate + radius),
        "radius_unclipped": radius,
        "alpha": alpha,
        "independent_seed_blocks": len(x),
        "positive": int((x > 0).sum()),
        "zero": int((x == 0).sum()),
        "negative": int((x < 0).sum()),
        "interval": "Hoeffding; fixed selected states; Monte Carlo error only",
    }


def summarize_branches(
    records: pd.DataFrame, replicates: int, alpha: float = 0.05
) -> dict:
    """Common seeds pair all four state/arm outcomes; never count four arms as independent."""
    data = records.copy()
    if replicates < 1 or data.empty or not data.kind.eq("intervention").all():
        raise ValueError(
            "Complete intervention records and positive replicate count required"
        )
    if data.success.isna().any() or not data.success.isin([True, False]).all():
        raise ValueError("Binary success required")
    if (
        not data.role.isin(["alarm", "control"]).all()
        or not data.rewind.isin([0, 50]).all()
    ):
        raise ValueError(
            "Expected alarm/control states and replan/50-step rollback arms"
        )
    keys = ["match_id", "replicate", "role", "rewind"]
    if data[keys + ["branch_seed"]].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Duplicate or missing branch identities")
    if (
        (
            data.groupby("match_id")[
                ["task", "condition", "landmark", "horizon"]
            ].nunique()
            != 1
        )
        .any()
        .any()
    ):
        raise ValueError(
            "Each matched set must share task, condition, landmark and horizon"
        )
    for _, group in data.groupby(["match_id", "role"]):
        if group.episode_id.nunique() != 1 or group.pair_id.nunique() != 1:
            raise ValueError("One fixed episode state per role required")
    states = data[["match_id", "role", "pair_id"]].drop_duplicates()
    if states.pair_id.duplicated().any():
        raise ValueError(
            "Source episode pairs cannot be reused across matched sets/conditions"
        )
    if not data.replicate.isin(range(replicates)).all():
        raise ValueError("Unexpected replicate index")
    if not (
        (data.budget == data.horizon - data.landmark)
        & (data.forward_steps > 0)
        & (data.forward_steps <= data.budget)
    ).all():
        raise ValueError("Unequal or invalid forward-action budget")
    if not (data.restored_step == data.landmark - data.rewind).all():
        raise ValueError("Wrong restored step")
    grouped = data.groupby(["match_id", "replicate"], sort=True)
    if (grouped.size() != 4).any() or len(
        grouped
    ) != data.match_id.nunique() * replicates:
        raise ValueError("Missing state/arm/replicate outcomes")
    if (grouped.branch_seed.nunique() != 1).any():
        raise ValueError("Common random numbers required within a seed block")
    if grouped.branch_seed.first().duplicated().any():
        raise ValueError("Independent seed blocks must have distinct seeds")
    data["arm"] = data.role + "_" + data.rewind.map({0: "replan", 50: "rollback"})
    wide = data.pivot(
        index=["match_id", "replicate"], columns="arm", values="success"
    ).astype(int)
    contrasts = {
        "h4a_control_minus_alarm_replan": wide.control_replan - wide.alarm_replan,
        "h4b_alarm_rollback_minus_replan": wide.alarm_rollback - wide.alarm_replan,
        "control_rollback_minus_replan_exploratory": wide.control_rollback
        - wide.control_replan,
    }
    return {
        "status": "development only",
        "matched_sets": int(data.match_id.nunique()),
        "states": len(states),
        "replicates_per_state_arm": replicates,
        "branches": len(data),
        "target": "equal-weight mean over fixed selected matched states and fresh policy/pixel noise",
        "rates": {column: float(wide[column].mean()) for column in wide},
        "contrasts": {
            name: bounded_mean(values.to_numpy(), alpha)
            for name, values in contrasts.items()
        },
        "by_cell": [
            {
                "task": task,
                "condition": condition,
                "role": role,
                "rewind": int(rewind),
                "branches": len(group),
                "successes": int(group.success.sum()),
            }
            for (task, condition, role, rewind), group in data.groupby(
                ["task", "condition", "role", "rewind"], sort=True
            )
        ],
    }
