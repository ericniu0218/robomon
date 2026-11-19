import numpy as np
import pandas as pd
import pytest

from robomon.analysis.interventions import bounded_mean, summarize_branches
from robomon.interventions import (
    boundary_histories,
    calibrate_alarm,
    match_alarm_states,
)
from robomon.utils.artifacts import completed_array_result, save_array_result


def histories():
    rows = []
    for pair in range(8):
        for condition in ("nominal", "shift"):
            values = [np.nan, 2.0 if pair % 2 == 0 else 0.2, 0.1, 0.3, 0.4, 0.2]
            for step, value in enumerate(values):
                rows.append(
                    {
                        "episode_id": f"e{pair}/{condition}",
                        "pair_id": f"pair{pair}",
                        "task": "libero_spatial/0",
                        "condition": condition,
                        "landmark": step * 25,
                        "consistency": value,
                        "failure": pair % 2,
                        "support": 100 * pair,
                    }
                )
    return pd.DataFrame(rows)


def test_selection_is_label_blind_order_invariant_and_disjoint_across_conditions():
    raw = histories()
    expected = match_alarm_states(boundary_histories(raw), 1.0, max_pairs_per_cell=2)
    changed = raw.sample(frac=1, random_state=7).copy()
    changed["failure"] = 1 - changed.failure
    changed["support"] *= -999
    assert (
        match_alarm_states(boundary_histories(changed), 1.0, max_pairs_per_cell=2)
        == expected
    )
    assert (
        match_alarm_states(
            boundary_histories(raw.drop(columns=["failure", "support"])),
            1.0,
            max_pairs_per_cell=2,
        )
        == expected
    )
    assert len(expected["matches"]) == 4
    used = []
    for match in expected["matches"]:
        alarm, control = match["alarm"], match["control"]
        assert alarm["score"] >= 1.0 > control["score"]
        assert alarm["landmark"] == control["landmark"] == 50
        assert (alarm["task"], alarm["condition"]) == (
            control["task"],
            control["condition"],
        )
        used.extend([alarm["pair_id"], control["pair_id"]])
    assert len(used) == len(set(used))


def test_histories_include_shadow_queries_causally_and_leave_forward_budget():
    raw = histories().query("episode_id == 'e0/nominal'").copy()
    raw["consistency"] = [np.nan, 0.3, 0.2, 0.9, 0.1, 0.2]
    boundaries = boundary_histories(raw)
    assert boundaries.score.tolist() == [0.3, 0.9]
    raw.loc[raw.landmark == 125, "consistency"] = 999
    pd.testing.assert_frame_equal(boundary_histories(raw), boundaries)
    extra = pd.concat(
        [
            raw,
            *[
                raw.iloc[[-1]].assign(landmark=step, consistency=999)
                for step in range(150, 276, 25)
            ],
        ]
    )
    assert boundary_histories(extra).landmark.tolist() == [50, 100, 150, 200]


def test_calibration_weights_episodes_equally_and_missing_controls_are_not_replaced():
    data = pd.DataFrame(
        [
            {"episode_id": episode, "condition": "nominal", "score": score}
            for episode, count, score in [
                ("short", 1, 10.0),
                ("long", 8, 0.0),
                ("middle", 2, 2.0),
            ]
            for _ in range(count)
        ]
    )
    calibration = calibrate_alarm(data, 0.5)
    assert calibration["threshold"] == 2.0 and calibration["episodes"] == 3
    boundaries = boundary_histories(histories())
    none = match_alarm_states(boundaries, 0.0)
    assert none["matches"] == []
    assert sum(row["first_alarms"] for row in none["coverage"]) == 16
    assert sum(row["alarms_with_exact_controls"] for row in none["coverage"]) == 0
    assert match_alarm_states(boundaries, 1000)["matches"] == []


@pytest.mark.parametrize("damage", ["heldout", "missing", "gap", "duplicate"])
def test_selection_rejects_invalid_or_heldout_histories(damage):
    raw = histories()
    if damage == "heldout":
        raw["task"] = "libero_spatial/1"
    elif damage == "missing":
        raw.loc[raw.landmark == 50, "consistency"] = np.nan
    elif damage == "gap":
        raw = raw[raw.landmark != 25]
    else:
        raw = pd.concat([raw, raw.iloc[[0]]])
    with pytest.raises(ValueError):
        boundary_histories(raw)


def branch_records():
    rows = []
    for match in range(2):
        for replicate in range(3):
            for role in ("alarm", "control"):
                for rewind in (0, 50):
                    rows.append(
                        {
                            "match_id": f"m{match}",
                            "role": role,
                            "kind": "intervention",
                            "replicate": replicate,
                            "rewind": rewind,
                            "branch_seed": 100 * match + replicate,
                            "task": "libero_spatial/0",
                            "condition": "nominal",
                            "landmark": 50,
                            "horizon": 280,
                            "episode_id": f"e{match}/{role}",
                            "pair_id": f"p{match}/{role}",
                            "success": role == "control" or rewind == 50,
                            "budget": 230,
                            "forward_steps": 20
                            if role == "control" or rewind == 50
                            else 230,
                            "restored_step": 50 - rewind,
                        }
                    )
    return pd.DataFrame(rows)


def test_branch_analysis_pairs_noise_and_arms_and_does_not_count_arms_as_samples():
    report = summarize_branches(branch_records(), 3)
    assert report["branches"] == 24 and report["states"] == 4
    for contrast in (
        "h4a_control_minus_alarm_replan",
        "h4b_alarm_rollback_minus_replan",
    ):
        result = report["contrasts"][contrast]
        assert result["estimate"] == 1.0
        assert result["independent_seed_blocks"] == result["positive"] == 6
        assert result["ci_high"] == 1.0
    assert (
        report["contrasts"]["control_rollback_minus_replan_exploratory"]["estimate"]
        == 0.0
    )
    # Degenerate observed noise must not lead to zero-width uncertainty.
    assert bounded_mean(np.zeros(30))["ci_high"] > 0.4


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "budget", "noise", "seed_reuse", "state_reuse"]
)
def test_branch_analysis_rejects_incomplete_or_unpaired_experiments(damage):
    rows = branch_records()
    if damage == "missing":
        rows = rows.iloc[1:]
    elif damage == "duplicate":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif damage == "budget":
        rows.loc[0, "budget"] += 50
    elif damage == "noise":
        rows.loc[0, "branch_seed"] = 999
    elif damage == "seed_reuse":
        rows["branch_seed"] = 1
    else:
        rows["pair_id"] = "same"
    with pytest.raises(ValueError):
        summarize_branches(rows, 3)


def test_branch_transaction_detects_config_changes_and_array_corruption(tmp_path):
    config = {"seeds": [11, 12, 13]}
    assert not completed_array_result(tmp_path, config)
    save_array_result(
        {"success": False}, {"states": np.zeros((3, 7))}, tmp_path, config
    )
    assert completed_array_result(tmp_path, config)
    with pytest.raises(ValueError, match="config mismatch"):
        completed_array_result(tmp_path, {"seeds": [1, 2, 3]})
    (tmp_path / "arrays.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="damaged"):
        completed_array_result(tmp_path, config)


def test_monte_carlo_bound_has_conservative_exact_coverage_on_binary_extremes():
    from scipy.stats import binom

    # Enumerate, rather than simulate, the distribution of independent {-1,+1} differences.
    for count in (3, 12, 40):
        for probability in (0.05, 0.2, 0.5, 0.8, 0.95):
            truth = 2 * probability - 1
            covered = 0.0
            for positive in range(count + 1):
                difference = np.r_[np.ones(positive), -np.ones(count - positive)]
                interval = bounded_mean(difference)
                if interval["ci_low"] <= truth <= interval["ci_high"]:
                    covered += float(binom.pmf(positive, count, probability))
            assert covered >= 0.95
