import numpy as np
import pandas as pd
import pytest

from robomon.analysis.landmarks import bootstrap_concordance, concordance, episode_max


def confounded_data():
    rows = []
    # A hard task fails 80%; an easy task fails 20%. Failures run longer.
    for task, n_fail in [("hard", 16), ("easy", 4)]:
        for episode in range(20):
            failure = int(episode < n_fail)
            for t in range(1, 6 if failure else 4):
                rows.append(
                    dict(
                        episode_id=f"{task}/{episode}",
                        task=task,
                        condition="nominal",
                        landmark=t,
                        failure=failure,
                        elapsed=t,
                        difficulty=int(task == "hard"),
                        planted=failure,
                    )
                )
    return pd.DataFrame(rows)


def test_time_confound_disappears_with_landmark_matching():
    data = confounded_data()
    assert concordance(episode_max(data, "elapsed"), "elapsed", ()).estimate == 1
    assert concordance(data, "elapsed", ("landmark",)).estimate == 0.5
    assert concordance(data, "elapsed").estimate == 0.5


def test_task_confound_disappears_only_with_task_matching():
    data = confounded_data()
    assert concordance(data, "difficulty", ("landmark",)).estimate > 0.75
    matched = concordance(data, "difficulty")
    assert matched.estimate == 0.5
    assert matched.informative_strata == 6
    assert matched.total_strata == 10
    assert matched.pairs == 2 * 3 * 16 * 4


def test_recovers_planted_signal_and_cluster_ci():
    data = confounded_data()
    result = bootstrap_concordance(data, "planted", n_boot=100, seed=10)
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 1
    null = bootstrap_concordance(data, "elapsed", n_boot=100, seed=10)
    assert null["ci_low"] == null["ci_high"] == 0.5


def test_soft_planted_effect_and_null():
    rng = np.random.default_rng(29)
    rows = []
    for task in range(6):
        for episode in range(120):
            failure = int(rng.random() < 0.5)
            for t in [0, 25, 50]:
                rows.append(
                    dict(
                        episode_id=f"{task}/{episode}",
                        task=task,
                        condition="nominal",
                        landmark=t,
                        failure=failure,
                        signal=failure + rng.normal(),
                        null=rng.normal(),
                    )
                )
    data = pd.DataFrame(rows)
    positive = bootstrap_concordance(data, "signal", n_boot=150, seed=1)
    assert 0.70 < positive["estimate"] < 0.82
    assert positive["ci_low"] > 0.5
    null = bootstrap_concordance(data, "null", n_boot=150, seed=1)
    assert null["ci_low"] < 0.5 < null["ci_high"]


def test_paired_conditions_preserve_bootstrap_units():
    data = confounded_data()
    data["pair_id"] = data.episode_id
    other = data.assign(condition="shift", episode_id=data.episode_id + "/shift")
    single = bootstrap_concordance(data, "planted", n_boot=40, cluster="pair_id")
    paired = bootstrap_concordance(
        pd.concat([data, other]), "planted", n_boot=40, cluster="pair_id"
    )
    assert paired["pairs"] == 2 * single["pairs"]
    assert paired["ci_low"] == single["ci_low"]


def test_empty_strata_missing_scores_and_duplicate_rejection():
    data = confounded_data()
    one_class = data[data.failure == 1]
    assert np.isnan(concordance(one_class, "elapsed").estimate)
    result = bootstrap_concordance(one_class, "elapsed", n_boot=10)
    assert result["bootstrap_degenerate"] == 10
    data.loc[data.landmark == 1, "planted"] = np.nan
    assert concordance(data, "planted").missing_scores == 40
    with pytest.raises(ValueError, match="duplicate"):
        concordance(pd.concat([data, data.iloc[:1]]), "elapsed")


def test_weighted_pairs_and_ties_hand_calculation():
    data = pd.DataFrame(
        dict(
            episode_id=list("abcd"),
            task=["t"] * 4,
            condition=["nominal"] * 4,
            landmark=[25] * 4,
            failure=[1, 1, 0, 0],
            score=[2.0, 1.0, 1.0, 0.0],
        )
    )
    assert concordance(data, "score").estimate == 3.5 / 4
