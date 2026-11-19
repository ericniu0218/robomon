import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from robomon.analysis.conditional import fit_pairwise_conditional


def sample(beta=0.8):
    rng = np.random.default_rng(15)
    rows = []
    for task in range(12):
        for episode in range(64):
            support, dispersion, consistency = rng.normal(size=3)
            failure = int(
                rng.random()
                < expit(
                    (task - 6) / 6
                    + beta * support
                    + 0.5 * dispersion
                    - 0.2 * consistency
                )
            )
            for t in (25, 50):
                rows.append(
                    dict(
                        task=task,
                        condition="nominal",
                        landmark=t,
                        episode_id=f"{task}/{episode}",
                        pair_id=f"{task}/{episode}",
                        failure=failure,
                        support=support,
                        dispersion=dispersion,
                        consistency=consistency,
                    )
                )
    return pd.DataFrame(rows)


def test_recovers_conditional_effect_with_episode_cluster_uncertainty():
    result = fit_pairwise_conditional(
        sample(),
        ("support", "dispersion", "consistency"),
        n_boot=80,
        seed=3,
        scales={"support": 1.0, "dispersion": 1.0, "consistency": 1.0},
    )
    assert result["status"] == "ok"
    effect = result["coefficients"]["support"]
    assert 0.6 < effect["coef"] < 1.0
    assert effect["ci_low"] > 0 and effect["ci_low"] < 0.8 < effect["ci_high"]


def test_exact_null_and_constant_within_stratum_control():
    data = sample(beta=0)
    positive = data.assign(support=1.0, episode_id=data.episode_id + "/p")
    negative = data.assign(support=-1.0, episode_id=data.episode_id + "/n")
    balanced = pd.concat([positive, negative], ignore_index=True)
    result = fit_pairwise_conditional(
        balanced, ("support", "dispersion", "consistency"), n_boot=20
    )
    assert abs(result["coefficients"]["support"]["coef"]) < 1e-6
    assert abs(result["coefficients"]["support"]["ci_low"]) < 1e-6
    data["task_only"] = data.task.astype(float)
    result = fit_pairwise_conditional(
        data, ("support", "dispersion", "task_only"), n_boot=10
    )
    assert result["status"] == "inconclusive" and result["reason"] == "unidentifiable"


def test_separation_is_inconclusive_instead_of_artificially_regularized():
    data = sample()
    data["support"] = data.failure.astype(float)
    result = fit_pairwise_conditional(
        data, ("support", "dispersion", "consistency"), n_boot=10
    )
    assert result["status"] == "inconclusive"
    data.loc[0, "consistency"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fit_pairwise_conditional(
            data, ("support", "dispersion", "consistency"), n_boot=10
        )
