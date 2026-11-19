import numpy as np
import pandas as pd
import pytest

from robomon.analysis.development import audit_tables, coverage_summary, development_results
from robomon.analysis.landmarks import episode_max
from robomon.data.demos import DATASET_REPO, DATASET_REVISION, dev_task_keys
from robomon.monitors.bank import BANK_FILES, SupportBank
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
)
from robomon.utils.io import config_hash, file_sha256, write_json, write_parquet


def test_bank_enforces_task_and_integrity(tmp_path):
    cfg = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
    }
    vectors = np.zeros((2, 960), dtype=np.float32)
    vectors[0, 0], vectors[1, 1] = 1, 1
    np.save(tmp_path / "embeddings.npy", vectors)
    write_parquet(
        pd.DataFrame(dict(task=["a", "b"], episode_index=[0, 1], frame_index=[0, 0])),
        tmp_path / "frames.parquet",
    )
    write_json(
        {"config": cfg, "config_hash": config_hash(cfg)}, tmp_path / "manifest.json"
    )
    write_json({"status": "complete"}, tmp_path / "results.json")
    write_json(
        {
            "config_hash": config_hash(cfg),
            "files": {n: file_sha256(tmp_path / n) for n in BANK_FILES},
        },
        tmp_path / "complete.json",
    )
    bank = SupportBank(tmp_path)
    assert bank.score(vectors[0], "a") == 0
    assert bank.score(vectors[0], "b") == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="no demonstrations"):
        bank.score(vectors[0], "c")
    (tmp_path / "embeddings.npy").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Damaged"):
        SupportBank(tmp_path)


def test_naive_max_retains_nominal_shift_bootstrap_pairing():
    rows = pd.DataFrame(
        dict(
            episode_id=["a", "a", "b", "b"],
            pair_id=["p"] * 4,
            task=["t"] * 4,
            condition=["nominal", "nominal", "shift", "shift"],
            failure=[0, 0, 1, 1],
            landmark=[0, 25, 0, 25],
            score=[0, 1, 1, 2],
        )
    )
    result = episode_max(rows, "score")
    assert result.pair_id.tolist() == ["p", "p"]
    assert result.score.tolist() == [1, 2]


def test_audit_uses_common_causal_landmarks_and_time_negative_control():
    rows = []
    for episode in range(8):
        failure = episode % 2
        for step in (0, 25, 50):
            rows.append(
                dict(
                    episode_id=str(episode),
                    pair_id=str(episode),
                    task="t",
                    condition="nominal",
                    failure=failure,
                    landmark=step,
                    elapsed=step,
                    dispersion=float(failure),
                    consistency=float(failure) if step else np.nan,
                    support=float(failure),
                )
            )
    output = audit_tables(pd.DataFrame(rows), n_boot=20)
    matched = [row for row in output if row["matching"] != "naive"]
    assert all(row["rows"] == 16 for row in matched)
    assert all(row["estimate"] == 0.5 for row in matched if row["signal"] == "elapsed")
    assert all(row["estimate"] == 1 for row in matched if row["signal"] == "support")


def test_two_smoke_episodes_cannot_be_treated_as_calibrated_dev_set():
    samples = pd.DataFrame(
        dict(
            task=["libero_spatial/0"] * 2,
            condition=["nominal", "shift"],
            success=[True, False],
            steps=[80, 280],
        )
    )
    coverage = coverage_summary(samples)
    assert coverage["failure_rate_gate_20_to_80_percent"]
    assert not coverage["eligible_for_frozen_signal_selection"]
    full = pd.DataFrame(
        [
            dict(task=t, condition=c, success=True, steps=80)
            for t in dev_task_keys()
            for c in ("nominal", "shift")
            for _ in range(16)
        ]
    )
    assert coverage_summary(full)["planned_dev_sample_complete"]
    assert not coverage_summary(full)["eligible_for_frozen_signal_selection"]


def test_balanced_overall_failure_rate_cannot_hide_uninformative_strata():
    rows, summaries = [], []
    for task_index, task in enumerate(dev_task_keys()):
        for condition in ("nominal", "shift"):
            for unit in range(16):
                failure = task_index % 2
                episode = f"{task}/{condition}/{unit}"
                rows.append(dict(episode_id=episode,pair_id=f"{task}/{unit}",task=task,condition=condition,
                                 landmark=25,failure=failure,elapsed=25.,dispersion=float(failure),
                                 consistency=float(failure),support=float(failure)))
                summaries.append(dict(task=task,condition=condition,success=not failure,steps=80))
    result = development_results(pd.DataFrame(rows),pd.DataFrame(summaries),n_boot=10)
    assert result["coverage"]["failure_rate_gate_20_to_80_percent"]
    assert not result["coverage"]["primary_evaluable"]
    assert not result["coverage"]["eligible_for_frozen_signal_selection"]
