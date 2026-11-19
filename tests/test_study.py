import hashlib
import json
import subprocess
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from robomon import study
from robomon.analysis.conditional import fit_pairwise_conditional
from robomon.analysis.landmarks import bootstrap_concordance
from robomon.interventions import match_alarm_states
from robomon.rollout import EpisodeSpec, Rollout
from robomon.study_data import validate_pairs
from robomon.utils.artifacts import save_episode
from robomon.utils.io import config_hash, write_json


def test_full_episode_plan_is_disjoint_paired_and_complete():
    dev, test = (
        study.episode_plan("dev", 0.6, 10.0),
        study.episode_plan("test", 0.6, 10.0),
    )
    assert len(dev) == 384 and len(test) == 896
    assert set(study.task_keys("dev")).isdisjoint(study.task_keys("test"))
    groups = {}
    for item in test:
        spec = EpisodeSpec(**item)
        groups.setdefault(spec.pair_id, []).append(spec)
    assert len(groups) == 448
    for group in groups.values():
        assert [x.condition for x in group] == ["nominal", "shift"]
        assert group[0].seed == group[1].seed
        assert group[0].brightness == 1 and group[1].brightness == 0.6


def test_hierarchy_cannot_be_rescued_by_later_positive_results():
    values = {
        key: {"status": "ok", "ci_low": 1.0, "ci_high": 2.0}
        for key in ("H1", "H3", "H4a", "H4b")
    }
    assert all(
        row["decision"] == "supported" for row in study.sequence_decisions(values)
    )
    values["H3"]["ci_low"] = -0.1
    decisions = study.sequence_decisions(values)
    assert [row["decision"] for row in decisions] == [
        "supported",
        "not_established",
        "descriptive_only_hierarchy_closed",
        "descriptive_only_hierarchy_closed",
    ]
    values["H1"] = {"status": "inconclusive", "ci_low": 0.9}
    assert study.sequence_decisions(values)[0]["decision"] == "inconclusive"
    assert (
        study.sequence_decisions(values)[1]["decision"]
        == "descriptive_only_hierarchy_closed"
    )

    values["H1"] = {"status": "ok", "ci_low": float("nan")}
    assert study.sequence_decisions(values)[0]["decision"] == "inconclusive"


@pytest.fixture
def committed_protocol(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "robomon"
    (root / "robomon").mkdir(parents=True)
    (root / "robomon" / "method.py").write_text("METHOD = 'declared'\n")
    for name in study.CONFIRMATORY_SCRIPTS:
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("# frozen script\n")
    monkeypatch.setattr(study.io, "ROOT", root)
    monkeypatch.setattr(study.io, "runtime_fingerprint", lambda: {"python": "test"})
    monkeypatch.setattr(study, "dependency_hashes", lambda: {"controller": "a"})
    monkeypatch.setattr(study, "asset_hashes", lambda: {"scene": "b"})
    monkeypatch.setattr(study, "model_hashes", lambda: {"weights": "c"})
    protocol = {
        "schema": 1,
        "status": "frozen",
        "split": study.task_split(),
        "visual_shift": {"brightness": 0.6, "noise_sigma": 10.0},
        "test_episodes": study.episode_plan("test", 0.6, 10.0),
        "execution": {
            "device": "mps",
            "threads": 4,
            "k": 8,
            "shared_prefix": True,
            "render_mode": "always",
            "reset_mode": "isolated",
        },
        "critical_sources": study.critical_source_hashes(),
        "runtime": {"python": "test"},
        "dependency_sources": {"controller": "a"},
        "simulator_assets": {"scene": "b"},
        "model_snapshots": {"weights": "c"},
    }
    path = root / "frozen_protocol.json"
    write_json(protocol, path)
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "robomon"], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=RoboMon Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Freeze test fixture",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = root / "receipt.json"
    write_json(
        {
            "commit": commit,
            "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        receipt,
    )
    return root, path, receipt


def test_gate_accepts_exact_commit_and_fingerprints(committed_protocol):
    _, path, receipt = committed_protocol
    loaded = study.load_frozen_protocol(path, receipt)
    assert loaded["_freeze_commit"] == json.loads(receipt.read_text())["commit"]


@pytest.mark.parametrize(
    "damage",
    ["source", "new_module", "protocol", "uncommitted_protocol", "assets", "weights"],
)
def test_gate_rejects_scientific_drift(committed_protocol, monkeypatch, damage):
    root, path, receipt = committed_protocol
    if damage == "source":
        (root / "robomon" / "method.py").write_text("METHOD = 'changed'\n")
    elif damage == "new_module":
        (root / "robomon" / "extra.py").write_text("NEW = True\n")
    elif damage in ("protocol", "uncommitted_protocol"):
        protocol = json.loads(path.read_text())
        protocol["unreviewed_change"] = True
        write_json(protocol, path)
        if damage == "uncommitted_protocol":
            record = json.loads(receipt.read_text())
            record["protocol_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(record, receipt)
    elif damage == "assets":
        monkeypatch.setattr(study, "asset_hashes", lambda: {"scene": "changed"})
    else:
        monkeypatch.setattr(study, "model_hashes", lambda: {"weights": "changed"})
    with pytest.raises(ValueError):
        study.load_frozen_protocol(path, receipt)


def test_nonmonotone_alarm_control_must_never_have_triggered():
    rows = []
    for episode, scores in (("early_alarm", [2.0, 0.1]), ("late_alarm", [0.1, 2.0])):
        for step, score in zip((50, 100), scores):
            rows.append(
                {
                    "episode_id": episode,
                    "pair_id": episode,
                    "task": "libero_spatial/1",
                    "condition": "nominal",
                    "landmark": step,
                    "score": score,
                    "horizon": 280,
                }
            )
    data = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="allowed partition"):
        match_alarm_states(data, 1.0)
    selected = match_alarm_states(
        data, 1.0, max_pairs_per_cell=2, allowed_tasks=["libero_spatial/1"]
    )
    assert selected["coverage"][0]["first_alarms"] == 2
    assert selected["coverage"][0]["alarms_with_exact_controls"] == 1


def test_bootstrap_keeps_units_without_common_landmarks_in_the_universe():
    rows = pd.DataFrame(
        [
            {
                "episode_id": "fail",
                "pair_id": "f",
                "task": "t",
                "condition": "nominal",
                "landmark": 25,
                "failure": 1,
                "score": 1.0,
                "a": 1.0,
                "b": 2.0,
                "c": 3.0,
            },
            {
                "episode_id": "pass",
                "pair_id": "p",
                "task": "t",
                "condition": "nominal",
                "landmark": 25,
                "failure": 0,
                "score": 0.0,
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
            },
        ]
    )
    universe = pd.DataFrame(
        {"task": ["t", "t", "t"], "pair_id": ["f", "p", "early_terminal"]}
    )
    conditional = bootstrap_concordance(
        rows, "score", n_boot=512, seed=7, cluster="pair_id"
    )
    complete = bootstrap_concordance(
        rows, "score", n_boot=512, seed=7, cluster="pair_id", cluster_universe=universe
    )
    assert conditional["estimate"] == complete["estimate"] == 1.0
    assert conditional["bootstrap_valid"] != complete["bootstrap_valid"]
    with pytest.raises(ValueError, match="omits"):
        bootstrap_concordance(
            rows, "score", cluster="pair_id", cluster_universe=universe.iloc[:1]
        )
    with pytest.raises(ValueError, match="universe"):
        fit_pairwise_conditional(
            rows, ("a", "b", "c"), cluster_universe=universe.iloc[:1]
        )


def paired_run(tmp_path, mismatch=False):
    specs = [
        EpisodeSpec(
            "libero_spatial",
            1,
            0,
            17,
            condition,
            0.6 if condition == "shift" else 1.0,
            10.0 if condition == "shift" else 0.0,
        )
        for condition in ("nominal", "shift")
    ]
    config = {
        "stage": "test_rollouts",
        "args": {"reset_mode": "isolated"},
        "episodes": [asdict(spec) for spec in specs],
    }
    write_json(
        {"config": config, "config_hash": config_hash(config)},
        tmp_path / "manifest.json",
    )
    write_json({"status": "complete", "completed": 2}, tmp_path / "results.json")
    for spec in specs:
        arrays = {
            "reset_state": np.zeros(4),
            "reset_body_pos": np.zeros((2, 3)),
            "reset_body_quat": np.ones((2, 4)),
        }
        if mismatch and spec.condition == "shift":
            arrays["reset_body_pos"][0, 0] = 0.01
        result = Rollout(
            {"reset_context": {"mode": "isolated", "prior_init_states": []}},
            pd.DataFrame({"landmark": [0]}),
            arrays,
        )
        save_episode(result, tmp_path / "episodes" / config_hash(asdict(spec)), config)


def test_paired_scene_validation_rejects_a_hidden_fixture_change(tmp_path):
    paired_run(tmp_path)
    assert (
        validate_pairs(
            tmp_path, allowed_tasks=["libero_spatial/1"], stage="test_rollouts"
        )["pairs"]
        == 1
    )
    paired_run(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="scene differs"):
        validate_pairs(
            tmp_path, allowed_tasks=["libero_spatial/1"], stage="test_rollouts"
        )


def test_test_driver_resumes_without_constructing_another_model(tmp_path, monkeypatch):
    from test_rollout import ToyPolicy, ToySession

    import scripts.run_test as driver

    class Session(ToySession):
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    protocol = {
        "_protocol_sha256": "synthetic",
        "_freeze_commit": "synthetic",
        "checkpoint_revision": "synthetic",
        "vlm_revision": "synthetic",
        "embedding_definition": "synthetic",
        "execution": {
            "device": "mps",
            "threads": 4,
            "k": 8,
            "shared_prefix": True,
            "render_mode": "always",
            "reset_mode": "isolated",
        },
        "test_episodes": [asdict(EpisodeSpec("libero_spatial", 1, 0, 17))],
    }
    monkeypatch.setattr(driver, "load_frozen_protocol", lambda *args: protocol)
    monkeypatch.setattr(driver, "LiberoSession", Session)
    constructions = []
    monkeypatch.setattr(
        driver,
        "SmolVLARunner",
        lambda **kwargs: constructions.append(True) or ToyPolicy(),
    )
    monkeypatch.setattr(study.io, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        driver.sys,
        "argv",
        ["run_test.py", "--protocol", "synthetic", "--receipt", "synthetic"],
    )
    driver.main()
    assert len(constructions) == 1
    driver.main()
    assert len(constructions) == 1


@pytest.mark.parametrize("threshold", [1.0, 1000.0])
def test_synthetic_heldout_h4_pipeline_including_empty_matching(
    tmp_path, monkeypatch, threshold
):
    """Exercise the production entry points without accessing any real held-out outcome."""
    from types import SimpleNamespace

    from test_rollout import ToyPolicy, ToySession

    import scripts.analyze_h4_test as analysis_driver
    import scripts.run_h4_test as branch_driver
    import scripts.run_test as rollout_driver
    import scripts.select_h4_test as selection_driver
    from robomon.policy.smolvla import (
        CHECKPOINT_REVISION,
        EMBEDDING_DEFINITION,
        VLM_REVISION,
    )

    class Session(ToySession):
        max_steps = 280
        render_mode = "always"

        def __init__(self, *args, **kwargs):
            pass

        def reset(self, initial):
            self.initial = initial
            observation = super().reset(initial)
            self.last_reset_context = {"mode": "isolated", "prior_init_states": []}
            self.rs = SimpleNamespace(
                sim=SimpleNamespace(
                    model=SimpleNamespace(
                        body_pos=np.zeros((1, 3)), body_quat=np.ones((1, 4))
                    )
                )
            )
            return observation

        def reset_for_replay(self, initial, history):
            assert not history
            return self.reset(initial)

        def observe(self):
            assert self.ready
            return {
                "pixels": {"a": np.full((2, 2, 3), self.t % 256, dtype=np.uint8)},
                "x": self.x,
            }

        def step(self, action, *, render=False):
            result = super().step(action, render=render)
            result.success = self.initial == 1 and self.t >= 180
            return result

        def close(self):
            pass

    class Policy(ToyPolicy):
        def query(self, *args, **kwargs):
            result = super().query(*args, **kwargs)
            result.embedding = np.ones(960)
            return result

    specs = [
        EpisodeSpec(
            "libero_spatial",
            1,
            initial,
            17 + initial,
            condition,
            0.6 if condition == "shift" else 1.0,
            10.0 if condition == "shift" else 0.0,
        )
        for initial in (0, 1)
        for condition in ("nominal", "shift")
    ]
    protocol = {
        "_protocol_sha256": "synthetic",
        "_freeze_commit": "synthetic",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
        "execution": {
            "device": "mps",
            "threads": 4,
            "k": 8,
            "shared_prefix": True,
            "render_mode": "always",
            "reset_mode": "isolated",
        },
        "test_episodes": [asdict(spec) for spec in specs],
        "method_text": "Synthetic plumbing check; no robotics claims",
        "h4": {
            "signal": "consistency",
            "reduction": "max",
            "alarm": {"threshold": threshold},
            "seed": 20261111,
            "replicates": 3,
            "pairs_per_cell": 2,
        },
    }
    monkeypatch.setattr(study.io, "ARTIFACTS", tmp_path / "artifacts")
    for module in (rollout_driver, selection_driver, branch_driver, analysis_driver):
        monkeypatch.setattr(module, "load_frozen_protocol", lambda *args: protocol)
    monkeypatch.setattr(rollout_driver, "LiberoSession", Session)
    monkeypatch.setattr(branch_driver, "LiberoSession", Session)
    monkeypatch.setattr(rollout_driver, "SmolVLARunner", lambda **kwargs: Policy())
    monkeypatch.setattr(branch_driver, "SmolVLARunner", lambda **kwargs: Policy())
    flags = ["--protocol", "synthetic", "--receipt", "synthetic"]
    monkeypatch.setattr(rollout_driver.sys, "argv", ["run_test.py", *flags])
    rollout_driver.main()
    raw = next((tmp_path / "artifacts/test_rollouts").iterdir())
    trunk = json.loads((raw / "manifest.json").read_text())["config"]
    frames = []
    for spec in specs:
        frame = pd.read_parquet(
            raw / "episodes" / config_hash(asdict(spec)) / "events.parquet"
        )
        frame.loc[frame.landmark > 0, "consistency"] = (
            2.0 if spec.init_state_id == 0 else 0.1
        )
        frames.append(frame)
    primary = tmp_path / "primary"
    primary.mkdir()
    pd.concat(frames).to_parquet(primary / "scored_landmarks.parquet", index=False)
    primary_config = {
        "stage": "test_analysis",
        "protocol_sha": "synthetic",
        "rollout_config": trunk,
    }
    write_json(
        {"config": primary_config, "config_hash": config_hash(primary_config)},
        primary / "manifest.json",
    )
    write_json(
        {
            "h1": {"status": "ok", "ci_low": 0.7},
            "h3": {
                "status": "ok",
                "coefficients": {
                    "support__last": {"coef": 1.0, "ci_low": 0.5, "ci_high": 1.5}
                },
            },
        },
        primary / "results.json",
    )
    monkeypatch.setattr(
        selection_driver.sys,
        "argv",
        ["select_h4_test.py", *flags, "--analysis", str(primary)],
    )
    selection_driver.main()
    selection = next((tmp_path / "artifacts/h4_selection_test").iterdir())
    monkeypatch.setattr(
        branch_driver.sys,
        "argv",
        ["run_h4_test.py", *flags, "--selection", str(selection), "--run", str(raw)],
    )
    branch_driver.main()
    branched = next((tmp_path / "artifacts/h4_test").iterdir())
    monkeypatch.setattr(
        analysis_driver.sys,
        "argv",
        ["analyze_h4_test.py", str(branched), *flags, "--analysis", str(primary)],
    )
    analysis_driver.main()
    result = json.loads(
        (
            next((tmp_path / "artifacts/h4_analysis_test").iterdir()) / "results.json"
        ).read_text()
    )
    if threshold == 1.0:
        assert result["gates_passed"] == 2 and result["branches"] == 12
        assert result["contrasts"]["h4a_control_minus_alarm_replan"]["estimate"] == 1.0
    else:
        assert result["gates_passed"] == result["branches"] == 0
        assert result["sequence_decisions"][2]["decision"] == "inconclusive"
