import numpy as np
import pandas as pd
import pytest

from robomon.rollout import Rollout
from robomon.utils.artifacts import completed_episode, save_episode
from robomon.utils.io import config_hash


def test_episode_resume_requires_complete_undamaged_matching_outputs(tmp_path):
    config = {"seed": 10, "checkpoint_revision": "abc"}
    assert not completed_episode(tmp_path, config)
    result = Rollout(
        {"success": True}, pd.DataFrame({"landmark": [0]}), {"x": np.arange(3)}
    )
    save_episode(result, tmp_path, config)
    assert completed_episode(tmp_path, config)
    with pytest.raises(ValueError, match="config mismatch"):
        completed_episode(tmp_path, {**config, "seed": 11})
    (tmp_path / "arrays.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="damaged"):
        completed_episode(tmp_path, config)


def test_config_identity_is_order_independent_and_not_time_based():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_source_snapshot_preserves_exact_bytes_across_later_code_changes(
    tmp_path, monkeypatch
):
    import hashlib
    import zipfile
    from robomon.utils import io

    (tmp_path / "robomon").mkdir()
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "robomon" / "runner.py"
    source.write_text("value = 1\n")
    monkeypatch.setattr(io, "ROOT", tmp_path)
    digest = io.source_hash()
    output = tmp_path / "run"
    metadata = io.write_source_snapshot(output, digest)
    source.write_text("value = 2\n")
    assert io.source_hash() != digest
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert archive.read("robomon/runner.py") == b"value = 1\n"
    assert (
        hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()
        == metadata["archive_sha256"]
    )
    with pytest.raises(ValueError, match="Source changed"):
        io.write_source_snapshot(output, digest)
