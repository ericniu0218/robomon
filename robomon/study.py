"""Committed protocol gate. No held-out entry point may bypass this verification."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sysconfig
from pathlib import Path

from robomon.rollout import EpisodeSpec, derived_seed
from robomon.splits import task_split
from robomon.utils import io

DEPENDENCY_FILES = (
    "lerobot/sim/integrations/libero.py",
    "libero/libero/envs/env_wrapper.py",
    "libero/libero/envs/bddl_base_domain.py",
    "robosuite/environments/base.py",
    "robosuite/controllers/osc.py",
    "robosuite/utils/control_utils.py",
    "robosuite/utils/binding_utils.py",
)
CONFIRMATORY_SCRIPTS = (
    "scripts/run_test.py",
    "scripts/analyze_test.py",
    "scripts/build_demo_bank.py",
    "scripts/select_h4_test.py",
    "scripts/run_h4_test.py",
    "scripts/analyze_h4_test.py",
)


def task_keys(partition: str) -> list[str]:
    if partition not in ("dev", "test"):
        raise ValueError("Unknown partition")
    return sorted(
        f"{suite}/{task}"
        for suite, cells in task_split()["suites"].items()
        for task in cells[partition]
    )


def episode_plan(partition: str, brightness: float, noise_sigma: float) -> list[dict]:
    from dataclasses import asdict

    result = []
    for task in task_keys(partition):
        suite, task_id = task.split("/")
        for initial in range(16):
            seed = derived_seed(20261110, f"{task}/{initial}")
            for condition in ("nominal", "shift"):
                result.append(
                    asdict(
                        EpisodeSpec(
                            suite,
                            int(task_id),
                            initial,
                            seed,
                            condition,
                            brightness if condition == "shift" else 1.0,
                            noise_sigma if condition == "shift" else 0.0,
                        )
                    )
                )
    return result


def critical_source_hashes() -> dict:
    paths = sorted((io.ROOT / "robomon").rglob("*.py"))
    paths += [io.ROOT / name for name in CONFIRMATORY_SCRIPTS]
    return {str(path.relative_to(io.ROOT)): io.file_sha256(path) for path in paths}


def dependency_hashes() -> dict:
    root = Path(sysconfig.get_paths()["purelib"])
    return {name: io.file_sha256(root / name) for name in DEPENDENCY_FILES}


def asset_hashes() -> dict:
    import yaml

    settings = yaml.safe_load((Path.home() / ".libero/config.yaml").read_text())
    roots = {
        "assets": Path.home() / ".cache/libero/assets",
        "bddl": Path(settings["bddl_files"]),
        "initial_states": Path(settings["init_states"]),
    }
    result = {}
    for label, root in roots.items():
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
        if not paths:
            raise ValueError(f"Missing simulator input files: {label}")
        result.update(
            {
                label + "/" + str(path.relative_to(root)): io.file_sha256(path)
                for path in paths
            }
        )
    return result


def model_hashes() -> dict:
    from huggingface_hub import snapshot_download

    from robomon.policy.smolvla import CHECKPOINT_REVISION, VLM_REVISION

    result = {}
    for repo, revision in (
        ("lerobot/smolvla_libero", CHECKPOINT_REVISION),
        ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", VLM_REVISION),
    ):
        root = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.jinja",
                    "*.safetensors",
                    "*.model",
                ],
                local_files_only=True,
            )
        )
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix
            in (".safetensors", ".json", ".txt", ".model", ".py", ".jinja")
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
        result.update(
            {
                repo + "/" + str(path.relative_to(root)): io.file_sha256(path)
                for path in paths
            }
        )
    return result


def _git(repo: Path, *args) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
    )


def load_frozen_protocol(protocol_path: Path | str, receipt_path: Path | str) -> dict:
    """Verify the commit's protocol and source, the executing source, dependencies and assets.

    The receipt is written after the freeze commit to avoid a self-referential Git hash. Source
    paths are relative to the RoboMon source root, including for checksum-verified archive resume.
    Reports and plots may change later; scientific code and the protocol cannot silently change.
    """
    protocol_path, receipt_path = (
        Path(protocol_path).resolve(),
        Path(receipt_path).resolve(),
    )
    raw = protocol_path.read_bytes()
    protocol, receipt = json.loads(raw), json.loads(receipt_path.read_text())
    if protocol.get("schema") != 1 or protocol.get("status") != "frozen":
        raise ValueError("A frozen protocol is required before held-out work")
    if protocol.get("split") != task_split():
        raise ValueError("Protocol task split differs from the declared partition")
    shift = protocol["visual_shift"]
    if protocol["test_episodes"] != episode_plan(
        "test", shift["brightness"], shift["noise_sigma"]
    ):
        raise ValueError(
            "Frozen held-out cohort must contain all 896 prespecified episodes"
        )
    expected_execution = {
        "device": "mps",
        "threads": 4,
        "k": 8,
        "shared_prefix": True,
        "render_mode": "always",
        "reset_mode": "isolated",
    }
    if protocol["execution"] != expected_execution:
        raise ValueError("Unexpected confirmatory execution contract")
    if hashlib.sha256(raw).hexdigest() != receipt["protocol_sha256"]:
        raise ValueError("Protocol differs from the freeze receipt")
    repo = protocol_path.parent.parent
    commit = receipt["commit"]
    if _git(repo, "cat-file", "-t", commit).strip() != b"commit":
        raise ValueError("Freeze receipt must name a real Git commit")
    relative_protocol = protocol_path.relative_to(repo).as_posix()
    if _git(repo, "show", f"{commit}:{relative_protocol}") != raw:
        raise ValueError("Protocol was not committed at the freeze receipt's SHA")
    for name, expected in protocol["critical_sources"].items():
        if io.file_sha256(io.ROOT / name) != expected:
            raise ValueError(f"Scientific source changed after freeze: {name}")
        blob = _git(repo, "show", f"{commit}:robomon/{name}")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(
                f"Scientific source is not present in the freeze commit: {name}"
            )
    # Enforce the complete list too: a new imported core module cannot escape the source gate.
    if set(critical_source_hashes()) != set(protocol["critical_sources"]):
        raise ValueError("Scientific source file list changed after freeze")
    if io.runtime_fingerprint() != protocol["runtime"]:
        raise ValueError("Runtime changed after freeze")
    if dependency_hashes() != protocol["dependency_sources"]:
        raise ValueError("Installed simulator/controller source changed after freeze")
    if asset_hashes() != protocol["simulator_assets"]:
        raise ValueError("Simulator assets changed after freeze")
    if model_hashes() != protocol["model_snapshots"]:
        raise ValueError("Cached model/tokenizer bytes changed after freeze")
    protocol["_freeze_commit"] = commit
    protocol["_protocol_sha256"] = receipt["protocol_sha256"]
    return protocol


def sequence_decisions(estimates: dict) -> list[dict]:
    """Only the prespecified H1 -> H3 -> H4a -> H4b sequence can make confirmatory claims.

    Directional rejection requires a two-sided 95% lower bound strictly above the null. If an
    earlier test is inconclusive or fails, all later bounds are descriptive only. Individual
    bootstrap coverage is approximate, so nominal family-wise control is not a finite-sample proof.
    """
    active, decisions = True, []
    for name, null in (("H1", 0.5), ("H3", 0.0), ("H4a", 0.0), ("H4b", 0.0)):
        row = estimates.get(name)
        available = (
            row is not None
            and row.get("status", "ok") == "ok"
            and row.get("ci_low") is not None
            and math.isfinite(row["ci_low"])
        )
        supported = bool(available and row["ci_low"] > null)
        decision = (
            "supported"
            if active and supported
            else "not_established"
            if active and available
            else "inconclusive"
            if active
            else "descriptive_only_hierarchy_closed"
        )
        decisions.append(
            {"hypothesis": name, "null": null, "decision": decision, "interval": row}
        )
        active = active and supported
    return decisions
