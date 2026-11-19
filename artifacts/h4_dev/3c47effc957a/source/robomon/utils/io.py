"""Artifact I/O: every pipeline stage writes Parquet plus a manifest describing exactly how it was produced."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

_TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "lerobot",
    "transformers",
    "mujoco",
    "robosuite",
    "hf-libero",
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "huggingface-hub",
    "safetensors",
    "pyarrow",
    "av",
    "datasets",
    "tokenizers",
    "pillow",
]


def runtime_fingerprint() -> dict:
    versions = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config dict (key order independent)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> dict[str, bytes]:
    files = {}
    for directory in (ROOT / "robomon", ROOT / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return files


def source_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, contents in files.items():
        digest.update(name.encode() + b"\0" + contents + b"\0")
    return digest.hexdigest()


def source_hash() -> str:
    """Hash actual source bytes, including uncommitted changes; never select inputs by mtime."""
    return source_digest(source_files())


def write_source_snapshot(
    directory: Path, expected_hash: str, files: dict[str, bytes] | None = None
) -> dict:
    files = source_files() if files is None else files
    if source_digest(files) != expected_hash:
        raise ValueError(
            "Source changed while stage configuration was being built; restart the stage"
        )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "source.zip.tmp"
    with zipfile.ZipFile(temporary, "w") as archive:
        for name, contents in files.items():
            item = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, contents)
    temporary.replace(directory / "source.zip")
    metadata = {
        "source_hash": expected_hash,
        "archive": "source.zip",
        "archive_sha256": file_sha256(directory / "source.zip"),
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    }
    write_json(metadata, directory / "source_snapshot.json")
    return metadata


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def stage_dir(stage: str, cfg: dict[str, Any]) -> Path:
    d = ARTIFACTS / stage / config_hash(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(
    out_dir: Path, stage: str, cfg: dict[str, Any], extra: dict[str, Any] | None = None
) -> None:
    snapshot = write_source_snapshot(out_dir, cfg.get("source_hash", source_hash()))
    auxiliary = {}
    if "diagnostic_sha" in cfg:
        diagnostic = Path(sys.argv[0])
        if not diagnostic.is_file() or file_sha256(diagnostic) != cfg["diagnostic_sha"]:
            raise ValueError("Diagnostic source changed while its configuration was being built")
        (out_dir / "diagnostic.py").write_bytes(diagnostic.read_bytes())
        auxiliary["diagnostic.py"] = cfg["diagnostic_sha"]
    manifest = {
        "stage": stage,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "source_hash": snapshot["source_hash"],
        "source_archive_sha256": snapshot["archive_sha256"],
        "auxiliary_sources": auxiliary,
        **runtime_fingerprint(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {}),
    }
    write_json(manifest, out_dir / "manifest.json")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: never leave a half-written artifact
