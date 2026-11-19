"""Artifact I/O: every pipeline stage writes Parquet plus a manifest describing exactly how it was produced."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

_TRACKED_PACKAGES = ["torch", "lerobot", "transformers", "mujoco", "robosuite", "hf-libero", "numpy", "pandas", "statsmodels"]


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config dict (key order independent)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def stage_dir(stage: str, cfg: dict[str, Any]) -> Path:
    d = ARTIFACTS / stage / config_hash(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(out_dir: Path, stage: str, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    versions = {}
    for p in _TRACKED_PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    manifest = {
        "stage": stage,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: never leave a half-written artifact
