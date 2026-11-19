"""Resume a dev rollout, demo bank or H4 dev stage from its checksum-verified archived source.

Usage: .venv/bin/python diagnostics/resume_snapshot.py artifacts/dev_rollouts/RUN_HASH
Use --verify-only to check the archive/runtime without starting simulation or model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = args.run.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    snapshot = json.loads((directory / "source_snapshot.json").read_text())
    archive_path = directory / snapshot["archive"]
    if (
        archive_path.parent != directory
        or hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != snapshot["archive_sha256"]
    ):
        raise ValueError("Source archive is missing or damaged")
    if snapshot["source_hash"] != config["source_hash"]:
        raise ValueError("Source archive belongs to a different implementation")
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(snapshot["files"]):
            raise ValueError("Source file list differs from the snapshot descriptor")
        contents = {}
        for name, expected in snapshot["files"].items():
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] not in ("robomon", "scripts")
            ):
                raise ValueError("Unsafe source archive path")
            value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError(f"Damaged archived source: {name}")
            contents[name] = value
    source = directory / "source"
    source.mkdir(exist_ok=True)
    for name, value in contents.items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Archived source would follow a path outside the snapshot")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    sys.path.insert(0, str(source))
    from robomon.utils import io

    if (
        io.source_hash() != config["source_hash"]
        or io.config_hash(config) != manifest["config_hash"]
    ):
        raise ValueError("Source/config hash does not reproduce the original run")
    if io.runtime_fingerprint() != config["runtime"]:
        raise ValueError(
            "Runtime differs; reproduce the recorded environment before resuming"
        )
    print(f"VERIFIED {directory}", flush=True)
    if args.verify_only:
        return
    io.ARTIFACTS = directory.parents[1]
    original_write = io.write_manifest

    def preserve_snapshot_link(*a, **kw):
        original_write(*a, **kw)
        current = json.loads((directory / "manifest.json").read_text())
        current["source_archive_sha256"] = snapshot["archive_sha256"]
        current["resume_launcher_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        io.write_json(current, directory / "manifest.json")

    io.write_manifest = preserve_snapshot_link
    stage = config["stage"]
    scripts = {
        "dev_rollouts": "run_dev.py",
        "demo_bank": "build_demo_bank.py",
        "h4_dev": "run_h4_dev.py",
        "test_rollouts": "run_test.py",
        "demo_bank_test": "build_demo_bank.py",
        "h4_test": "run_h4_test.py",
    }
    if stage not in scripts:
        raise ValueError(
            "Snapshot resume supports dev_rollouts, demo_bank and h4_dev stages"
        )
    script = source / "scripts" / scripts[stage]
    argv = [str(script)]
    for key, value in config.get("cli_args", config["args"]).items():
        if value is None or value is False:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is True:
            continue
        argv.extend(map(str, value if isinstance(value, list) else [value]))
    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
