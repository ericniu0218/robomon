import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def policy():
    from robomon.policy.smolvla import SmolVLARunner
    return SmolVLARunner(threads=4)


@pytest.fixture(scope="session")
def session():
    from robomon.envs.libero_session import LiberoSession
    return LiberoSession("libero_spatial", 0)
