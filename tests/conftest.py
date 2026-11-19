import os
import sys
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--policy-device", choices=("cpu", "mps"), default="cpu")


@pytest.fixture(scope="session")
def policy(request):
    from robomon.policy.smolvla import SmolVLARunner

    return SmolVLARunner(threads=4, device=request.config.getoption("--policy-device"))


@pytest.fixture(scope="session")
def session(policy):
    # Loading the policy first avoids a native-library initialization crash on Linux.
    from robomon.envs.libero_session import LiberoSession

    env = LiberoSession("libero_spatial", 0, render_mode="scheduled")
    yield env
    env.close()
