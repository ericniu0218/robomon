"""A single LIBERO environment with render-on-demand and state save/restore.

Why: software rendering costs ~0.19 s per step on CPU while physics costs ~0.0175 s. SmolVLA executes
50 actions open-loop, so images are only needed when the policy (or a monitor) is queried. The camera
observables are disabled during plain steps and re-enabled for an explicit `observe()` call.

Built on lerobot's `LiberoEnv` so that initial states, settle steps, control mode and observation
formatting are exactly the ones `lerobot-eval` uses.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


@dataclass
class StepResult:
    success: bool
    done: bool  # robosuite horizon reached


class LiberoSession:
    def __init__(self, suite_name: str, task_id: int, image_size: int = 256):
        from libero.libero import benchmark
        from lerobot.sim.integrations.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.suite_name, self.task_id = suite_name, task_id
        self.env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                             obs_type="pixels_agent_pos", observation_height=image_size,
                             observation_width=image_size, init_states=True)
        self.task_description: str = self.env.task_description
        self.max_steps: int = int(self.env._max_episode_steps)
        self.n_init_states: int = len(self.env._init_states)
        self._cams: list[str] = []
        self._rendering = True

    # ---- internals --------------------------------------------------------------------------
    @property
    def core(self):  # LIBERO ControlEnv (OffScreenRenderEnv)
        return self.env._env

    @property
    def rs(self):  # underlying robosuite environment
        return self.env._env.env

    def _set_rendering(self, on: bool) -> None:
        if on == self._rendering:
            return
        for c in self._cams:
            self.rs._observables[c].set_enabled(on)
            self.rs._observables[c].set_active(on)
        self._rendering = on

    # ---- episode API ------------------------------------------------------------------------
    def reset(self, init_state_id: int) -> dict:
        """Reset to a fixed LIBERO initial state. Returns the formatted observation (with images)."""
        if self.env._env is not None:
            self._set_rendering(True)
        self.env.init_state_id = int(init_state_id)
        obs, _ = self.env.reset(seed=int(init_state_id))
        self._cams = [k for k in self.rs._observables if k.endswith("_image")]
        self._rendering = True
        self._set_rendering(False)
        return obs

    def step(self, action: np.ndarray) -> StepResult:
        """Physics-only step (no camera rendering)."""
        self._set_rendering(False)
        _, _, done, _ = self.core.step(np.asarray(action, dtype=np.float64))
        return StepResult(success=bool(self.core.check_success()), done=bool(done))

    def observe(self) -> dict:
        """Render both cameras now and return lerobot's formatted observation."""
        self._set_rendering(True)
        raw = self.rs._get_observations(force_update=True)
        self._set_rendering(False)
        return self.env._format_raw_obs(raw)

    # ---- state save / restore (for counterfactual branching) ------------------------------------
    def get_state(self) -> np.ndarray:
        return np.array(self.core.get_sim_state(), dtype=np.float64).copy()

    def set_state(self, state: np.ndarray) -> None:
        self.core.set_state(np.asarray(state, dtype=np.float64))
        self.rs.sim.forward()

    def close(self) -> None:
        self.env.close()


def perturb_images(obs: dict, brightness: float, noise_sigma: float, rng: np.random.Generator) -> dict:
    """Visual shift applied to the uint8 camera images: multiply brightness, add Gaussian pixel noise.

    brightness=1 and noise_sigma=0 is the identity. The same strength is used for a whole episode;
    the noise itself is redrawn at every observation (sensor noise).
    """
    if brightness == 1.0 and noise_sigma == 0.0:
        return obs
    out = dict(obs)
    out["pixels"] = {}
    for k, img in obs["pixels"].items():
        x = img.astype(np.float32) * brightness
        if noise_sigma > 0:
            x = x + rng.normal(0.0, noise_sigma, size=x.shape).astype(np.float32)
        out["pixels"][k] = np.clip(x, 0, 255).astype(np.uint8)
    return out
