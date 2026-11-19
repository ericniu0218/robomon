"""A single LIBERO environment with standard rendering and optional scheduled rendering.

Why: software rendering costs ~0.19 s per step on CPU while physics costs ~0.0175 s. SmolVLA executes
50 actions open-loop, so images are only needed when the policy (or a monitor) is queried. The camera
observables keep their clocks running, but expensive camera sensors are sampled only on requested
steps. Call `step(action, render=True)` immediately before `observe()`; forcing a render after a
physics step changes the observation timestamp relative to LeRobot.

Always-render is the default: on macOS, render history can change pixels by one intensity level
even at identical simulator states. Scheduled rendering is an experimental optimization only.

Built on lerobot's `LiberoEnv` so that initial states, settle steps, control mode and observation
formatting are exactly the ones `lerobot-eval` uses.
"""

from __future__ import annotations

import os
import sys
from functools import wraps

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


@dataclass
class StepResult:
    success: bool
    done: bool  # robosuite horizon reached


class LiberoSession:
    def __init__(
        self,
        suite_name: str,
        task_id: int,
        image_size: int = 256,
        render_mode: str = "always",
        reset_mode: str = "native",
    ):
        from libero.libero import benchmark
        from lerobot.sim.integrations.libero import LiberoEnv

        if render_mode not in ("always", "scheduled"):
            raise ValueError("render_mode must be always or scheduled")
        if reset_mode not in ("native", "isolated"):
            raise ValueError("reset_mode must be native or isolated")
        self.render_mode = render_mode
        self.reset_mode = reset_mode
        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.suite_name, self.task_id = suite_name, task_id
        self.env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            observation_height=image_size,
            observation_width=image_size,
            init_states=True,
        )
        self.task_description: str = self.env.task_description
        self.max_steps: int = int(self.env._max_episode_steps)
        self.n_init_states: int = len(self.env._init_states)
        self._cams: list[str] = []
        self._camera_sensors: dict = {}
        self._rendering = True
        self._observation: dict | None = None
        self.reset_history: list[int] = []
        self.last_reset_context: dict | None = None
        self._replay_reset_key = None
        self._replay_initializers: tuple = ()

    # ---- internals --------------------------------------------------------------------------
    @property
    def core(self):  # LIBERO ControlEnv (OffScreenRenderEnv)
        return self.env._env

    @property
    def rs(self):  # underlying robosuite environment
        return self.env._env.env

    def _set_rendering(self, on: bool) -> None:
        # Disabling observables also stops their clocks. Keep them enabled and active, and bypass
        # just the sensor, so requested frames have the exact standard-pipeline sampling time.
        self._rendering = on or self.render_mode == "always"

    def _wrap_camera_sensors(self) -> None:
        for name in self._cams:
            observable = self.rs._observables[name]
            # A hard reset keeps Observable objects but replaces their sensor callables.
            if self._camera_sensors.get(name) is observable._sensor:
                continue
            original = observable._sensor

            @wraps(original)
            def gated_sensor(cache, sensor=original, camera=observable):
                return sensor(cache) if self._rendering else camera.obs

            observable._sensor = gated_sensor
            self._camera_sensors[name] = gated_sensor

    # ---- episode API ------------------------------------------------------------------------
    def _fresh_core(self) -> None:
        self.env.close()
        self._camera_sensors.clear()
        self._cams = []
        self._observation = None
        self.reset_history = []
        self._replay_reset_key = None

    def reset(self, init_state_id: int) -> dict:
        """Reset to a fixed LIBERO initial state. Returns the formatted observation (with images)."""
        if not 0 <= init_state_id < self.n_init_states:
            raise ValueError("init_state_id is outside the available initial states")
        if self.reset_mode == "isolated":
            self._fresh_core()
        prior = self.reset_history.copy()
        if self.env._env is not None:
            self._set_rendering(True)
        self.env.init_state_id = int(init_state_id)
        obs, _ = self.env.reset(seed=int(init_state_id))
        self._cams = [k for k in self.rs._observables if k.endswith("_image")]
        self._wrap_camera_sensors()
        self._rendering = True
        self._set_rendering(False)
        self._observation = obs
        self.last_reset_context = {"mode": self.reset_mode, "prior_init_states": prior}
        self.reset_history.append(int(init_state_id))
        return obs

    def reset_for_replay(
        self, init_state_id: int, prior_init_states: tuple[int, ...]
    ) -> dict:
        """Reconstruct the native reset history, then restore its sampler list before each replay.

        LIBERO appends object_property_initializers on every hard reset. They consume random
        draws before fixture placement; reset(seed) alone does not reproduce the same scene.
        Warm the declared history once per selected state. Restore the pre-reset sampler list
        on subsequent branches, preserving the original native pipeline and its fixture poses.
        Every branch must still pass the exact physics-prefix and original-tail gates.
        """
        history = tuple(map(int, prior_init_states))
        if self.reset_mode == "isolated":
            if history:
                raise ValueError(
                    "Isolated replay requires an empty prior reset history"
                )
            return self.reset(init_state_id)
        key = (int(init_state_id), history)
        if self._replay_reset_key != key:
            self._fresh_core()
            for previous in history:
                self.reset(previous)
            self.env._ensure_env()
            self._replay_initializers = tuple(self.rs.object_property_initializers)
            self._replay_reset_key = key
        self.rs.object_property_initializers = list(self._replay_initializers)
        self.reset_history = list(history)
        return self.reset(init_state_id)

    def step(self, action: np.ndarray, *, render: bool = False) -> StepResult:
        """Advance physics; optionally sample cameras at the standard control-step timestamp."""
        self._set_rendering(render)
        try:
            raw, _, done, _ = self.core.step(np.asarray(action, dtype=np.float64))
            self._observation = self.env._format_raw_obs(raw) if render else None
        finally:
            self._set_rendering(False)
        return StepResult(success=bool(self.core.check_success()), done=bool(done))

    def observe(self) -> dict:
        """Return the observation from reset or the most recent explicitly rendered step."""
        if self._observation is None:
            raise RuntimeError(
                "Observation unavailable: request render=True on the preceding step"
            )
        return self._observation

    # ---- state save / restore (for counterfactual branching) ------------------------------------
    def get_state(self) -> np.ndarray:
        """Physics vector only: static model parameters and controller history are not included."""
        return np.array(self.core.get_sim_state(), dtype=np.float64).copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set physics only; this does not restore a valid full branch checkpoint."""
        self.core.set_state(np.asarray(state, dtype=np.float64))
        self.rs.sim.forward()
        self._observation = None

    def close(self) -> None:
        self.env.close()


def perturb_images(
    obs: dict, brightness: float, noise_sigma: float, rng: np.random.Generator
) -> dict:
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
