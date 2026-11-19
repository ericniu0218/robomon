"""Reproduce the render-history diagnostic on development object task 0.

Run from robomon with `.venv/bin/python diagnostics/render_history.py`.
Compare identical replayed actions, not separately sampled policies. No STATE output means all
physics states agreed within 1e-10. This is a diagnostic, not a required passing optimization test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import numpy as np
import torch
from robomon.policy.smolvla import SmolVLARunner
from robomon.envs.libero_session import LiberoSession
from robomon.rollout import derived_seed

p = SmolVLARunner(threads=4)
s = LiberoSession("libero_object", 0, render_mode="scheduled")
seed = 8961987772630546032
obs = s.reset(0)
images = {0: deepcopy(obs)}
actions = []
states = [s.get_state()]
for t in range(150):
    if t % 50 == 0:
        if t:
            obs = s.observe()
            images[t] = deepcopy(obs)
        q = p.query(
            obs,
            s.task_description,
            1,
            torch.Generator().manual_seed(derived_seed(seed, "execution", t)),
        )
    a = q.exec_actions[t % 50]
    s.step(a, render=(t + 1) % 50 == 0)
    actions.append(a.copy())
    states.append(s.get_state())
obs = s.reset(0)
s._set_rendering(True)
print("RESET_DIFF", np.max(abs(states[0] - s.get_state())), flush=True)
for t, a in enumerate(actions):
    if t % 50 == 0:
        print(
            "OBS",
            t,
            {
                cam: int(
                    np.abs(
                        obs["pixels"][cam].astype(int) - images[t]["pixels"][cam]
                    ).max()
                )
                for cam in obs["pixels"]
            },
            "EEF",
            np.max(
                abs(
                    obs["robot_state"]["eef"]["pos"]
                    - images[t]["robot_state"]["eef"]["pos"]
                )
            ),
            flush=True,
        )
    obs, *_ = s.env.step(a.astype(np.float64))
    diff = np.max(abs(states[t + 1] - s.get_state()))
    if diff > 1e-10:
        print("STATE", t + 1, diff, flush=True)
s.close()
# Compare two full-render replays of exactly the same actions as a determinism control.
s = LiberoSession("libero_object", 0, render_mode="always")
captured = []
for repeat in range(2):
    obs = s.reset(0)
    s._set_rendering(True)
    current = {0: deepcopy(obs)}
    for t, a in enumerate(actions):
        obs, *_ = s.env.step(a.astype(np.float64))
        if (t + 1) % 50 == 0:
            current[t + 1] = deepcopy(obs)
    captured.append(current)
for t in (0, 50, 100, 150):
    for cam in captured[0][t]["pixels"]:
        x = captured[0][t]["pixels"][cam].astype(int) - captured[1][t]["pixels"][
            cam
        ].astype(int)
        print(
            "ALWAYS_REPEAT",
            t,
            cam,
            "max",
            np.abs(x).max(),
            "count",
            np.count_nonzero(x),
            flush=True,
        )
s.close()
