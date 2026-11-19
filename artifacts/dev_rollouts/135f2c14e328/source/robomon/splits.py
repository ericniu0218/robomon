"""Development/test task partition, created before new outcome inspection."""
import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPLIT_SEED = 20261110


def task_split() -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    suites = {}
    for suite in SUITES:
        # Task 0 is reserved for engineering checks. Spatial 0 was already exposed in the handoff.
        dev = sorted([0, *map(int, rng.choice(np.arange(1, 10), size=2, replace=False))])
        suites[suite] = {"dev": dev, "test": [i for i in range(10) if i not in dev]}
    return {"seed": SPLIT_SEED, "reserved_dev_task": 0, "suites": suites}
