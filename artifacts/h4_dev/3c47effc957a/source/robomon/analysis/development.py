"""Development-only scoring with explicit inputs, common landmarks, and no automatic H1 selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robomon.analysis.landmarks import bootstrap_concordance, episode_max
from robomon.data.demos import dev_task_keys
from robomon.monitors.bank import SupportBank
from robomon.monitors.signals import history_score
from robomon.policy.smolvla import (
    CHECKPOINT_REVISION,
    EMBEDDING_DEFINITION,
    VLM_REVISION,
)
from robomon.rollout import EpisodeSpec
from robomon.utils.artifacts import completed_episode
from robomon.utils.io import config_hash

SIGNALS = ("elapsed", "dispersion", "consistency", "support")


def score_completed_run(
    run: Path, bank: SupportBank
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return score_run(
        run, bank, allowed_tasks=dev_task_keys(), expected_stage="dev_rollouts"
    )


def score_run(
    run: Path, bank: SupportBank, *, allowed_tasks: list[str], expected_stage: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Low-level scorer. Held-out callers must first verify a committed frozen protocol."""
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if (
        config.get("stage") != expected_stage
        or config_hash(config) != manifest["config_hash"]
    ):
        raise ValueError("Expected a verified development rollout manifest")
    for key, expected in {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "vlm_revision": VLM_REVISION,
        "embedding_definition": EMBEDDING_DEFINITION,
    }.items():
        if config.get(key) != expected or bank.config.get(key) != expected:
            raise ValueError(
                f"Incompatible rollout/bank {key}; legacy embeddings require explicit revalidation"
            )
    if config["args"]["device"] != bank.config["device"]:
        raise ValueError(
            "Rollout and demonstration embeddings must use the same device"
        )
    for package in ("torch", "torchvision", "lerobot", "transformers"):
        if (
            config["runtime"]["packages"][package]
            != bank.config["runtime"]["packages"][package]
        ):
            raise ValueError(f"Rollout/bank package mismatch: {package}")
    results = json.loads((run / "results.json").read_text())
    if results.get("status") != "complete" or results.get("completed") != len(
        config["episodes"]
    ):
        raise ValueError("Development run is incomplete")
    rows, summaries = [], []
    for episode in config["episodes"]:
        spec = EpisodeSpec(**episode)
        if spec.task not in allowed_tasks:
            raise ValueError(
                "Development analysis cannot inspect held-out task outcomes"
            )
        directory = run / "episodes" / config_hash(episode)
        if not completed_episode(directory, config):
            raise ValueError(f"Incomplete episode: {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        data = pd.read_parquet(directory / "events.parquet")
        if summary["episode_id"] != spec.episode_id or not summary["monitor"]:
            raise ValueError("Wrong episode or missing monitor data")
        if (
            data.empty
            or not data.episode_id.eq(spec.episode_id).all()
            or not data.task.eq(spec.task).all()
            or not data.condition.eq(spec.condition).all()
            or not data.failure.eq(int(not summary["success"])).all()
        ):
            raise ValueError("Events disagree with episode identity or outcome")
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            embeddings = arrays["embeddings"]
            if (
                embeddings.shape != (len(data), 960)
                or not np.array_equal(arrays["query_steps"], data.landmark.to_numpy())
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Query embeddings do not align with landmark rows")
            data["support"] = [bank.score(vector, spec.task) for vector in embeddings]
        if (
            data.landmark >= summary["steps"]
        ).any() or not data.landmark.is_monotonic_increasing:
            raise ValueError("Noncausal or terminal landmark")
        rows.append(data)
        summaries.append(summary)
    joined = pd.concat(rows, ignore_index=True)
    if joined.duplicated(["episode_id", "landmark"]).any():
        raise ValueError("Repeated episode in run selection")
    return joined, pd.DataFrame(summaries), config


def add_history_reductions(rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, episode in rows.groupby("episode_id", sort=True):
        episode = episode.sort_values("landmark").copy()
        for signal in SIGNALS:
            values = episode[signal].to_numpy()
            for reduction in ("max", "mean", "last"):
                episode[f"{signal}__{reduction}"] = [
                    history_score(values[: i + 1], reduction)
                    for i in range(len(values))
                ]
        pieces.append(episode)
    return pd.concat(pieces, ignore_index=True)


def audit_tables(
    rows: pd.DataFrame, *, n_boot: int = 1000, seed: int = 20261111
) -> list[dict]:
    """Use landmarks with all four raw signals available for every matched comparison.

    Raw episode maxima define the naive comparator, including all available queries. Time/task
    estimates use causal max/mean/last histories at common landmarks. CIs preserve condition pairs.
    """
    if not np.isfinite(rows[["elapsed", "dispersion", "support"]].to_numpy()).all():
        raise ValueError("Expected finite elapsed, dispersion and support signals")
    enriched = add_history_reductions(rows)
    common = enriched.dropna(subset=list(SIGNALS)).copy()
    output = []
    for signal in SIGNALS:
        naive = episode_max(rows, signal)
        output.append(
            {
                "signal": signal,
                "reduction": "episode_max",
                "matching": "naive",
                **bootstrap_concordance(
                    naive,
                    signal,
                    (),
                    n_boot=n_boot,
                    seed=seed,
                    cluster="pair_id",
                    cluster_universe=rows,
                ),
            }
        )
        for reduction in ("max", "mean", "last"):
            column = f"{signal}__{reduction}"
            for name, strata in [
                ("time", ("landmark",)),
                ("time_condition", ("condition", "landmark")),
                ("time_task_condition", ("task", "condition", "landmark")),
            ]:
                estimate = bootstrap_concordance(
                    common,
                    column,
                    strata,
                    n_boot=n_boot,
                    seed=seed,
                    cluster="pair_id",
                    cluster_universe=rows,
                )
                if (
                    signal == "elapsed"
                    and estimate["pairs"]
                    and estimate["estimate"] != 0.5
                ):
                    raise AssertionError("Elapsed-time negative control failed")
                output.append(
                    {
                        "signal": signal,
                        "reduction": reduction,
                        "matching": name,
                        **estimate,
                    }
                )
    return output


def coverage_summary(summaries: pd.DataFrame) -> dict:
    rows = []
    for (task, condition), group in summaries.groupby(["task", "condition"], sort=True):
        rows.append(
            {
                "task": task,
                "condition": condition,
                "episodes": len(group),
                "successes": int(group.success.sum()),
                "failures": int((~group.success).sum()),
                "mean_steps": float(group.steps.mean()),
            }
        )
    present = {(row["task"], row["condition"]) for row in rows}
    required = {
        (task, condition)
        for task in dev_task_keys()
        for condition in ("nominal", "shift")
    }
    failure_rate = float((~summaries.success).mean())
    enough = required <= present and all(row["episodes"] >= 16 for row in rows)
    return {
        "cells": rows,
        "episodes": len(summaries),
        "failure_rate": failure_rate,
        "all_dev_conditions_present": required <= present,
        "planned_dev_sample_complete": enough,
        "failure_rate_gate_20_to_80_percent": 0.2 <= failure_rate <= 0.8,
        "eligible_for_frozen_signal_selection": enough and 0.2 <= failure_rate <= 0.8,
        "selected_signal": None,
    }


def development_results(rows, summaries, *, n_boot=1000, seed=20261111):
    audit = audit_tables(rows, n_boot=n_boot, seed=seed)
    coverage = coverage_summary(summaries)
    primary = [r for r in audit if r["matching"] == "time_task_condition"]
    evaluable = bool(primary) and all(
        r["pairs"] > 0
        and np.isfinite(r["estimate"])
        and r["bootstrap_valid"] >= 0.9 * n_boot
        for r in primary
    )
    coverage["primary_evaluable"] = evaluable
    coverage["eligible_for_frozen_signal_selection"] &= evaluable
    return {"scope": "development only", "coverage": coverage, "audit": audit}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
