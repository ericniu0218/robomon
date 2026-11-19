#!/usr/bin/env python3
"""Render development figures from one explicitly identified, completed analysis artifact."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robomon.utils.io import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--reduction", choices=["max", "mean", "last"], default="max")
    args = parser.parse_args()
    source = args.analysis / "results.json"
    data = json.loads(source.read_text())
    if data["scope"] != "development only":
        raise ValueError("Expected development analysis")
    output = args.analysis / "figures"
    output.mkdir(exist_ok=True)
    cells = data["coverage"]["cells"]
    tasks = sorted({row["task"] for row in cells})
    conditions = ["nominal", "shift"]
    lookup = {(row["task"], row["condition"]): row for row in cells}
    matrix = np.full((len(tasks), 2), np.nan)
    for i, task in enumerate(tasks):
        for j, condition in enumerate(conditions):
            row = lookup.get((task, condition))
            if row:
                matrix[i, j] = row["successes"] / row["episodes"]
    fig, ax = plt.subplots(figsize=(6.4, 6.8), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(2), ["Nominal", "Visual shift"])
    ax.set_yticks(
        range(len(tasks)), [t.replace("libero_10", "long").replace("libero_", "").replace("/", " · ") for t in tasks]
    )
    for i, task in enumerate(tasks):
        for j, condition in enumerate(conditions):
            row = lookup.get((task, condition))
            if row:
                rgb = np.asarray(image.cmap(image.norm(matrix[i, j])))[:3]
                linear = np.where(
                    rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
                )
                luminance = float(np.sum(linear * np.array([0.2126, 0.7152, 0.0722])))
                ax.text(
                    j,
                    i,
                    f"{row['successes']}/{row['episodes']}",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.179 else "white",
                    fontsize=10,
                )
    ax.set_title("Development task success", loc="left", fontweight="bold", pad=14)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04, label="Success fraction")
    fig.savefig(output / "task_success.png", dpi=180)
    plt.close(fig)

    colors = ["#0072B2", "#D55E00", "#009E73", "#000000"]
    markers = ["o", "s", "^", "D"]
    signals = ["dispersion", "consistency", "support", "elapsed"]
    labels = [
        "Action dispersion",
        "Chunk consistency",
        "Demo support distance",
        "Elapsed steps",
    ]
    matching = ["naive", "time", "time_condition", "time_task_condition"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), layout="constrained")
    for i, (signal, label, color, marker) in enumerate(
        zip(signals, labels, colors, markers, strict=True)
    ):
        selected = [
            next(
                row
                for row in data["audit"]
                if row["signal"] == signal
                and row["matching"] == kind
                and row["reduction"] == ("episode_max" if kind == "naive" else args.reduction)
            )
            for kind in matching
        ]
        values = np.array(
            [np.nan if row["estimate"] is None else row["estimate"] for row in selected]
        )
        low = np.array(
            [np.nan if row["ci_low"] is None else row["ci_low"] for row in selected]
        )
        high = np.array(
            [np.nan if row["ci_high"] is None else row["ci_high"] for row in selected]
        )
        x = np.arange(4) + (i - 1.5) * 0.055
        ax.plot(
            x,
            values,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=6,
        )
        ax.vlines(x, low, high, color=color, linewidth=1.2)
        for xx, lo, hi in zip(x, low, high, strict=True):
            ax.hlines([lo, hi], xx - 0.025, xx + 0.025, color=color, linewidth=1.2)
    ax.axhline(0.5, color=".5", linestyle="--", linewidth=1)
    ax.set_xticks(
        range(4),
        [
            "Episode maximum",
            "Same landmark",
            "Same landmark\nand condition",
            "Same landmark,\ntask and condition",
        ],
    )
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Failure concordance (AUROC)")
    fig.suptitle("Development audit", x=.06, ha="left", fontweight="bold", fontsize=14)
    ax.set_title(
        f"Matched estimates use causal {args.reduction} values at common landmarks.",
        loc="left", fontsize=9, pad=10,
    )
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.savefig(output / f"audit_{args.reduction}.png", dpi=180)
    plt.close(fig)
    write_json(
        {
            "source_results_sha": file_sha256(source),
            "reduction": f"{args.reduction} for matched comparisons",
            "scope": "development only",
            "palette": colors,
            "files": {
                name: file_sha256(output / name)
                for name in ("task_success.png", f"audit_{args.reduction}.png")
            },
        },
        output / f"provenance_{args.reduction}.json",
    )
    print(output)


if __name__ == "__main__":
    main()
