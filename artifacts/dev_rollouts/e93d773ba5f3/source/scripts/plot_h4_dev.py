#!/usr/bin/env python3
"""Plot checked H4 development summaries, separating raw rates from paired contrast bounds."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robomon.utils.io import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_path = args.analysis / "results.json"
    data = json.loads(result_path.read_text())
    if data["status"] != "development only":
        raise ValueError("Expected H4 development summaries")
    destination = args.output or args.analysis / "figures" / "h4_pilot.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    colors = {"replan": "#0072B2", "rollback": "#D55E00"}
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    fig.suptitle(
        "Alarm and rollback development pilot", fontsize=17, fontweight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.91,
        f"{data['matched_sets']} matched sets · {data['states']} states · {data['replicates_per_state_arm']} paired seeds per state/arm",
        ha="center",
        fontsize=11,
    )
    n = data["matched_sets"] * data["replicates_per_state_arm"]
    for offset, arm in [(-0.18, "replan"), (0.18, "rollback")]:
        values = [data["rates"][f"{role}_{arm}"] for role in ("alarm", "control")]
        bars = axes[0].bar(
            [offset, 1 + offset],
            values,
            width=0.32,
            color=colors[arm],
            label="Replan" if arm == "replan" else "Oracle rollback",
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{round(value * n)}/{n}",
                ha="center",
                fontsize=10,
            )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Alarm states", "Exact controls"],
        ylim=(0, 1.14),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        ylabel="Branch success fraction",
    )
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    for y, key, label, color in [
        (
            1,
            "h4a_control_minus_alarm_replan",
            "Control − alarm\n(replan)",
            colors["replan"],
        ),
        (
            0,
            "h4b_alarm_rollback_minus_replan",
            "Rollback − replan\n(alarm states)",
            colors["rollback"],
        ),
    ]:
        row = data["contrasts"][key]
        axes[1].hlines(y, row["ci_low"], row["ci_high"], color=color, lw=2)
        axes[1].plot(row["estimate"], y, "o", color=color)
        axes[1].text(
            row["estimate"],
            y + 0.18,
            f"{row['estimate']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
            ha="center",
            fontsize=9,
        )
    axes[1].axvline(0, color="0.5", linestyle="--", lw=1)
    axes[1].set(
        xlim=(-1, 1),
        ylim=(-0.6, 1.6),
        yticks=[1, 0],
        yticklabels=["Control − alarm\n(replan)", "Rollback − replan\n(alarm states)"],
        xlabel="Success fraction difference",
    )
    axes[1].set_title("95% pointwise Monte Carlo bounds", fontsize=11, pad=12)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y" if axis is axes[0] else "x", color="0.9", lw=0.7)
    fig.text(
        0.5,
        0.025,
        "Development only. Bounds condition on fixed selected states; oracle rollback is simulation-only.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.16, wspace=0.53)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    write_json(
        {
            "scope": "development only",
            "analysis_results_sha": file_sha256(result_path),
            "plot_script_sha": file_sha256(Path(__file__)),
            "files": {destination.name: file_sha256(destination)},
        },
        destination.with_suffix(".provenance.json"),
    )
    print(destination)


if __name__ == "__main__":
    main()
