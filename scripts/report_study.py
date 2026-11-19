#!/usr/bin/env python3
"""Render frozen result JSONs into the core-study report and shareable scientific figures.

This script adds no hypothesis tests. All inferential decisions come from the frozen analysis.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robomon.study import load_frozen_protocol
from robomon.utils.io import ROOT, file_sha256, write_json

COLORS = {
    "elapsed": "#777777",
    "dispersion": "#D55E00",
    "consistency": "#0072B2",
    "support": "#009E73",
}


def number(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}f}"


def interval(row):
    return f"{number(row.get('estimate', row.get('coef')))} [{number(row.get('ci_low'))}, {number(row.get('ci_high'))}]"


def render_summary(primary, h4):
    """Describe the completed study in plain language using its frozen results."""
    h1 = primary["h1"]
    rates = h4["rates"]
    signal = {
        "consistency": "plan-change",
        "dispersion": "plan-disagreement",
        "support": "reference-distance",
    }[h1["signal"]]
    history = {"mean": "average", "max": "highest", "last": "latest"}[
        h1["reduction"]
    ]
    return "\n".join(
        [
            "# RoboMon — Robot Failure Monitoring",
            "",
            "## Question",
            "",
            "Can simple warning scores predict robot failure beyond knowing the task and how long the robot has been trying? Can acting on those warnings help it recover?",
            "",
            "## What I built",
            "",
            "I built an evaluation pipeline around SmolVLA, an existing robot-control model, and LIBERO, a simulated robot benchmark. It records robot behavior and measures four signals: elapsed time, disagreement among possible movement plans, changes in plans over time, and distance from reference demonstrations.",
            "",
            "## How I tested it",
            "",
            f"- Used **384 development attempts** to choose settings, then fixed the study rules before **{primary['episodes']:,} evaluation attempts** on 28 separate tasks with normal and degraded camera images.",
            "- Compared ongoing attempts at the **same task, camera condition, and timestep** to separate warning information from task difficulty and elapsed time.",
            f"- Ran **{h4['branches']:,} alternative continuations** from {h4['states']} selected alarm/control states, comparing fresh replanning with rewinding 50 actions. Both strategies received the same remaining action budget.",
            f"- Found a simulator reset bug that changed fixture positions. I corrected the reset procedure and verified exact replay for all {h4['gates_passed']} selected states before testing interventions.",
            "",
            "## Main findings",
            "",
            f"- **Useful warnings:** The {history} {signal} score ranked failed attempts above successful ones in **{h1['estimate']:.1%} of matched comparisons** (95% interval: {h1['ci_low']:.1%}–{h1['ci_high']:.1%}).",
            "- **Reference similarity added information:** Distance from demonstrations remained associated with failure after accounting for both plan-disagreement signals in the statistical model.",
            f"- **Alarms identified difficult states:** Fresh replanning succeeded in **{rates['alarm_replan']:.1%}** of alarm-state continuations versus **{rates['control_replan']:.1%}** at matched controls.",
            f"- **Recovery remains unresolved:** Rewinding raised observed alarm-state success to **{rates['alarm_rollback']:.1%}**, but the uncertainty was too large to establish a reliable improvement.",
            "",
            "![Warning quality and intervention effects](figures/summary_results.png)",
            "",
            "Left: warning quality after matching; 50% is chance. Right: differences in success rates; crossing zero leaves the benefit uncertain. Bars show 95% intervals; recovery bounds apply only to the selected states.",
            "",
            "## Limits",
            "",
            "This is an independent study of one pretrained model in simulation. Evaluation tasks were held out from monitor tuning but were familiar to the policy. Recovery results concern the selected states; physical-robot recovery and safety were not tested.",
            "",
            "[Full report and figures](FINAL_REPORT.md) · [Study methods](CONFIRMATORY_METHODS.md)",
            "",
        ]
    )


def plot_summary(primary, h4, protocol, path):
    """Compact overview of matched prediction and paired intervention results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
    labels = []
    for y, (signal, label) in enumerate(
        (
            ("elapsed", "Time alone"),
            ("dispersion", "Plan disagreement"),
            ("support", "Reference distance"),
            ("consistency", "Plan changes"),
        )
    ):
        labels.append(label)
        reduction = (
            "max"
            if signal == "elapsed"
            else protocol["per_signal_reductions"][signal]["reduction"]
        )
        row = next(
            row
            for row in primary["audit"]
            if row["matching"] == "time_task_condition"
            and row["signal"] == signal
            and row["reduction"] == reduction
        )
        axes[0].errorbar(
            100 * row["estimate"],
            y,
            xerr=[
                [100 * (row["estimate"] - row["ci_low"])],
                [100 * (row["ci_high"] - row["estimate"])],
            ],
            fmt="o",
            color=COLORS[signal],
            markersize=6,
            capsize=3,
            lw=2,
        )
    axes[0].axvline(50, color="0.6", ls="--", lw=1)
    axes[0].set(
        yticks=range(4),
        yticklabels=labels,
        xticks=[50, 60, 70, 80],
        xlim=(45, 83),
        ylim=(-0.5, 3.5),
        xlabel="Failure-ranking score (%)",
        title="A. Warning quality",
    )
    for y, name, color in (
        (1, "h4a_control_minus_alarm_replan", COLORS["consistency"]),
        (0, "h4b_alarm_rollback_minus_replan", COLORS["dispersion"]),
    ):
        row = h4["contrasts"][name]
        axes[1].errorbar(
            100 * row["estimate"],
            y,
            xerr=[
                [100 * (row["estimate"] - row["ci_low"])],
                [100 * (row["ci_high"] - row["estimate"])],
            ],
            fmt="o",
            color=color,
            markersize=6,
            capsize=3,
            lw=2,
        )
    axes[1].axvline(0, color="0.6", ls="--", lw=1)
    axes[1].set(
        yticks=[0, 1],
        yticklabels=["Rewind vs replan\n(at alarms)", "Control vs alarm\n(fresh replanning)"],
        xticks=[0, 25, 50, 75],
        xlim=(-15, 85),
        ylim=(-0.65, 1.65),
        xlabel="Success difference (percentage points)",
        title="B. Intervention effects",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="0.92", lw=0.7)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.83, wspace=0.95)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_audit(primary, protocol, path):
    fig, axis = plt.subplots(figsize=(10, 6))
    levels = [
        ("naive", "Full-episode\nmaximum"),
        ("time", "Time"),
        ("time_condition", "Time + condition"),
        ("time_task_condition", "Task + condition\n+ time"),
    ]
    for offset, signal in zip(
        (-0.09, -0.03, 0.03, 0.09), ("elapsed", "dispersion", "consistency", "support")
    ):
        reduction = (
            "max"
            if signal == "elapsed"
            else protocol["per_signal_reductions"][signal]["reduction"]
        )
        values = []
        for index, (matching, _) in enumerate(levels):
            target = "episode_max" if matching == "naive" else reduction
            row = next(
                item
                for item in primary["audit"]
                if item["signal"] == signal
                and item["matching"] == matching
                and item["reduction"] == target
            )
            value = row["estimate"]
            values.append(np.nan if value is None else value)
            if row["ci_low"] is not None:
                axis.vlines(
                    index + offset,
                    row["ci_low"],
                    row["ci_high"],
                    color=COLORS[signal],
                    alpha=0.75,
                    lw=1.5,
                )
        label = signal if signal == "elapsed" else f"{signal} ({reduction})"
        axis.plot(
            np.arange(4) + offset,
            values,
            "o-",
            color=COLORS[signal],
            label=label,
            lw=1.5,
        )
    axis.axhline(0.5, color="0.5", ls="--", lw=1)
    axis.set(
        xticks=np.arange(4),
        xticklabels=[label for _, label in levels],
        ylim=(0, 1.03),
        ylabel="AUROC / pair-weighted concordance",
        title="Held-out failure-monitor audit",
    )
    axis.legend(loc="lower left", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.025,
        "Frozen dev-selected reductions; matching changes risk sets and pair weights. Pointwise 95% intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_h3(primary, path):
    h3 = primary["h3"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.axvline(0, color="0.5", ls="--", lw=1)
    for y, signal in enumerate(("consistency", "dispersion", "support")):
        row = (h3.get("coefficients") or {}).get(signal + "__last")
        if row:
            axis.plot(row["coef"], y, "o", color=COLORS[signal])
            if row["ci_low"] is not None:
                axis.hlines(
                    y, row["ci_low"], row["ci_high"], color=COLORS[signal], lw=2
                )
        else:
            axis.text(0.05, y, "Inconclusive", transform=axis.get_yaxis_transform())
    axis.set(
        yticks=[0, 1, 2],
        yticklabels=["consistency", "dispersion", "support"],
        ylim=(-0.5, 2.5),
        xlabel="Conditional log-odds coefficient per frozen development SD",
        title="Held-out conditional association (last-value features)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.025,
        f"Pairwise composite likelihood; basic paired-unit bootstrap. Numerical status: {h3['status']}.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_h4(result, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.suptitle(
        "Held-out alarm states and oracle rollback",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    count = result["matched_sets"] * result.get("replicates_per_state_arm", 3)
    if count:
        for offset, arm, color in (
            (-0.18, "replan", "#0072B2"),
            (0.18, "rollback", "#D55E00"),
        ):
            values = [
                result["rates"][role + "_" + arm] for role in ("alarm", "control")
            ]
            bars = axes[0].bar(
                [offset, 1 + offset],
                values,
                width=0.32,
                color=color,
                label="Replan" if arm == "replan" else "Oracle rollback",
            )
            for bar, value in zip(bars, values):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{round(value * count)}/{count}",
                    ha="center",
                    fontsize=10,
                )
        for y, key, color in (
            (1, "h4a_control_minus_alarm_replan", "#0072B2"),
            (0, "h4b_alarm_rollback_minus_replan", "#D55E00"),
        ):
            row = result["contrasts"][key]
            axes[1].hlines(y, row["ci_low"], row["ci_high"], color=color, lw=2)
            axes[1].plot(row["estimate"], y, "o", color=color)
            axes[1].text(
                0.5,
                0.83 if y else 0.37,
                interval(row),
                transform=axes[1].transAxes,
                ha="center",
                fontsize=9,
            )
        axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No exact matched states",
            transform=axes[0].transAxes,
            ha="center",
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Alarm states", "Exact controls"],
        ylim=(0, 1.14),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        ylabel="Branch success fraction",
    )
    axes[1].axvline(0, color="0.5", ls="--", lw=1)
    axes[1].set(
        xlim=(-1, 1),
        ylim=(-0.6, 1.6),
        yticks=[1, 0],
        yticklabels=["Control − alarm\n(replan)", "Rollback − replan\n(alarm states)"],
        xlabel="Success fraction difference",
        title="95% fixed-state Monte Carlo bounds",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.91,
        f"{result['matched_sets']} matched sets · {result['states']} states · {count} independent paired seed blocks",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Fixed selected states only. Confirmatory claims follow the frozen hierarchy; oracle rollback is simulation-only.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.16, wspace=0.53)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--h4", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_frozen_protocol(args.protocol, args.receipt)
    primary = json.loads((args.analysis / "results.json").read_text())
    h4 = json.loads((args.h4 / "results.json").read_text())
    hm = json.loads((args.h4 / "manifest.json").read_text())["config"]
    if (
        hm["primary_results_sha"] != file_sha256(args.analysis / "results.json")
        or hm["primary_manifest_sha"] != file_sha256(args.analysis / "manifest.json")
        or hm["protocol_sha"] != protocol["_protocol_sha256"]
    ):
        raise ValueError("Results do not belong to one frozen analysis")
    selection = json.loads((args.selection / "results.json").read_text())
    if primary["episodes"] != 896 or primary["paired_scenes_verified"] != 448:
        raise ValueError("Incomplete held-out cohort")
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    paths = [
        figures / "final_audit.png",
        figures / "final_h3.png",
        figures / "final_h4.png",
        figures / "summary_results.png",
    ]
    plot_audit(primary, protocol, paths[0])
    plot_h3(primary, paths[1])
    plot_h4(h4, paths[2])
    plot_summary(primary, h4, protocol, paths[3])
    decisions = h4["sequence_decisions"]
    supported = [
        row["hypothesis"] for row in decisions if row["decision"] == "supported"
    ]
    decision_map = {row["hypothesis"]: row["decision"] for row in decisions}
    conclusions = [
        "The frozen sequence supports "
        + (", ".join(supported) if supported else "no positive hypothesis")
        + "."
    ]
    if decision_map["H1"] == "supported":
        conclusions.append(
            "The frozen signal retains predictive association after matching."
        )
    if decision_map["H3"] == "supported":
        conclusions.append(
            "Training-source support adds positive conditional association within the declared working model."
        )
    if h4["rates"]:
        conclusions.append(
            f"Fresh-replan branch success is {h4['rates']['alarm_replan']:.1%} at alarm states versus {h4['rates']['control_replan']:.1%} at controls. Alarm rollback branch success is {h4['rates']['alarm_rollback']:.1%}; its success-fraction difference bound is {interval(h4['contrasts']['h4b_alarm_rollback_minus_replan'])}."
        )
        conclusions.append(
            "A positive recovery benefit is "
            + ("supported" if decision_map["H4b"] == "supported" else "not established")
            + " by the frozen test."
        )
    else:
        conclusions.append(
            "No exact matched states were available; intervention effects are inconclusive."
        )
    lines = [
        f"# RoboMon core-study report — {datetime.now(ZoneInfo('America/Los_Angeles')).date()}",
        "",
        "The core H1–H4 study is complete. H2 is descriptive; the optional H5 stretch was excluded before testing.",
        "",
        " ".join(conclusions),
        "",
        f"**Frozen before test outcomes:** `{protocol['_freeze_commit']}`. The committed protocol and methods are in [frozen_protocol.json](frozen_protocol.json) and [CONFIRMATORY_METHODS.md](CONFIRMATORY_METHODS.md).",
        "",
        "## Prespecified results",
        "",
        "| Hypothesis | Estimate and two-sided 95% interval/bound | Fixed-sequence decision |",
        "|---|---|---|",
    ]
    for decision in decisions:
        row = decision["interval"] or {}
        lines.append(
            f"| {decision['hypothesis']} | {interval(row)} | {decision['decision'].replace('_', ' ')} |"
        )
    lines += [
        "",
        "The confirmatory sequence is H1 → H3 → H4a → H4b. After its first failure or inconclusive result, later intervals are descriptive only. Bootstrap coverage and nominal family-wise control are approximate; H4 bounds cover only branch-noise Monte Carlo error conditional on selected states.",
        "",
        "## Held-out prediction audit",
        "",
        "Collected **896 episodes**: 28 monitor-test tasks × 2 conditions × 16 initial-state/policy-seed pairs. All **448 paired initial physics/static scenes** agree exactly. SmolVLA was trained on these task identities; holdout refers to monitor tuning.",
        "",
        f"The frozen H1 choice is **{protocol['h1']['signal']} / {protocol['h1']['reduction']}**, selected solely by the maximum development primary point estimate. Its primary concordance is **{interval(primary['h1'])}**, using {primary['h1']['pairs']:,.0f} comparable pairs and {primary['h1']['informative_strata']} informative strata of {primary['h1']['total_strata']}.",
        "",
        "![Held-out audit](figures/final_audit.png)",
        "",
        "Naive scores use full-episode maxima; matched scores use the frozen causal reduction chosen for each signal on development data. Matching changes risk sets and pair weights, so differences are descriptive rather than a causal decomposition. All signals/reductions and strata/degenerate-draw counts remain in the analysis artifact.",
        "",
        "| Condition | Successes / episodes | Success fraction |",
        "|---|---:|---:|",
    ]
    # Values are read verbatim from the frozen audit; rendering adds no tests or selection.
    insertion = lines.index("| Condition | Successes / episodes | Success fraction |")
    table = [
        "| Signal (frozen causal reduction) | Naive episode max | Time | Time + condition | Task + condition + time |",
        "|---|---:|---:|---:|---:|",
    ]
    for signal in ("elapsed", "dispersion", "consistency", "support"):
        reduction = (
            "max"
            if signal == "elapsed"
            else protocol["per_signal_reductions"][signal]["reduction"]
        )
        values = []
        for matching in ("naive", "time", "time_condition", "time_task_condition"):
            row = next(
                item
                for item in primary["audit"]
                if item["signal"] == signal
                and item["matching"] == matching
                and item["reduction"]
                == ("episode_max" if matching == "naive" else reduction)
            )
            values.append(interval(row))
        label = signal if signal == "elapsed" else signal + " / " + reduction
        table.append("| " + label + " | " + " | ".join(values) + " |")
    lines[insertion:insertion] = table + [""]
    for condition in ("nominal", "shift"):
        cells = [cell for cell in primary["coverage"] if cell["condition"] == condition]
        n = sum(cell["episodes"] for cell in cells)
        success = sum(cell["successes"] for cell in cells)
        lines.append(f"| {condition} | {success}/{n} | {success / n:.1%} |")
    lines += [
        "",
        f"Visual shift: brightness {protocol['visual_shift']['brightness']}, Gaussian pixel-noise σ={protocol['visual_shift']['noise_sigma']} on the 0–255 scale. Its strength was frozen from isolated-reset development calibration, not adjusted to held-out success.",
        "",
        "The 50–60% calibration target applied to development; the held-out task mix was collected at exactly that frozen shift strength. No held-out retuning or outcome-based replacement occurred.",
        "",
        "## Conditional training-source support association",
        "",
        "![Conditional association](figures/final_h3.png)",
        "",
        "| Last-value feature | Coefficient per frozen dev SD [95% basic-bootstrap interval] |",
        "|---|---|",
    ]
    for signal in ("support", "dispersion", "consistency"):
        row = (primary["h3"].get("coefficients") or {}).get(signal + "__last", {})
        lines.append(f"| {signal} | {interval(row)} |")
    lines += [
        "",
        f"Numerical status: **{primary['h3']['status']}**; {primary['h3'].get('bootstrap_valid', 0):,} / 2,000 valid bootstrap fits. H1 has {primary['h1']['bootstrap_valid']:,} / 2,000 valid draws. This is a pairwise conditional logistic composite likelihood, controlling task/condition/time and two measured linear uncertainty proxies. A support coefficient is an association within this working model, not proof of causal training-data knowledge or complete model uncertainty. The bank measures distance to a pinned version of the named training source; exact historical training bytes are unknown.",
        "",
        "## Frozen selected-state interventions",
        "",
        f"Outcome-blind matching selected **{h4['matched_sets']} sets / {h4['states']} states** from {sum(cell['selected_pairs'] > 0 for cell in selection['coverage'])}/56 task/condition cells. All **{h4['gates_passed']} known-answer gates** passed before {h4['branches']:,} intervention branches. Missing alarms/controls are reported without replacement. Three seed blocks per matched set pair both states and both arms; arms receive equal remaining forward-action budgets.",
        "",
        "![Held-out intervention results](figures/final_h4.png)",
        "",
        "| State / arm | Success fraction |",
        "|---|---:|",
    ]
    for name, value in h4["rates"].items():
        lines.append(f"| {name.replace('_', ' ')} | {value:.1%} |")
    lines += [
        "",
        "| Contrast | Estimate [95% fixed-state MC bound] | Positive / zero / negative seed-block differences |",
        "|---|---|---:|",
    ]
    for name, row in h4["contrasts"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {interval(row)} | {row.get('positive', '—')} / {row.get('zero', '—')} / {row.get('negative', '—')} |"
        )
    lines += [
        "",
        "H4a compares fresh-replan success at exact controls versus alarm states. H4b compares oracle rollback versus fresh replanning at alarm states. Bounds are conservative Hoeffding intervals on independent paired seed-block differences; they do not cover state selection or new-task transfer. A wide bound is inconclusive rather than evidence of no effect. Oracle reset/replay is simulation-only and does not measure physical recovery costs.",
        "",
        "## Development history, scope and reproducibility",
        "",
        "The historical native-reset audit exposed a LIBERO sampler-history bug. It changed static fixture positions outside qpos/qvel. The failed partial pilot was preserved and excluded; [RESET_AUDIT.md](RESET_AUDIT.md) provides a minimal independent repro. The full replacement development calibration and held-out study use isolated cores, exact paired-scene checks, and fresh cores for every intervention branch. Historical and replacement cohorts are not pooled.",
        "",
        "This is an independent audit/replication using simple inspired baselines, not a new detector or a reproduction of every published method. [LITERATURE.md](LITERATURE.md) records the checked positioning. Results concern the declared tasks, initial states, checkpoint, visual condition and native MPS stack, not arbitrary robot tasks or real-robot safety.",
        "",
        f"Primary artifact: `{args.analysis}`. Intervention artifact: `{args.h4}`. Selection: `{args.selection}`. Model/tokenizer, simulator/controller, assets, BDDL and initial-state hashes are frozen in the protocol. Exact source archives, atomic transactions and checked resume preserve launch-time code; raw arrays/archives stay local under repository policy. Figures have input/source/PNG provenance.",
        "",
    ]
    destination = ROOT / "FINAL_REPORT.md"
    destination.write_text("\n".join(lines))
    summary_path = ROOT / "SUMMARY.md"
    summary_path.write_text(render_summary(primary, h4))
    provenance = {
        "protocol_sha": protocol["_protocol_sha256"],
        "freeze_commit": protocol["_freeze_commit"],
        "primary_results_sha": file_sha256(args.analysis / "results.json"),
        "h4_results_sha": file_sha256(args.h4 / "results.json"),
        "selection_results_sha": file_sha256(args.selection / "results.json"),
        "report_script_sha": file_sha256(Path(__file__)),
        "primary_manifest_sha": file_sha256(args.analysis / "manifest.json"),
        "h4_manifest_sha": file_sha256(args.h4 / "manifest.json"),
        "selection_manifest_sha": file_sha256(args.selection / "manifest.json"),
        "render_runtime": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "summary_sha": file_sha256(summary_path),
        "files": {path.name: file_sha256(path) for path in paths},
        "report_sha": file_sha256(destination),
    }
    write_json(provenance, figures / "final_provenance.json")
    print(destination)


if __name__ == "__main__":
    main()
