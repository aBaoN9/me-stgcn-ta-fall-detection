from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs/images"
VARIANTS = ["A0_pose_average", "A1_motion_average", "A2_pose_attention", "A3_motion_attention", "A3_cost_225"]
LABELS = ["A0 · Pose", "A1 · Motion", "A2 · Pose + TA", "A3 · Motion + TA", "A3 · Cost 2.25"]
COLORS = ["#80949a", "#416f89", "#6b799e", "#177668", "#b57627"]


def save(figure, name):
    figure.savefig(TARGET / name, dpi=150, facecolor="#f5f8f7", bbox_inches="tight")
    plt.close(figure)




def result_figures():
    table = pd.read_csv(ROOT / "results/component_ablation/pooled_by_seed.csv")
    fixed = table[table.policy == "fixed"]
    statistics = fixed.groupby("variant")[["accuracy", "sensitivity", "specificity"]].agg(["mean", "std"]).reindex(VARIANTS) * 100
    figure, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    positions = np.arange(len(VARIANTS))
    for axis, metric, heading in zip(axes, ["accuracy", "sensitivity", "specificity"], ["Accuracy", "Sensitivity · Fall", "Specificity · ADL"]):
        means = statistics[(metric, "mean")].to_numpy()
        deviations = statistics[(metric, "std")].to_numpy()
        axis.barh(positions, means, color=COLORS, height=0.6, xerr=deviations, capsize=4, error_kw={"ecolor": "#233d46", "elinewidth": 1.2})
        axis.set_xlim(0, 108)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("Percent · axis starts at zero", fontsize=9)
        axis.set_title(heading, loc="left", size=14, weight="bold", pad=18)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.set_yticks(positions, LABELS)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for position, mean in zip(positions, means):
            axis.text(2, position, f"{mean:.2f}%", va="center", color="white", size=11, weight="bold")
    axes[0].invert_yaxis()
    figure.suptitle("07  /  MOTION, ATTENTION AND THE COST OF MISSING A FALL", x=0.03, ha="left", size=17, weight="bold", color="#162c34")
    figure.text(0.03, 0.045, "Source: pooled_by_seed.csv · fixed threshold 0.5 · mean ± sample SD over seeds 42 / 142 / 242", size=10, color="#4e676e")
    figure.text(0.03, 0.015, "Each seed evaluates the same 160 videos by outer subject. SD is not a confidence interval; no 480-video claim.", size=9, color="#4e676e")
    figure.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(figure, "ablation.png")

    figure, axis = plt.subplots(figsize=(11.5, 7))
    offsets = [(8, 10), (8, -25), (8, -23), (-100, 13), (10, 8)]
    for index, variant in enumerate(VARIANTS):
        sensitivity = statistics.loc[variant, ("sensitivity", "mean")]
        false_alarm = 100 - statistics.loc[variant, ("specificity", "mean")]
        axis.errorbar(false_alarm, sensitivity, xerr=statistics.loc[variant, ("specificity", "std")], yerr=statistics.loc[variant, ("sensitivity", "std")], fmt="o", markersize=10, color=COLORS[index], capsize=4, alpha=0.9)
        axis.annotate(LABELS[index], (false_alarm, sensitivity), xytext=offsets[index], textcoords="offset points", weight="bold", fontsize=11, color=COLORS[index])
    axis.set_xlim(0, 28)
    axis.set_ylim(70, 100)
    axis.set_xlabel("False alarms on ADL (%)  →  lower is better", labelpad=12)
    axis.set_ylabel("Sensitivity on Fall (%)  →  higher is better", labelpad=12)
    axis.set_title("Same accuracy does not mean the same errors", loc="left", size=19, weight="bold", color="#162c34", pad=23)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.11, 0.035, "Fixed threshold 0.5 · 3-seed mean ± sample SD · A3 cost: 3 / 3 / 2 missed falls, not zero.", size=10, color="#4e676e")
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    save(figure, "tradeoff.png")


if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": "#f5f8f7", "text.color": "#162c34", "axes.labelcolor": "#4e676e", "xtick.color": "#4e676e", "ytick.color": "#162c34"})
    result_figures()
    print(f"Rendered 2 result figures in {TARGET}")
