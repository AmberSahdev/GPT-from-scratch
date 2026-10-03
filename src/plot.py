"""Codex generated plotting code. Saved training and evaluation results."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, GRAY = "#4775D1", "#DE8964", "#9CA6B5"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": "#24324B", "axes.labelcolor": "#657189",
    "xtick.color": "#657189", "ytick.color": "#657189",
    "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 9, "ytick.major.pad": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#E2E7EF", "axes.titleweight": "bold",
    "axes.titlelocation": "center", "axes.labelpad": 12,
    "axes.titlesize": 14, "axes.titlepad": 12,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#EDF0F5", "grid.linewidth": 0.8,
    "lines.linewidth": 2.2, "lines.markersize": 4.5,
    "lines.markeredgecolor": "auto", "lines.markeredgewidth": 0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
}
RL_STYLES = [
    ("Pretrained + RL", "Pretrained + RL", BLUE, "-", "o"),
    ("RL without pretraining", "RL without pretraining", ORANGE, ":", "^"),
]

STEP_FORMAT = FuncFormatter(lambda value, _: f"{value:g}k" if value else "0")


def finish_chart(fig, title, filename):
    """Keep both README images the same size and typography."""
    fig.text(0.5, 0.97, title, fontsize=20, fontweight="bold", va="top", ha="center")
    fig.subplots_adjust(left=0.065, right=0.965, top=0.77, bottom=0.13, wspace=0.30)
    Path("assets").mkdir(exist_ok=True)
    fig.savefig(Path("assets") / filename, dpi=180, facecolor="white")
    plt.close(fig)


@plt.rc_context(STYLE)
def plot_pretraining(path="runs/experiment-01/pretrain.csv"):
    with open(path) as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    
    for field, label, color, style in [
        ("train_loss", "Train", BLUE, "-"),
        ("validation_loss", "Validation", ORANGE, "--"),
    ]:
        values = [float(row[field]) for row in rows]
        axes[0].plot(epochs, values, label=label, color=color, linestyle=style, marker="o")
    axes[0].set(title="Prediction loss", ylabel="Cross-entropy", ylim=(0, None))
    axes[0].legend(loc="upper right")

    accuracy = [float(row["validation_accuracy"]) for row in rows]

    axes[1].plot(epochs, accuracy, color=BLUE, marker="o")
    axes[1].set(title="Teacher-action agreement", ylabel="Validation agreement", ylim=(0, 1.02))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].annotate(f"{accuracy[-1]:.1%}", (epochs[-1], accuracy[-1]),
                     xytext=(8, 0), textcoords="offset points", va="center", color=BLUE)
    
    for ax in axes:
        ax.set(xlabel="Epoch", xlim=(0.7, epochs[-1] + 1.2), xticks=epochs)
    finish_chart(fig, "Pretraining loss and teacher-action agreement",
                 "pretraining.png")


@plt.rc_context(STYLE)
def plot_rl(path="assets/rl_progress.json"):
    evidence = json.loads(Path(path).read_text())
    results = evidence["results"]
    baseline = json.loads(Path("assets/model_comparison.json").read_text())["results"]["Pretrained"]
    panels = [("reward_at_least_200_rate", 100, "Success rate", "Flights scoring ≥200"),
              ("mean_reward", 1, "Average flight score", "Original simulator reward")]
    final_step = max(row["rl_steps"] for rows in results.values() for row in rows) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    
    for ax, (field, factor, title, ylabel) in zip(axes, panels):
        for source, label, color, style, marker in RL_STYLES:
            rows = results[source]
            steps = np.asarray([row["rl_steps"] for row in rows]) / 1000
            values = np.asarray([row[field] for row in rows]) * factor
            ax.plot(steps, values, label=label, color=color, linestyle=style, marker=marker)
            offset = 0 if factor == 100 else (10 if source == "Pretrained + RL" else -10)
            ax.annotate(f"{values[-1]:.0f}%" if factor == 100 else f"{values[-1]:.1f}",
                        (steps[-1], values[-1]), xytext=(9, offset), textcoords="offset points",
                        va="center", fontsize=11, fontweight="bold", color=color)
        ax.axhline(baseline[field] * factor, label="Pretrained baseline", color=GRAY,
                   linewidth=1, linestyle=(0, (5, 4)), zorder=0)
        ax.set(title=title, xlabel="RL simulator steps", ylabel=ylabel,
               xlim=(-4, final_step + 30), xticks=np.arange(0, final_step + 1, 32))
        ax.xaxis.set_major_formatter(STEP_FORMAT)
        
        if factor == 100:
            ax.set(ylim=(-5, 105), yticks=[0, 20, 40, 60, 80, 100])
            ax.yaxis.set_major_formatter(PercentFormatter(100))
        else:
            ax.axhline(0, color=GRAY, linewidth=0.7, zorder=0)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, handlelength=2.6, columnspacing=2.5)
    
    finish_chart(fig, "Flight performance during RL training",
                 "rl_rewards.png")


if __name__ == "__main__":
    plot_pretraining()
    plot_rl()
