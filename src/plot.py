import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("data/one_episode.npz")
FIGURE_PATH = Path("assets/one_episode.png")


def main():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]

    print("States:", observations.shape, "Actions:", actions.shape)
    print("First state:", observations[0], "First action:", actions[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(observations[:, 0], label="x")
    axes[0].plot(observations[:, 1], label="y")
    axes[0].legend()
    axes[0].set_ylabel("Scaled position")

    # Actions are discrete choices, so a step plot is clearer than interpolated sloping lines.
    axes[1].step(np.arange(len(actions)), actions, where="post")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_ylabel("Action")

    # cumsum shows the running episode score, not just isolated step rewards.
    axes[2].plot(np.cumsum(rewards))
    axes[2].set_ylabel("Cumulative reward")
    axes[2].set_xlabel("Simulator step")

    # Adjust spacing so axis labels do not overlap.
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def plot_pretraining(path="runs/experiment-01/pretrain.csv"):
    with open(path) as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(
        epochs, [float(row["validation_loss"]) for row in rows], label="validation"
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, [float(row["validation_accuracy"]) for row in rows])
    axes[1].set(xlabel="Epoch", ylabel="Teacher-action agreement", ylim=(0, 1))

    fig.tight_layout()

    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/pretraining.png", dpi=150)
    plt.close(fig)


def plot_rl(path="runs/experiment-01/rl/evaluations.npz"):
    with np.load(path, allow_pickle=False) as data:
        steps = data["timesteps"]
        scores = data["results"]

    mean, std = scores.mean(axis=1), scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean, label="Mean evaluation reward")

    ax.fill_between(
        steps, mean - std, mean + std, alpha=0.2, label="One episode standard deviation"
    )
    ax.axhline(200, color="gray", linestyle="--", label="200-point reference")
    ax.set(xlabel="Environment steps", ylabel="Episode reward")
    ax.legend()

    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/rl_rewards.png", dpi=150)
    plt.close(fig)


def plot_training(run_path="runs/experiment-01", rl_path=None):
    """Connect the stages visually while preserving their different metrics and clocks."""
    run_path = Path(run_path)
    with (run_path / "pretrain.csv").open() as file:
        rows = list(csv.DictReader(file))
    rl_path = Path(rl_path) if rl_path is not None else run_path / "rl"
    with np.load(rl_path / "evaluations.npz", allow_pickle=False) as data:
        steps, scores = data["timesteps"], data["results"]

    epochs = [int(row["epoch"]) for row in rows]
    mean, std = scores.mean(axis=1), scores.std(axis=1)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        fig.suptitle("Pretraining → reinforcement learning", fontsize=18, fontweight="bold", y=0.98)
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows],
                     color="#2563eb", linewidth=2, label="Training")
        axes[0].plot(epochs, [float(row["validation_loss"]) for row in rows],
                     color="#0891b2", linewidth=2, linestyle="--", label="Validation")
        # RL starts from the best validation checkpoint, not necessarily the final epoch.
        import torch
        checkpoint = torch.load(run_path / "pretrained.pt", map_location="cpu", weights_only=True)
        selected_epoch = checkpoint["epoch"]
        axes[0].axvline(selected_epoch, color="#64748b", linestyle=":",
                       label=f"Best saved checkpoint: epoch {selected_epoch}")
        axes[0].set(title="1 · Learn the heuristic's actions", xlabel="Pretraining epoch",
                    ylabel="Cross-entropy loss ↓")
        axes[0].legend(frameon=False, fontsize=9)

        axes[1].plot(steps / 1000, mean, color="#7c3aed", linewidth=2,
                     marker="o", markersize=4, label="Mean evaluation reward")
        selected = int(np.argmax(mean))
        axes[1].scatter(steps[selected] / 1000, mean[selected], s=90,
                        facecolors="none", edgecolors="#d97706", linewidths=2,
                        zorder=4, label="Checkpoint used in GIFs")
        axes[1].fill_between(steps / 1000, mean - std, mean + std,
                             color="#7c3aed", alpha=0.12, label="±1 episode standard deviation")
        if steps[0] == 0:
            # Show whether PPO improves on the policy it actually started from.
            axes[1].axhline(mean[0], color="#2563eb", linestyle=":", linewidth=1.5,
                           label=f"Before RL: {mean[0]:.1f}")
        axes[1].axhline(200, color="#64748b", linestyle="--", linewidth=1,
                       label="200-point reference")
        axes[1].set(title="2 · Optimize flight rewards with PPO", xlabel="RL environment steps (thousands)",
                    ylabel="Episode reward ↑")
        axes[1].legend(frameon=False, fontsize=9)
        for ax in axes:
            ax.grid(axis="y", alpha=0.15)
        fig.text(0.5, 0.02, f"Different objectives: losses and rewards are not directly comparable. RL: {scores.shape[1]} flights per evaluation.",
                 ha="center", color="#475569", fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        Path("assets").mkdir(exist_ok=True)
        fig.savefig("assets/training.png", dpi=160, facecolor="white")
        plt.close(fig)


def plot_accuracy(run_path="runs/experiment-01", rl_path=None):
    """Keep action-label accuracy separate from success in actual simulator flights."""
    run_path = Path(run_path)
    accuracy = json.loads(Path("assets/action_accuracy.json").read_text())
    stages = ["untrained", "pretrained", "rl"]
    labels = ["Untrained", "Pretrained", "Pretrained + RL"]
    rl_path = Path(rl_path) if rl_path is not None else run_path / "rl"
    evaluations = [json.loads(((rl_path if stage == "rl" else run_path) /
                              f"{stage}_evaluation.json").read_text()) for stage in stages]
    values = [[100 * accuracy[stage]["accuracy"] for stage in stages],
              [100 * evaluation["reward_at_least_200_rate"] for evaluation in evaluations]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        titles = ["Teacher-action accuracy", "Flight success: reward ≥ 200"]
        for ax, percentages, title, color in zip(axes, values, titles, ["#0891b2", "#7c3aed"]):
            ax.plot(range(3), percentages, color=color, marker="o", linewidth=2, markersize=7)
            for index, percentage in enumerate(percentages):
                ax.annotate(f"{percentage:.1f}%", (index, percentage),
                            xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11)
            ax.set(title=title, ylabel="Percent", ylim=(-4, 112), xticks=range(3), xticklabels=labels)
            ax.axvline(1.5, color="#64748b", linestyle="--", linewidth=1)
            ax.text(1.52, 4, "RL fine-tuning starts", rotation=90,
                    fontsize=9, color="#475569", va="bottom")
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", alpha=0.15)
            ax.margins(x=0.15)
        fig.text(0.5, 0.02, "Action accuracy: the same 60 validation flights. Flight success: the same 100 new simulation seeds.",
                 ha="center", fontsize=10, color="#475569")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig("assets/accuracy.png", dpi=160, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
