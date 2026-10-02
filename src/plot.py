import csv
from pathlib import Path

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


if __name__ == "__main__":
    main()
