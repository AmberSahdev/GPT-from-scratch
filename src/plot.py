from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    with np.load("data/one_episode.npz", allow_pickle=False) as data:
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

    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/one_episode.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
