"""
I couldn't find a training dataset online, so we are generating it.
I think that's what Gymnasium intends anyway.
"""

import argparse
from pathlib import Path

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from .environment import make_environment



def collect_episode(env, seed):
    # returns a dictionary mapping observation + action to new state
    observation, _ = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    terminations, truncations = [], []

    while True:
        # Ask the teacher which engine to use; convert its result to a plain integer ID.
        action = int(heuristic(env, observation))

        observations.append(observation.copy())
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        terminations.append(terminated)
        truncations.append(truncated)

        if terminated or truncated:
            break
    return {
        # float32 stores states as 32-bit floats, matching the neural network's usual input type.
        # int64 stores class IDs in the integer type PyTorch cross-entropy expects.
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminations, dtype=bool),
        "truncated": np.asarray(truncations, dtype=bool),
        "final_observation": np.asarray(
            observation, dtype=np.float32
        ),  # Keep the last next-state too; it has no corresponding teacher action in this episode.
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/one_episode.npz")
    )  # zip archive of named numpy arrays
    args = parser.parse_args()

    if args.episodes < 1 or args.output.suffix != ".npz":
        parser.error("Use a positive episode count and a .npz output path")

    if args.output.exists():
        parser.error("Choose a new output path; this one already exists")

    episodes = []
    env = make_environment()
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            episode = collect_episode(env, seed)
            episodes.append(episode)
            print(
                seed,
                "steps:",
                len(episode["actions"]),
                "reward:",
                float(episode["rewards"].sum()),
            )
    finally:
        env.close()

    # Combine episode arrays into one flat dataset
    arrays = {
        key: np.concatenate([episode[key] for episode in episodes])
        for key in ("observations", "actions", "rewards", "terminated", "truncated")
    }
    arrays["episode_offsets"] = np.concatenate(
        ([0], np.cumsum([len(ep["actions"]) for ep in episodes]))
    )

    arrays["episode_seeds"] = np.arange(args.seed, args.seed + args.episodes)
    arrays["final_observations"] = np.stack(
        [ep["final_observation"] for ep in episodes]
    )
    arrays["environment"] = np.asarray("LunarLander-v3")
    arrays["teacher"] = np.asarray(
        "Gymnasium heuristic; wind disabled; discrete actions"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    print("Saved", len(arrays["actions"]), "examples to", args.output)


def print_data():
    # helper function for me to visualize data
    with np.load("data/one_episode.npz") as data:
        n = len(data["actions"])

        for step in [0, 1, n - 2, n - 1]:
            if step == n - 2:
                print("\n...")

            print(f"\nStep {step}")
            print("  State: ", data["observations"][step])
            print("  Action:", data["actions"][step])
            print("  Reward:", data["rewards"][step])


# Run the program when invoked with python -m src.<module>, not when imported.
if __name__ == "__main__":
    main()
    # print_data()
