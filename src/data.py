"""
I couldn't find a training dataset online, so we are generating it.
I think that's what Gymnasium intends anyway.
"""

from pathlib import Path

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic
from torch.utils.data import Dataset

from .environment import CONTINUOUS_ACTIONS, ENABLE_WIND, ENV_ID, make_environment


# Edit these settings before running python -m src.data.
EPISODES = 300
FIRST_SEED = 0
OUTPUT_PATH = Path("data/demonstrations.npz")
INSPECTION_PATH = Path("data/one_episode.npz")


class TrajectoryDataset(Dataset):
    """
    Expose generated training data through PyTorch's Dataset interface.
    Dataset is PyTorch's interface for retrieving one example by index. 
    A DataLoader later stacks examples into batches.
    """
    def __init__(self, path, episode_ids, context_length):
        self.context_length = context_length
        with np.load(path, allow_pickle=False) as data:
            self.observations = data["observations"].copy()
            self.actions = data["actions"].copy()
            offsets = data["episode_offsets"].copy()

        self.indices = []
        for episode_id in episode_ids:
            start, end = map(int, offsets[episode_id : episode_id + 2])
            self.indices.extend((start, index) for index in range(start, end))

    # DataLoader calls this to learn how many training examples are available.
    def __len__(self):
        return len(self.indices)

    # DataLoader calls this to get one history window and its action label.
    def __getitem__(self, index):
        episode_start, current = self.indices[index]

        # Never reach before this episode's reset or further back than the context budget.
        start = max(episode_start, current - self.context_length + 1)

        # Exclude every future state to avoid leaking the answer.
        window = self.observations[start : current + 1]

        # Early timesteps have too little history; calculate how much padding they need.
        missing = self.context_length - len(window)

        if missing:
            # Repeat s0 at the left, matching the live frame-stack wrapper's reset padding.
            padding = np.repeat(
                self.observations[episode_start : episode_start + 1], missing, axis=0
            )
            window = np.concatenate((padding, window))

        return window.copy(), self.actions[current]


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
    if EPISODES < 1 or OUTPUT_PATH.suffix != ".npz":
        raise ValueError("Use positive EPISODES and a .npz OUTPUT_PATH")

    if OUTPUT_PATH.exists():
        raise FileExistsError("Choose a new OUTPUT_PATH; this dataset already exists")

    episodes = []
    env = make_environment()
    try:
        for seed in range(FIRST_SEED, FIRST_SEED + EPISODES):
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

    arrays["episode_seeds"] = np.arange(FIRST_SEED, FIRST_SEED + EPISODES)
    arrays["final_observations"] = np.stack(
        [ep["final_observation"] for ep in episodes]
    )
    arrays["environment"] = np.asarray(ENV_ID)
    arrays["teacher"] = np.asarray(
        f"Gymnasium heuristic; wind={ENABLE_WIND}; continuous={CONTINUOUS_ACTIONS}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **arrays)

    print("Saved", len(arrays["actions"]), "examples to", OUTPUT_PATH)


def print_data():
    # helper function for me to visualize data
    with np.load(INSPECTION_PATH, allow_pickle=False) as data:
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
