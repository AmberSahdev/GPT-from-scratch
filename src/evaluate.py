# measures a saved model without training it
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .environment import (
    CONTINUOUS_ACTIONS,
    ENABLE_WIND,
    ENV_ID,
    make_stacked_environment,
    run_episode,
)
from .model import GPT, ModelConfig

EPISODES = 100
FIRST_SEED = 70000  # for identical evaluation flights for every model and training budget
OUTPUT_PATH = Path("assets/model_comparison.json")
EVALUATE_PROGRESS = True  # Also measure the saved checkpoints along both learning curves
PROGRESS_OUTPUT_PATH = Path("assets/rl_progress.json")
DATA_PATH = Path("data/demonstrations.npz")
SPLIT_PATH = Path("runs/experiment-01/split.npz")

CHECKPOINTS = {
    "Untrained": "runs/experiment-01/untrained.pt",
    "Pretrained": "runs/experiment-01/pretrained.pt",
    "RL without pretraining": "runs/experiment-01/ppo-scratch-192k-checkpoints-20261002/final.zip",
    "Pretrained + RL": "runs/experiment-01/ppo-pretrained-192k-checkpoints-20261002/final.zip",
}
PROGRESS_RUNS = {
    name: Path(path).parent for name, path in CHECKPOINTS.items() if Path(path).suffix == ".zip"
}


def evaluate(action_function, context_length, episodes, first_seed):
    results = []
    env = make_stacked_environment(context_length)

    try:
        for index in range(episodes):
            # Use a reproducible seed for each evaluation episode; all stages use the same list.
            seed = first_seed + index

            results.append(run_episode(env, action_function, seed))
    finally:
        env.close()

    rewards = np.asarray([row["reward"] for row in results])

    return {
        "mean_reward": float(rewards.mean()),
        # Average boolean values to measure the fraction clearing the score threshold.
        "reward_at_least_200_rate": float(np.mean(rewards >= 200)),
        "episodes": results,
    }


def load_policy(path):
    """Load either a plain GPT (.pt) or an SB3 PPO policy (.zip), for inference only."""
    path = Path(path)
    if path.suffix == ".zip":
        agent = PPO.load(path, device="cpu")
        agent.policy.set_training_mode(False)

        def predict(states):
            return agent.predict(np.asarray(states), deterministic=True)[0]

        return predict, agent.observation_space.shape[0], agent.num_timesteps

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def predict(states):
        with torch.inference_mode():
            states = torch.as_tensor(states, dtype=torch.float32)
            return model(states).argmax(-1).numpy()

    return predict, config.context_length, 0


def evaluate_checkpoint(path, episodes=EPISODES, first_seed=FIRST_SEED):
    """Measure real flights and teacher agreement separately, using fixed test data."""
    path = Path(path)

    predict, context_length, training_steps = load_policy(path)
    results = evaluate(lambda history: int(predict(history[None])[0]),
                       context_length, episodes, first_seed)
    
    with np.load(SPLIT_PATH, allow_pickle=False) as split:
        validation_ids = split["validation_ids"]
    dataset = TrajectoryDataset(DATA_PATH, validation_ids, context_length)
    correct = 0
    
    for states, actions in DataLoader(dataset, batch_size=256):
        correct += int(np.sum(predict(states.numpy()) == actions.numpy()))
    
    results.update(
        checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rl_training_steps=int(training_steps), action_accuracy=correct / len(dataset),
        correct_actions=correct, validation_examples=len(dataset),
        validation_episodes=len(validation_ids),
    )
    
    training_path = path.parent / "training.json"
    
    if path.suffix == ".zip":
        results["training"] = json.loads(training_path.read_text())
    
    return results


def comparison_table(results):
    lines = ["| Model | Success rate | Average score | Teacher-action agreement |",
             "| :--- | ---: | ---: | ---: |"]
    
    for name, row in results.items():
        label = "RL" if name == "RL without pretraining" else name
        lines.append(f"| {label} | "
                     f"{100 * row['reward_at_least_200_rate']:.0f}% | {row['mean_reward']:.1f} | "
                     f"{100 * row['action_accuracy']:.2f}% |")
    
    return "\n".join(lines)


def progress_protocol(episodes, first_seed):
    return {
        "flight_episodes": episodes, "flight_first_seed": first_seed,
        "flight_last_seed": first_seed + episodes - 1,
        "action_selection": "deterministic argmax", "reward_units": "original game rewards",
        "environment": ENV_ID, "continuous_actions": CONTINUOUS_ACTIONS, "enable_wind": ENABLE_WIND,
        "success": "Fraction of episodes with reward at least 200.",
    }


def evaluate_progress(run_directory, episodes=EPISODES, first_seed=FIRST_SEED):
    """Measure checkpoint flights, saving each result so interrupted runs can resume."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    
    torch.set_num_threads(1)
    
    run_directory = Path(run_directory)
    learning = json.loads((run_directory / "learning_progress.json").read_text())
    output = run_directory / "test_progress.json"
    protocol = progress_protocol(episodes, first_seed)

    cached = json.loads(output.read_text()) if output.exists() else {}
    cached_rows = {row["rl_steps"]: row for row in cached.get("checkpoints", [])}
    
    if cached.get("protocol") != protocol:
        cached_rows = {}
    rows = []
    
    for recorded in learning["checkpoints"]:
        path = Path(recorded["checkpoint"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = cached_rows.get(recorded["rl_steps"])

        if row is not None and row.get("checkpoint_sha256") == digest:
            row = {**row, "checkpoint": str(path)}
            print("Reusing test flights:", path, flush=True)
        else:
            print("Evaluating test flights:", path, flush=True)
            predict, context_length, actual_steps = load_policy(path)
            if actual_steps != recorded["rl_steps"]:
                raise ValueError(f"Checkpoint step count does not match learning_progress.json: {path}")
            measured = evaluate(lambda history: int(predict(history[None])[0]),
                                context_length, episodes, first_seed)
            row = {"rl_steps": actual_steps, "checkpoint": str(path),
                   "checkpoint_sha256": digest, **measured}
        expected_seeds = list(range(first_seed, first_seed + episodes))

        if [episode["seed"] for episode in row["episodes"]] != expected_seeds:
            raise ValueError(f"Checkpoint evaluation has a different flight cohort: {path}")
        
        row["learning_rate"] = recorded["learning_rate"]
        rows.append(row)
        
        output.write_text(json.dumps({"protocol": protocol, "checkpoints": rows}, indent=2) + "\n")
    
    return {"protocol": protocol, "checkpoints": rows}


def save_comparison(results):
    # Compare every model on the same flights.
    reference_seeds = [row["seed"] for row in results["Pretrained"]["episodes"]]
    for row in results.values():
        if [episode["seed"] for episode in row["episodes"]] != reference_seeds:
            raise ValueError("Comparisons require identical ordered flight seeds")
        
    evidence = {
        "protocol": {
            **progress_protocol(EPISODES, FIRST_SEED),
            "accuracy": "Teacher-action agreement on held-out demonstrations; stored as action_accuracy.",
            "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            "checkpoint_selection": "Final checkpoint of each named run, not a selected peak.",
            "comparison": "Practical configurations with settings recorded per row; one training run per row.",
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2) + "\n")
    print(comparison_table(results))
    print("Saved", OUTPUT_PATH)


def main():
    torch.set_num_threads(1)

    if EPISODES < 1:
        raise ValueError("EPISODES must be positive")
    results = {}
    
    for name, path in CHECKPOINTS.items():
        print("Evaluating", name, flush=True)
        results[name] = evaluate_checkpoint(path, EPISODES, FIRST_SEED)
    save_comparison(results)
    
    if EVALUATE_PROGRESS:
        curves = {
            name: evaluate_progress(directory, EPISODES, FIRST_SEED)["checkpoints"]
            for name, directory in PROGRESS_RUNS.items()
        }
        evidence = {"protocol": progress_protocol(EPISODES, FIRST_SEED), "results": curves}
        PROGRESS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_OUTPUT_PATH.write_text(json.dumps(evidence, indent=2) + "\n")
        print("Saved", PROGRESS_OUTPUT_PATH)


if __name__ == "__main__":
    main()
