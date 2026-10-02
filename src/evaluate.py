# measures a saved model without training it
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from PIL import Image
from stable_baselines3 import PPO

from .environment import make_environment
from .model import GPT, ModelConfig

# Edit these settings before running python -m src.evaluate.
STAGE = "untrained"  # Choose "untrained", "pretrained", or "rl".
CHECKPOINT_PATH = Path("runs/experiment-01/untrained.pt")
# For RL: Path("runs/experiment-01/rl/best_model.zip") or final.zip.
EPISODES = 20
FIRST_SEED = 10000  # Keep identical across model stages for a fair comparison.
GIF_PATH = None  # Example: Path("assets/untrained.gif"); None skips rendering.


def make_stacked_environment(context_length, render_mode=None):
    """
    Regular environment:  (8,)    — one state, eight measurements
    Wrapped environment:  (8, 8)  — eight states, eight measurements each
    """
    return FrameStackObservation(
        make_environment(render_mode), stack_size=context_length, padding_type="reset"
    )


def evaluate(action_function, context_length, episodes, first_seed, gif_path=None):
    results = []
    env = make_stacked_environment(context_length, "rgb_array" if gif_path else None)

    try:
        for index in range(episodes):
            # Use a reproducible seed for each evaluation episode; all stages use the same list.
            seed = first_seed + index

            history, _ = env.reset(seed=seed)

            # Collect RGB images only when making a GIF, not while measuring every episode.
            frames = []

            # Reset this episode's accumulated score and length; never carry them across resets.
            total_reward, steps = 0.0, 0

            # Keep advancing this episode until an ending flag tells us to stop.
            while True:
                # Capture only the first episode and every third frame.
                if gif_path and index == 0 and steps % 3 == 0:
                    # rgb_array rendering -> Pillow frame
                    frames.append(Image.fromarray(env.render()))
                
                # Call the supplied policy with the current history
                action = action_function(history)

                # The wrapper shifts history and appends the new state after applying the command.
                history, reward, terminated, truncated, _ = env.step(action)

                # Add this step's reward to the episode score: total_reward = total_reward + reward.
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            raw = env.unwrapped
            
            # Physical rest without body crash; can include landings off the pad.
            # Count termination at rest without a body-crash flag; this can include an off-pad landing.
            safe_rest = bool(terminated and not raw.game_over and not raw.lander.awake)
            
            results.append({"seed": seed, "reward": total_reward, "steps": steps,
                            "safe_rest": safe_rest, "truncated": bool(truncated)})
            
            if gif_path and index == 0 and frames:
                frames.append(Image.fromarray(env.render()))
                gif_path.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                               duration=60, loop=0)
    finally:
        env.close()

    rewards = np.asarray([row["reward"] for row in results])

    return {
        "mean_reward": float(rewards.mean()),
        # Episode-to-episode score spread, not uncertainty across independent training runs.
        "std_reward": float(rewards.std()),
        "safe_rest_rate": float(np.mean([row["safe_rest"] for row in results])),
        # Average boolean values to measure the fraction clearing the score threshold.
        "reward_at_least_200_rate": float(np.mean(rewards >= 200)),
        "episodes": results,
    }


def main():
    if EPISODES < 1:
        raise ValueError("EPISODES must be positive")
    if STAGE not in ("untrained", "pretrained", "rl"):
        raise ValueError("Use STAGE = 'untrained', 'pretrained', or 'rl'")

    if STAGE == "rl":
        # SB3 checkpoints are ZIP files containing the GPT, action head, and value head.
        # Evaluation uses CPU so it also works without the training GPU.
        agent = PPO.load(CHECKPOINT_PATH, device="cpu")
        context_length = agent.observation_space.shape[0]

        def choose(history):
            # Select the most likely action; predict performs inference without learning.
            action, _ = agent.predict(history, deterministic=True)
            return int(np.asarray(action).item())
    else:
        # Untrained/pretrained stages use our original PyTorch checkpoint format.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint["config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        context_length = config.context_length

        def choose(history):
            with torch.no_grad():
                # Add the batch axis: (T,8) -> (1,T,8).
                states = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)
                return int(model(states).argmax(dim=-1).item())
    
    results = evaluate(choose, context_length, EPISODES, FIRST_SEED, GIF_PATH)

    results["checkpoint"] = str(CHECKPOINT_PATH)
    results["action_selection"] = "deterministic argmax"
    output = CHECKPOINT_PATH.parent / f"{STAGE}_evaluation.json"

    # Save readable JSON scores so README numbers can be traced to actual episodes.
    output.write_text(json.dumps(results, indent=2) + "\n")

    print("Mean reward:", results["mean_reward"], "+/-", results["std_reward"])
    print("Safe rest rate:", results["safe_rest_rate"])
    print("Reward >= 200 rate:", results["reward_at_least_200_rate"])
    print("Saved", output)

if __name__ == "__main__":
    main()
