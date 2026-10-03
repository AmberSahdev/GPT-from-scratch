"""Codex generated code to render gifs for five matching flights from the saved models."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .environment import make_stacked_environment, run_episode
from .evaluate import load_policy

COMPARISON_PATH = Path("assets/model_comparison.json")
ASSET_PATH = Path("assets")
FIRST_SEED = 10000
EPISODES = 5
FRAME_STRIDE = 4
FRAME_SIZE = (480, 320)
STAGES = {
    "untrained": "Untrained",
    "pretrained": "Pretrained",
    "rl": "Pretrained + RL",
    "rl_scratch": "RL without pretraining",
}


def record_flight(choose, context_length, seed):
    env = make_stacked_environment(context_length, "rgb_array")
    frames = []

    def capture_frame(env, steps, done):
        if steps % FRAME_STRIDE == 0 or done:
            frames.append(Image.fromarray(env.render()).resize(FRAME_SIZE))

    try:
        result = run_episode(env, choose, seed, on_step=capture_frame)
    finally:
        env.close()

    return frames, result


def make_sequence(title, flights, destination):
    width, height = FRAME_SIZE
    font = ImageFont.load_default(size=16)
    frames, durations = [], []

    for flight_index, (images, result) in enumerate(flights, start=1):
        for index, frame in enumerate(images):
            canvas = Image.new("RGB", (width, height + 30), "#101827")
            canvas.paste(frame, (0, 30))
            label = f"{title} | {flight_index}/{len(flights)} | seed {result['seed']}"
            final = index == len(images) - 1
            if final:
                label += f" | score {result['reward']:.0f}"
            ImageDraw.Draw(canvas).text((8, 6),
                                        label,
                                        font=font,
                                        fill="#e2e8f0")
            frames.append(canvas)
            # Four simulator steps at 50 Hz take 80 ms; hold each result for a second.
            durations.append(1000 if final else 80)
    frames[0].save(destination,
                   save_all=True,
                   append_images=frames[1:],
                   duration=durations,
                   loop=0,
                   optimize=True,
                   disposal=2)


def render_stage(stage, comparison):
    model_name = STAGES[stage]
    source = comparison[model_name]
    checkpoint = Path(source["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    if digest != source["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint differs from the saved comparison: {checkpoint}")
    predict, context_length, training_steps = load_policy(checkpoint)

    if training_steps != source["rl_training_steps"]:
        raise ValueError(
            f"Training budget differs from the saved comparison: {checkpoint}")

    def choose(history):
        return int(predict(history[None])[0])

    flights = []

    for seed in range(FIRST_SEED, FIRST_SEED + EPISODES):
        flight = record_flight(choose, context_length, seed)
        flights.append(flight)
        print(stage, seed, "score:", round(flight[1]["reward"], 1), flush=True)

    destination = ASSET_PATH / f"{stage}_comparison.gif"
    make_sequence(model_name, flights, destination)
    rewards = [result["reward"] for _, result in flights]

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "rl_training_steps": int(training_steps),
        "gif": str(destination),
        "action_selection": "deterministic argmax",
        "mean_reward": sum(rewards) / len(rewards),
        "episodes": [result for _, result in flights],
    }


def main():
    torch.set_num_threads(1)
    ASSET_PATH.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(COMPARISON_PATH.read_text())["results"]
    summary = {stage: render_stage(stage, comparison) for stage in STAGES}
    (ASSET_PATH / "showcase_scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved four GIFs and their five-flight scores; no training or benchmark updates.")


if __name__ == "__main__":
    main()
