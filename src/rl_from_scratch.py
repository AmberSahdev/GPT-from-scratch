"""Train a randomly initialized GPT with Reinforcement Learning - PPO"""

from pathlib import Path

from . import rl_finetuning as ppo

INITIAL_CHECKPOINT = Path("runs/experiment-01/untrained.pt")
TOTAL_TIMESTEPS = ppo.TOTAL_TIMESTEPS
OUTPUT_DIR = INITIAL_CHECKPOINT.parent / f"ppo-scratch-{TOTAL_TIMESTEPS}-steps"
LEARNING_RATE = 0.0003
UPDATE_EPOCHS = 10
CLIP_RANGE = 0.2
TARGET_KL = 0.02
ENTROPY_COEFFICIENT = 0.01
REWARD_SCALE = 0.01
SEED = 42


def main():
    ppo.INITIAL_CHECKPOINT = INITIAL_CHECKPOINT
    ppo.OUTPUT_DIR = OUTPUT_DIR
    ppo.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
    ppo.LEARNING_RATE = LEARNING_RATE
    ppo.UPDATE_EPOCHS = UPDATE_EPOCHS
    ppo.CLIP_RANGE = CLIP_RANGE
    ppo.TARGET_KL = TARGET_KL
    ppo.ENTROPY_COEFFICIENT = ENTROPY_COEFFICIENT
    ppo.REWARD_SCALE = REWARD_SCALE
    ppo.SEED = SEED
    ppo.main()


if __name__ == "__main__":
    main()
