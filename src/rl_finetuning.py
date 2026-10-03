"""Fine-tune the GPT with PPO and save checkpoints for evaluation."""

import json
from pathlib import Path
from time import perf_counter

import torch
from gymnasium.wrappers import TransformReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .environment import make_stacked_environment
from .model import GPT, ModelConfig
from .utils import get_device

INITIAL_CHECKPOINT = Path("runs/experiment-01/pretrained.pt")
TOTAL_TIMESTEPS = 2**18  # 262,144 simulator steps = 128 complete rollouts
LEARNING_RATE = 1e-5
ROLLOUT_STEPS = 2**11  # 2,048 fresh simulator steps before each PPO update cycle
BATCH_SIZE = 128  # 16 minibatches per rollout
UPDATE_EPOCHS = 2
CLIP_RANGE = 0.05
TARGET_KL = 0.005
ENTROPY_COEFFICIENT = 0.0  # Action sampling still explores without an entropy bonus
REWARD_SCALE = 0.01  # Scale value targets; evaluation keeps original game rewards
CPU_THREADS = 1  # Faster for small matrix operations on CPU
SEED = 42
CHECKPOINT_INTERVAL = 4 * ROLLOUT_STEPS  # Save every 8,192 steps
OUTPUT_DIR = INITIAL_CHECKPOINT.parent / f"ppo-pretrained-{TOTAL_TIMESTEPS}-steps"


class GPTFeaturesExtractor(BaseFeaturesExtractor):
    """Expose the newest GPT features to SB3's action and value heads."""

    def __init__(self, observation_space, config):
        config = ModelConfig(**config)
        super().__init__(observation_space, features_dim=config.embedding_dim)
        self.gpt = GPT(config)

        self.gpt.action_head = torch.nn.Identity() # pass through

    def forward(self, observations):
        return self.gpt.get_features(observations)[:, -1, :]


class CheckpointCallback(BaseCallback):
    """Save initial weights and completed PPO updates for the learning curves."""

    def __init__(self, output_dir, interval):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.checkpoints = []

    def _on_training_start(self):
        self._save_checkpoint()

    def _on_rollout_start(self):
        # This hook runs after the previous rollout's optimizer updates.
        if self.model.num_timesteps % self.interval == 0:
            self._save_checkpoint()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, final=False):
        steps = self.model.num_timesteps
        if self.checkpoints and self.checkpoints[-1]["rl_steps"] == steps:
            return
        filename = "final.zip" if final else f"checkpoints/step_{steps:06d}.zip"
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.checkpoints.append({
            "rl_steps": steps,
            "checkpoint": str(path),
            "learning_rate": float(self.model.policy.optimizer.param_groups[0]["lr"]),
        })
        (self.output_dir / "learning_progress.json").write_text(
            json.dumps({"checkpoints": self.checkpoints}, indent=2) + "\n")


def make_rl_environment(context_length, reward_scale=1.0):
    history_env = make_stacked_environment(context_length)

    # Record original scores before scaling the rewards used for learning.
    monitored = Monitor(history_env)

    if reward_scale != 1.0:
        return TransformReward(monitored, lambda reward: reward * reward_scale)

    return monitored


def transfer_pretrained_policy(agent, pretrained):
    """Restore the GPT and action head, leaving the new value head untouched."""
    feature_weights = {
        name: tensor
        for name, tensor in pretrained.state_dict().items()
        if not name.startswith("action_head.")
    }

    agent.policy.features_extractor.gpt.load_state_dict(feature_weights)
    agent.policy.action_net.load_state_dict(pretrained.action_head.state_dict())


def main():
    if TOTAL_TIMESTEPS < 1:
        raise ValueError("TOTAL_TIMESTEPS must be positive")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Choose a new OUTPUT_DIR; {OUTPUT_DIR} already contains a run")

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(CPU_THREADS)

    print("Training device:", device)
    print("PyTorch CPU threads:", torch.get_num_threads())
    torch.manual_seed(SEED)

    # Load the saved GPT and its action head.
    checkpoint = torch.load(INITIAL_CHECKPOINT,
                            map_location="cpu",
                            weights_only=True)

    config = ModelConfig(**checkpoint["config"])

    pretrained = GPT(config).to(device)
    pretrained.load_state_dict(checkpoint["model"])
    pretrained.eval()

    training_env = make_rl_environment(config.context_length,
                                       reward_scale=REWARD_SCALE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # SB3 adds action/value heads to the shared GPT and implements PPO.
        agent = PPO(
            "MlpPolicy",
            training_env,
            device=device,
            seed=SEED,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=UPDATE_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=ENTROPY_COEFFICIENT,
            clip_range=CLIP_RANGE,
            target_kl=TARGET_KL,
            policy_kwargs={
                "features_extractor_class": GPTFeaturesExtractor,
                "features_extractor_kwargs": {
                    "config": checkpoint["config"]
                },
                "net_arch": [],
                "share_features_extractor": True,
                "ortho_init": False,  # Restore the pretrained weights below.
            },
        )

        transfer_pretrained_policy(agent, pretrained)

        print(
            "Trainable parameters:",
            sum(p.numel() for p in agent.policy.parameters()
                if p.requires_grad))

        agent.set_logger(configure(str(OUTPUT_DIR), ["stdout", "csv"]))

        # Verify the handoff. Keep this probe before training to preserve the RNG sequence.
        agent.policy.set_training_mode(False)
        probe = torch.randn(5, config.context_length, 8, device=device)

        with torch.no_grad():
            expected = pretrained(probe).softmax(-1)
            actual = agent.policy.get_distribution(probe).distribution.probs

        assert torch.allclose(
            expected, actual,
            atol=1e-5), "Policy transfer changed probabilities"

        print("Verified saved action probabilities survived transfer")
        callback = CheckpointCallback(OUTPUT_DIR, CHECKPOINT_INTERVAL)

        started = perf_counter()
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        # Save the settings needed to interpret this run later.
        metadata = {
            "initialization": "pretrained" if "epoch" in checkpoint else "scratch",
            "initial_checkpoint": str(INITIAL_CHECKPOINT),
            "pretrained_epoch": checkpoint.get("epoch"),
            "total_timesteps": agent.num_timesteps,
            "rollout_steps": ROLLOUT_STEPS,
            "batch_size": BATCH_SIZE,
            "update_epochs": UPDATE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "clip_range": CLIP_RANGE,
            "target_kl": TARGET_KL,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "reward_scale": REWARD_SCALE,
            "seed": SEED,
            "elapsed_seconds": perf_counter() - started,
        }

        (OUTPUT_DIR / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"RL training completed in {metadata['elapsed_seconds']:.1f} seconds.")
    finally:
        training_env.close()


if __name__ == "__main__":
    main()
