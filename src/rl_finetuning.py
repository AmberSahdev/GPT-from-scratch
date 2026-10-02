import json
from pathlib import Path
from time import perf_counter

import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .environment import make_environment
from .model import GPT, ModelConfig

# Edit these settings before running python -m src.rl_finetuning.
INITIAL_CHECKPOINT = Path("runs/experiment-01/pretrained.pt")
TOTAL_TIMESTEPS = 640000
LEARNING_RATE = 1e-5
ROLLOUT_STEPS = 2000  # 320 complete rollouts give exactly 640,000 simulator steps.
BATCH_SIZE = 125  # 16 minibatches per rollout, with no partial batch.
UPDATE_EPOCHS = 2
CLIP_RANGE = 0.05
TARGET_KL = 0.005
ENTROPY_COEFFICIENT = 0.0  # Action sampling still explores without an extra entropy bonus.
SEED = 42
EVALUATION_SEED = 20000
EVALUATION_FREQUENCY = 5000
EVALUATION_EPISODES = 10
OUTPUT_DIR = INITIAL_CHECKPOINT.parent / "rl"


def get_device():
    # Match pretraining: use the Mac GPU, then NVIDIA GPU, then CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# A thin SB3 adapter around our GPT class
class GPTFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, config):
        config = ModelConfig(**config)
        super().__init__(observation_space, features_dim=config.embedding_dim)
        self.gpt = GPT(config)
        
        # SB3 supplies its own action/value heads; avoid a duplicate unused action layer.
        self.gpt.action_head = torch.nn.Identity() # returns input unchanged

    def forward(self, observations):
        # get_features gives (B, T, embedding_dim).
        # SB3 expects one (B, embedding_dim) vector per history.
        return self.gpt.get_features(observations)[:, -1, :]


class RepeatableEvalCallback(EvalCallback):
    """Compare each checkpoint on the same evaluation sequence, including step zero."""

    def _on_training_start(self):
        # Evaluate the transferred pretrained policy before any PPO updates.
        # n_calls is zero, so EvalCallback's normal evaluation condition is satisfied.
        self._on_step()

    def _on_training_end(self):
        # Include the fully updated final policy in checkpoint selection and the graph.
        self.n_calls = 0
        self._on_step()

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # VecEnv applies this seed on the reset performed by evaluate_policy.
            self.eval_env.seed(EVALUATION_SEED)
        return super()._on_step()


class SeededEvaluationMonitor(Monitor):
    """Give every evaluation flight its own seed, independent of previous actions."""

    def __init__(self, env):
        super().__init__(env)
        self.next_seed = EVALUATION_SEED

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:
            kwargs["seed"] = self.next_seed
        self.next_seed = kwargs["seed"] + 1
        return super().reset(**kwargs)


def make_rl_environment(context_length, evaluation=False):
    history_env = FrameStackObservation(
        make_environment(), stack_size=context_length, padding_type="reset"
    )
    return SeededEvaluationMonitor(history_env) if evaluation else Monitor(history_env)


def main():
    if TOTAL_TIMESTEPS < 1:
        raise ValueError("TOTAL_TIMESTEPS must be positive")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Choose a new OUTPUT_DIR; {OUTPUT_DIR} already contains a run")

    device = get_device()
    print("Training device:", device)
    print("PyTorch CPU threads:", torch.get_num_threads())
    torch.manual_seed(SEED)

    checkpoint = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    pretrained = GPT(config).to(device)
    pretrained.load_state_dict(checkpoint["model"])
    pretrained.eval()
    
    training_env = make_rl_environment(config.context_length)
    evaluation_env = make_rl_environment(config.context_length, evaluation=True)
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = PPO(
            "MlpPolicy", training_env, device=device, seed=SEED, verbose=1,
            # Collect fresh transitions, then update in minibatches for a small number of passes.
            learning_rate=LEARNING_RATE, n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE, n_epochs=UPDATE_EPOCHS,
            # gamma discounts future rewards. gae_lambda controls advantage estimation. Entropy bonus encourages exploration. target_kl can stop overly large policy updates early.
            gamma=0.99, gae_lambda=0.95, ent_coef=ENTROPY_COEFFICIENT,
            clip_range=CLIP_RANGE, target_kl=TARGET_KL,
            policy_kwargs={
                "features_extractor_class": GPTFeaturesExtractor,
                "features_extractor_kwargs": {"config": checkpoint["config"]},
                "net_arch": [],
                # Disable SB3's optional initialization; we explicitly restore pretrained weights next.
                "ortho_init": False,
            },
        )
        
        # copy the pretrained model’s saved tensors, excluding its action head.
        feature_weights = {
            name: tensor for name, tensor in pretrained.state_dict().items()
            if not name.startswith("action_head.")
        }

        # strict loading (the default) catches missing or mismatched feature-layer weights
        agent.policy.features_extractor.gpt.load_state_dict(feature_weights)

        # Transfer the teacher-trained classifier into SB3's own four-action layer
        agent.policy.action_net.load_state_dict(pretrained.action_head.state_dict())

        # GPT, the action head, and the value head all learn together in PPO.
        print("Trainable parameters:", sum(p.numel() for p in agent.policy.parameters() if p.requires_grad))
        agent.set_logger(configure(str(output_dir), ["stdout", "csv"]))

        # Before learning, prove the action distribution survived the handoff.
        ### 
        agent.policy.set_training_mode(False)
        probe = torch.randn(5, config.context_length, 8, device=device)
        
        with torch.no_grad():
            expected = pretrained(probe).softmax(-1)
            actual = agent.policy.get_distribution(probe).distribution.probs

        assert torch.allclose(expected, actual, atol=1e-5), "Policy transfer changed probabilities"

        print("Verified pretrained action probabilities survived transfer")
        agent.save(output_dir / "before_rl")
        ###

        callback = RepeatableEvalCallback(
            evaluation_env, best_model_save_path=str(output_dir),
            log_path=str(output_dir), eval_freq=EVALUATION_FREQUENCY,
            n_eval_episodes=EVALUATION_EPISODES, deterministic=True,
        )
        
        started = perf_counter()
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        agent.save(output_dir / "final")
        metadata = {
            "pretrained_epoch": checkpoint["epoch"],
            "total_timesteps": agent.num_timesteps,
            "rollout_steps": ROLLOUT_STEPS, "batch_size": BATCH_SIZE,
            "update_epochs": UPDATE_EPOCHS, "learning_rate": LEARNING_RATE,
            "freeze_gpt": False, "clip_range": CLIP_RANGE,
            "target_kl": TARGET_KL, "entropy_coefficient": ENTROPY_COEFFICIENT,
            "evaluation_seed_protocol": "seed_each_episode",
            "elapsed_seconds": perf_counter() - started,
            "best_evaluation_reward": callback.best_mean_reward,
            "best_evaluation_step": callback.evaluations_timesteps[
                max(range(len(callback.evaluations_results)),
                    key=lambda i: sum(callback.evaluations_results[i]) / len(callback.evaluations_results[i]))
            ],
        }
        (output_dir / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"RL training completed in {metadata['elapsed_seconds']:.1f} seconds.")
    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
