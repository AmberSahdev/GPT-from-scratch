from pathlib import Path

import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .environment import make_environment
from .model import GPT, ModelConfig

# Edit these settings before running python -m src.rl_finetuning.
INITIAL_CHECKPOINT = Path("runs/experiment-01/pretrained.pt")
TOTAL_TIMESTEPS = 100000
LEARNING_RATE = 1e-4
ROLLOUT_STEPS = 1024
BATCH_SIZE = 64
UPDATE_EPOCHS = 4
SEED = 42
EVALUATION_SEED = 20000
EVALUATION_FREQUENCY = 10000
EVALUATION_EPISODES = 10


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
        self.gpt.action_head = torch.nn.Identity()

    def forward(self, observations):
        # get_features gives (B, T, embedding_dim).
        # SB3 expects one (B, embedding_dim) vector per history.
        return self.gpt.get_features(observations)[:, -1, :]


def make_rl_environment(context_length):
    history_env = FrameStackObservation(
        make_environment(), stack_size=context_length, padding_type="reset"
    )
    return Monitor(history_env)


def main():
    if TOTAL_TIMESTEPS < 1:
        raise ValueError("TOTAL_TIMESTEPS must be positive")

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
    evaluation_env = make_rl_environment(config.context_length)
    
    output_dir = INITIAL_CHECKPOINT.parent / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = PPO(
            "MlpPolicy", training_env, device=device, seed=SEED, verbose=1,
            # Collect 1024 transitions per rollout, update in batches of 64, and reuse that rollout for 4 passes.
            learning_rate=LEARNING_RATE, n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE, n_epochs=UPDATE_EPOCHS,
            # gamma discounts future rewards. gae_lambda controls advantage estimation. Entropy bonus encourages exploration. target_kl can stop overly large policy updates early.
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01, target_kl=0.02,
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

        evaluation_env.reset(seed=EVALUATION_SEED)
        callback = EvalCallback(
            evaluation_env, best_model_save_path=str(output_dir),
            log_path=str(output_dir), eval_freq=EVALUATION_FREQUENCY,
            n_eval_episodes=EVALUATION_EPISODES, deterministic=True,
        )
        
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        agent.save(output_dir / "final")
    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
