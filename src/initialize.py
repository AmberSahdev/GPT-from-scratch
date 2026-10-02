"""Save the randomly initialized GPT before either training stage."""

from dataclasses import asdict
from pathlib import Path

import torch

from .model import GPT, ModelConfig

CHECKPOINT_PATH = Path("runs/experiment-01/untrained.pt")
SEED = 42


def main():
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(f"Already exists: {CHECKPOINT_PATH}")
    torch.manual_seed(SEED)
    config = ModelConfig()
    model = GPT(config)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model": model.state_dict()}, CHECKPOINT_PATH)
    print("Saved", CHECKPOINT_PATH)
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    main()
