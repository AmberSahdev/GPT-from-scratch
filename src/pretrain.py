import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .model import GPT, ModelConfig

# Settings for supervised training
INITIAL_CHECKPOINT = Path("runs/experiment-01/untrained.pt")
DATA_PATH = Path("data/demonstrations.npz")
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TRAIN_FRACTION = 0.8
SEED = 42


def get_device():
    # MPS is PyTorch's GPU backend for supported Macs; CUDA is for NVIDIA GPUs.
    # is_available checks that the backend is usable, not just compiled into PyTorch.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Evaluate teacher-action agreement on data without changing model weights
def measure(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    with torch.no_grad():
        for states, actions in loader:
            # Move inputs and labels to the same device as the model.
            states, actions = states.to(device), actions.to(device)

            logits = model(states) # Forward pass

            # Cross entropy loss
            total_loss += float(F.cross_entropy(logits, actions, reduction="sum"))

            # Count action accuracy
            correct += int((logits.argmax(-1) == actions).sum())

            # Count examples
            count += len(actions)

    return total_loss / count, correct / count


def main():
    if EPOCHS < 1 or BATCH_SIZE < 1:
        raise ValueError("Use positive EPOCHS and BATCH_SIZE")
    if not 0 < TRAIN_FRACTION < 1:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1")
    output_dir = INITIAL_CHECKPOINT.parent
    device = get_device()
    print("Training device:", device)

    torch.manual_seed(SEED)

    checkpoint = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    
    with np.load(DATA_PATH, allow_pickle=False) as data:
        number_of_episodes = len(data["episode_offsets"]) - 1
    
    if number_of_episodes < 2:
        raise ValueError("Need at least two episodes for train/validation")
    
    # shuffle episode IDs reproducibly.
    shuffled = np.random.default_rng(SEED).permutation(number_of_episodes)
    
    # 80/20 training/validation split
    split = max(1, min(number_of_episodes - 1, int(TRAIN_FRACTION * number_of_episodes)))
    train_ids, validation_ids = shuffled[:split], shuffled[split:]
    train = TrajectoryDataset(DATA_PATH, train_ids, config.context_length)
    validation = TrajectoryDataset(DATA_PATH, validation_ids, config.context_length)

    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=BATCH_SIZE)
    
    # AdamW optimizer
    # lr controls step size, decay penalizes large weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    np.savez(output_dir / "split.npz", train_ids=train_ids, validation_ids=validation_ids)
    
    best_loss = float("inf")
    # Measure the full training loop, including validation and checkpoint saves.
    # perf_counter measures elapsed wall-clock time with a monotonic clock.
    training_started = perf_counter()
    with (output_dir / "pretrain.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation_loss", "validation_accuracy"])
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, count = 0.0, 0
            
            for states, actions in train_loader:
                states, actions = states.to(device), actions.to(device)
                
                logits = model(states)
                
                loss = F.cross_entropy(logits, actions)
                
                optimizer.zero_grad(set_to_none=True)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM) # gradient clipping
                
                # Update learned weights using AdamW and the gradients just computed.
                optimizer.step()
                total_loss += loss.item() * len(actions)
                count += len(actions)
            
            train_loss = total_loss / count
            validation_loss, accuracy = measure(model, validation_loader, device)
            writer.writerow([epoch, train_loss, validation_loss, accuracy])
            
            file.flush() # Write buffered log rows

            print(epoch, "train:", round(train_loss, 4), "validation:",
                  round(validation_loss, 4), "accuracy:", round(accuracy, 3))
            
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save({"config": checkpoint["config"], "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "epoch": epoch,
                            "validation_loss": validation_loss}, output_dir / "pretrained.pt")

    elapsed_seconds = perf_counter() - training_started
    print(f"Training completed in {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60:.2f} minutes).")

if __name__ == "__main__":
    main()
