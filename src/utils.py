"""Small helpers shared across training scripts."""

import torch


def get_device():
    """Use the Mac GPU, then NVIDIA GPU, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
