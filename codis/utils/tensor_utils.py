"""Short utilities to deal with tensors."""
import numpy as np
import torch


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array."""
    return x.detach().cpu().numpy()
