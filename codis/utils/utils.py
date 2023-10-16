"""Short utilities to deal with tensors."""
from itertools import zip_longest

import numpy as np
import torch


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array."""
    return x.detach().cpu().numpy()


def grouper(iterable, n):
    """Iterate in groups of n elements, e.g. grouper(3, 'ABCDEF') --> ABC DEF.
    Args:
        n: The number of elements per group.
        iterable: The iterable to be grouped.
    Returns:
        An iterator over the groups.
    """
    args = [iter(iterable)] * n
    return (list(group) for group in zip_longest(*args))
