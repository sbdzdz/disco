"""Plotting utilities."""
import numpy as np
from matplotlib import pyplot as plt


def show_images_grid(imgs, n_max=25):
    """Show a grid of images.
    Only the first n_max images are shown.
    Args:
        imgs: A tensor of shape (N, C, H, W) or (N, H, W).
        n_max: The maximum number of images to show.
    Returns:
            None
    """
    num_images = min(imgs.shape[0], n_max)
    imgs = imgs.squeeze(1) if imgs.ndim == 4 else imgs
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 2, ncols * 2))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs[ax_i], cmap="Greys_r", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    plt.show()


def show_density(imgs):
    """Show the density of a set of images averaged over the batch dimension.
    Args:
        imgs: A tensor of shape (N, C, H, W) or (N, H, W).
    Returns:
            None
    """
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation="nearest", cmap="Greys_r")
    ax.grid("off")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
