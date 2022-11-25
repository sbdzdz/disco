"""Plotting utilities."""
import numpy as np
from matplotlib import pyplot as plt


def show_images_grid(imgs_, Nmax=25):
    num_images = min(imgs_.shape[0], Nmax)
    imgs_ = imgs_.squeeze(1) if imgs_.ndim == 4 else imgs_
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 2, ncols * 2))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap="Greys_r", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    plt.show()


def show_density(imgs):
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation="nearest", cmap="Greys_r")
    ax.grid("off")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
