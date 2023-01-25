import io

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Iterable
import numpy as np
from itertools import islice

from codis.data.infinite_dsprites import (
    InfiniteDSprites,
    InfiniteDSpritesTriplets,
    Latents,
)


def draw_shapes(nrows=5, ncols=12, fig_height=10):
    """Plot an n x n grid of random shapes.
    Args:
        n: The number of rows and columns in the grid.
    Returns:
        None
    """
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols / nrows * fig_height, fig_height),
        layout="tight",
        subplot_kw={"aspect": 1.0},
    )
    dataset = InfiniteDSprites()
    for ax in axes.flat:
        spline = dataset.generate_shape()
        ax.axis("off")
        ax.plot(spline[0], spline[1], label="spline", color="red")
    plt.savefig("shapes.png")


def draw_shapes_animated(
    nrows=5,
    ncols=12,
    fig_height=10,
    scale_range: Iterable = np.linspace(0.5, 1, 6),
    orientation_range: Iterable = np.linspace(0, 2 * np.pi, 40),
    position_x_range: Iterable = np.linspace(0, 1, 32),
    position_y_range: Iterable = np.linspace(0, 1, 32),
):
    """Create an animated GIF showing an nrows x ncols grid of shapes undergoing transformations.
    Args:
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=256)
    shapes = [dataset.generate_shape() for _ in range(nrows * ncols)]
    scales, orientations, positions_x, positions_y = generate_latent_progression(
        scale_range,
        orientation_range,
        position_x_range,
        position_y_range,
    )

    color = 0
    frames = [
        [
            dataset.draw(
                Latents(color, shape, scale, orientation, position_x, position_y)
            )
            for shape in shapes
        ]
        for scale, orientation, position_x, position_y in zip(
            scales, orientations, positions_x, positions_y
        )
    ]
    plot_on_grid(nrows, ncols, fig_height, frames)


def generate_latent_progression(
    scale_range, orientation_range, position_x_range, position_y_range
):
    length = (
        len(scale_range)
        + len(orientation_range)
        + len(position_x_range)
        + len(position_y_range)
    )
    scales, orientations, positions_x, positions_y = (
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
    )

    start = 0
    scales[start : len(scale_range)] = scale_range
    scales[len(scale_range) :] = scale_range[-1]

    start = len(scale_range)
    orientations[start : start + len(orientation_range)] = orientation_range
    orientations[start + len(orientation_range) :] = orientation_range[-1]

    start = len(scale_range) + len(orientation_range)
    positions_x[start : start + len(position_x_range)] = position_x_range
    positions_x[start + len(position_x_range) :] = position_x_range[-1]

    start = len(scale_range) + len(orientation_range) + len(position_x_range)
    positions_y[start : start + len(position_y_range)] = position_y_range
    positions_y[start + len(position_y_range) :] = position_y_range[-1]
    return scales, orientations, positions_x, positions_y


def plot_on_grid(nrows, ncols, fig_height, frames):
    with imageio.get_writer("zoom_out.gif", mode="I") as writer:
        for frame in tqdm(frames):
            _, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols / nrows * fig_height, fig_height),
                layout="tight",
                subplot_kw={"aspect": 1.0},
            )
            for ax, image in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image)
                buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            image = imageio.imread(buffer)
            writer.append_data(image)


def draw_single_shape(
    path="shape.png",
    orientation=0,
    scale=1,
    position_x=0.5,
    position_y=0.5,
):
    """Plot a single random shape with given latents applied and save it to disk.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=256)
    shape = dataset.generate_shape()
    latents = Latents(
        color=0,
        shape=shape,
        scale=scale,
        orientation=orientation,
        position_x=position_x,
        position_y=position_y,
    )
    image = dataset.draw(latents)
    plt.imshow(image, aspect=1.0, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)


def draw_triplets():
    dataset = InfiniteDSpritesTriplets()
    for (image_original, image_transform, image_reference), action in islice(
        dataset, 10
    ):
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(image_original, aspect=1.0, cmap="gray")
        plt.axis("off")
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(image_transform, aspect=1.0, cmap="gray")
        plt.axis("off")
        plt.title("Transformed")
        plt.subplot(1, 3, 3)
        plt.imshow(image_reference, aspect=1.0, cmap="gray")
        plt.axis("off")
        plt.title("Reference")
        plt.suptitle(f"Action: {action}")
        plt.savefig(f"triplet_{action}.png", bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close()


if __name__ == "__main__":
    draw_triplets()
