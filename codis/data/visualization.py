import io
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from codis.data.infinite_dsprites import (
    InfiniteDSprites,
    InfiniteDSpritesTriplets,
    Latents,
)


def draw_shapes(nrows=5, ncols=12, fig_height=10):
    """Plot an n x n grid of random shapes.
    Args:
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
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

    with imageio.get_writer("shapes.gif", mode="I") as writer:
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


def generate_latent_progression(
    scale_range, orientation_range, position_x_range, position_y_range
):
    """Generate a sequence of latents that can be used to animate a shape.
    Args:
        scale_range: The range of scales to use.
        orientation_range: The range of orientations to use.
        position_x_range: The range of x positions to use.
        position_y_range: The range of y positions to use.
    Returns:
        A tuple of (scales, orientations, positions_x, positions_y) that can be used to animate a shape.
    """
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


def draw_triplet(fig_height=10):
    """Plot a triplet of shapes form the InfiniteDSpritesTriplets.
    See Montero et al. 2020 for details of the composition task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSpritesTriplets(image_size=256)
    (images, action) = next(iter(dataset))
    _, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(3 * fig_height, fig_height),
        subplot_kw={"aspect": 1.0},
        layout="tight",
    )
    for ax, img in zip(axes.flat, images):
        ax.axis("off")
        ax.imshow(img)
    plt.savefig(f"triplet_{action}.png", bbox_inches="tight")
    plt.close()


def draw_same_different_task(fig_height=10):
    dataset = InfiniteDSprites(image_size=256)
    latents_reference = dataset.sample_latents()
    latents_same = dataset.sample_latents()
    latents_different = dataset.sample_latents()

    reference = dataset.draw(latents_reference)
    for latent in ["shape", "scale", "orientation", "position_x", "position_y"]:
        same = dataset.draw(
            latents_same._replace(**{latent: latents_reference[latent]})
        )
        different = dataset.draw(latents_different)
        pairs = [(reference, same), (reference, different)]
        for pair, label in zip(pairs, ["same", "different"]):
            _, axes = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(2 * fig_height, fig_height),
                subplot_kw={"aspect": 1.0},
                layout="tight",
            )
            for ax, img in zip(axes.flat, pair):
                ax.axis("off")
                ax.imshow(img)
            plt.savefig(
                f"task_classification_{latent}_{label}.png",
                bbox_inches="tight",
                pad_inches=0,
            )


def draw_single_shape(path="shape.png", latents: Latents = None):
    """Plot a single random shape with given latents applied and save it to disk.
    Args:
        path: The path to save the image to.
        latents: The latents to apply to the shape.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=512)
    if latents is None:
        latents = dataset.sample_latents()
    image = dataset.draw(latents)
    plt.imshow(image, aspect=1.0, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close("all")


if __name__ == "__main__":
    draw_triplet()
