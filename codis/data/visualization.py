import io

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Iterable
import numpy as np

from codis.data.infinite_dsprites import InfiniteDSprites, Latents


def draw_shapes_on_grid(nrows=5, ncols=12, fig_height=10):
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


def animate_shapes_on_grid(
    nrows=5,
    ncols=12,
    fig_height=10,
    scale_range: Iterable = np.linspace(0.5, 1, 6),
    orientation_range: Iterable = np.linspace(0, 2 * np.pi, 40),
    position_x_range: Iterable = np.linspace(0, 1, 32),
    position_y_range: Iterable = np.linspace(0, 1, 32),
):
    """Create an animated GIF showing an animated grid of nrows x ncols shapes.
    Args:
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=128)
    shapes = [dataset.generate_shape() for _ in range(nrows * ncols)]
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


if __name__ == "__main__":
    animate_shapes_on_grid()
