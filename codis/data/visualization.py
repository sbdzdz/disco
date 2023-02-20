"""Visualization utilities for the dSprites and InfiniteDSprites datasets.
Example usage:
    python -c "from codis.data.visualization import draw_shape; draw_shape()"
"""
import io
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from codis.data.infinite_dsprites import (
    InfiniteDSprites,
    InfiniteDSpritesTriplets,
    Latents,
)


def draw_shape(path: Path = Path("img/shape.png"), latents: Optional[Latents] = None):
    """Draw a single shape from given or randomly sampled latents and save it to disk.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)


def draw_shapes(
    path: Path = Path("img/shapes.png"),
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)


def draw_shapes_animated(
    path: Path = Path("img/shapes.gif"),
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
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
        dataset
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

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I") as writer:
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
            writer.append_data(image)  # type: ignore


def generate_latent_progression(dataset):
    """Generate a sequence of latents that can be used to animate a shape.
    Args:
        scale_range: The range of scales to use.
        orientation_range: The range of orientations to use.
        position_x_range: The range of x positions to use.
        position_y_range: The range of y positions to use.
    Returns:
        A tuple of (scales, orientations, positions_x, positions_y) representing a smooth animation.
    """
    scale_range, orientation_range, position_x_range, position_y_range = (
        dataset.ranges["scale"],
        dataset.ranges["orientation"],
        dataset.ranges["position_x"],
        dataset.ranges["position_y"],
    )
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


def draw_triplet(path: Path = Path("img/triplet.png"), fig_height: float = 10):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_name(f"{path.stem}_{action}{path.suffix}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def draw_classification_tak(
    path: Path = Path("img/classification.png"), fig_height: float = 10
):
    """Draw an example of the binary classification task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
            path = path.with_name(f"{path.stem}_{latent}_{label}{path.suffix}")
            plt.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0,
            )


def draw_analogy_task(path: Path = Path("img/analogy.png"), fig_height: float = 10):
    """Draw an example of the analogy task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=256)
    latents_reference_source = dataset.sample_latents()
    latents_reference_target = dataset.sample_latents()._replace(
        shape=latents_reference_source.shape
    )

    query_shape = dataset.generate_shape()
    latents_query_source = latents_reference_source._replace(shape=query_shape)
    latents_query_target = latents_reference_target._replace(shape=query_shape)

    # Draw the images on a single grid with reference source and target in the top row and query source and target in the bottom row.
    reference_source = dataset.draw(latents_reference_source)
    reference_target = dataset.draw(latents_reference_target)
    query_source = dataset.draw(latents_query_source)
    query_target = dataset.draw(latents_query_target)
    images = [reference_source, reference_target, query_source, query_target]
    _, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(fig_height, fig_height),
        subplot_kw={"aspect": 1.0},
        layout="tight",
    )
    for ax, img in zip(axes.flat, images):
        ax.axis("off")
        ax.imshow(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def draw_hard_analogy_task(
    path: Path = Path("img/hard_analogy.png"), fig_height: float = 10
):
    """Draw an example of the hard analogy task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=256)
    latents_reference_source = dataset.sample_latents()
    latents_reference_target = dataset.sample_latents()._replace(
        shape=latents_reference_source.shape
    )
    # get the differenc of the x position with a periodic boundary condition
    position_x_delta = latents_reference_target.position_x - latents_reference_source.position_x
    position_y_delta = latents_reference_target.position_y - latents_reference_source.position_y
    orientation_delta = latents_reference_target.orientation - latents_reference_source.orientation
    scale_delta = latents_reference_target.scale - latents_reference_source.scale

    latents_query_source = dataset.sample_latents()
    position_x = (latents_query_source.position_x + position_x_delta) % dataset.ranges["position_x"].max()
    position_y = (latents_query_source.position_y + position_y_delta) % dataset.ranges["position_y"].max()
    orientation = (latents_query_source.orientation + orientation_delta) % dataset.ranges["orientation"].max()
    scale = (latents_query_source.scale + scale_delta) % dataset.ranges["scale"].max()
    latents_query_target = latents_query_source._replace(
        position_x=position_x,
        position_y=position_y,
        orientation=orientation,
        scale=scale,
    )

    # Draw the images on a single grid with reference source and target in the top row and query source and target in the bottom row.
    reference_source = dataset.draw(latents_reference_source)
    reference_target = dataset.draw(latents_reference_target)
    query_source = dataset.draw(latents_query_source)
    query_target = dataset.draw(latents_query_target)
    images = [reference_source, reference_target, query_source, query_target]
    _, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(fig_height, fig_height),
        subplot_kw={"aspect": 1.0},
        layout="tight",
    )
    for ax, img in zip(axes.flat, images):
        ax.axis("off")
        ax.imshow(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
