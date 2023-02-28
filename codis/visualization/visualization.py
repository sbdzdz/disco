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

from codis.data import (
    InfiniteDSprites,
    InfiniteDSpritesTriplets,
    InfiniteDSpritesAnalogies,
    Latents,
)

repo_root = Path(__file__).parent.parent.parent


def draw_batch_grid(
    images,
    path: Path = repo_root / "img/batch_grid.png",
    fig_height: float = 10,
    n_max: int = 16,
    show=False,
):
    """Show a batch of images on a grid.
    Only the first n_max images are shown.
    Args:
        images: A tensor of shape (N, C, H, W) or (N, H, W).
        n_max: The maximum number of images to show.
    Returns:
        None
    """
    num_images = min(images.shape[0], n_max)
    if images.ndim == 4:
        images = images.squeeze(1)
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))
    fig, axes = plt.subplots(
        ncols, nrows, figsize=(ncols / nrows * fig_height, fig_height)
    )

    for ax, img in zip(axes.flat, images[:num_images]):
        ax.imshow(img, cmap="Greys_r", interpolation="nearest")
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def draw_batch_and_reconstructions(
    x,
    x_hat,
    path: Path = repo_root / "img/reconstructions.png",
    fig_height: float = 10,
    n_max: int = 16,
    show=False,
):
    """Show a batch of images and their reconstructions on a grid.
    Only the first n_max images are shown.
    Args:
        x: A tensor of shape (N, C, H, W) or (N, H, W).
        x_hat: A tensor of shape (N, C, H, W) or (N, H, W).
        n_max: The maximum number of images to show.
    Returns:
        None
    """
    num_images = min(x.shape[0], n_max)
    if x.ndim == 4:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols / nrows * fig_height, fig_height),
    )
    fig.tight_layout()
    for ax, img, img_hat in zip(axes.flat, x[:num_images], x_hat[:num_images]):
        concatenated = np.concatenate([img, img_hat], axis=1)
        ax.imshow(concatenated, cmap="Greys_r", interpolation="nearest")
        ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def draw_batch_density(
    images,
    path: Path = repo_root / "img/images_density.png",
    fig_height: float = 10,
    show=False,
):
    """Show a batch of images averaged over the batch dimension.
    Args:
        imgs: A tensor of shape (N, C, H, W) or (N, H, W).
    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(fig_height, fig_height))
    ax.imshow(images.mean(axis=0), interpolation="nearest", cmap="Greys_r")
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()


def draw_shape(
    path: Path = repo_root / "img/shape.png", latents: Optional[Latents] = None
):
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
    path: Path = repo_root / "img/shapes.png",
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


def draw_shape_animated(
    path: Path = repo_root / "img/shape.gif", fig_height: float = 10
):
    """Create an animated GIF showing a shape undergoing transformations.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=512)
    shape = dataset.generate_shape()
    scales, orientations, positions_x, positions_y = generate_latent_progression(
        dataset
    )
    color = np.random.choice(dataset.ranges["color"])
    frames = [
        dataset.draw(Latents(color, shape, scale, orientation, position_x, position_y))
        for scale, orientation, position_x, position_y in zip(
            scales, orientations, positions_x, positions_y
        )
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I") as writer:
        for frame in tqdm(frames):
            _, ax = plt.subplots(figsize=(fig_height, fig_height))
            ax.axis("off")
            ax.imshow(frame)
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
            plt.close()
            image = imageio.imread(buffer)
            writer.append_data(image)  # type: ignore


def draw_shapes_animated(
    path: Path = repo_root / "img/shapes.gif",
    nrows: int = 5,
    ncols: int = 11,
    fig_height: float = 10,
):
    """Create an animated GIF showing a grid of shapes undergoing transformations.
    Args:
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(image_size=256)
    shapes = [dataset.generate_shape() for _ in range(nrows * ncols)]
    colors = np.random.choice(dataset.ranges["color"], size=nrows * ncols)
    scales, orientations, positions_x, positions_y = generate_latent_progression(
        dataset
    )

    frames = [
        [
            dataset.draw(
                Latents(color, shape, scale, orientation, position_x, position_y)
            )
            for shape, color in zip(shapes, colors)
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
            buffer = io.BytesIO()
            for ax, image in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image)
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
        A tuple of latent value sequences representing a smooth animation.
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


def draw_triplet(path: Path = repo_root / "img/triplet.png", fig_height: float = 10):
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


def draw_classification_task(
    path: Path = repo_root / "img/classification.png", fig_height: float = 10
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


def draw_analogy_task(
    path: Path = repo_root / "img/analogy.png", fig_height: float = 10
):
    """Draw an example of the analogy task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSpritesAnalogies(image_size=512)
    image = next(iter(dataset))
    _, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(fig_height, fig_height),
        subplot_kw={"aspect": 1.0},
        layout="tight",
    )
    axes.axis("off")
    axes.imshow(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)


def draw_hard_analogy_task(
    path: Path = repo_root / "img/hard_analogy.png", fig_height: float = 10
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
    latents_query_source = dataset.sample_latents()
    latents_query_target = latents_query_source

    for latent in ["scale", "orientation", "position_x", "position_y"]:
        delta = latents_reference_target[latent] - latents_reference_source[latent]
        latent_range = dataset.ranges[latent]
        range_min, range_max = latent_range.min(), latent_range.max()
        new_value = latents_query_source[latent] + delta
        new_value = range_min + (new_value - range_min) % (range_max - range_min)
        latents_query_target = latents_query_target._replace(**{latent: new_value})

    # draw the images on a single grid
    images = [
        dataset.draw(latents_reference_source),
        dataset.draw(latents_reference_target),
        dataset.draw(latents_query_source),
        dataset.draw(latents_query_target),
    ]
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
