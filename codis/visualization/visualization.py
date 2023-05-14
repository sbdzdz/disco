"""Visualization utilities for the dSprites and InfiniteDSprites datasets.
Example usage:
    python -c "from codis.data.visualization import draw_shapes; draw_shapes()"
"""
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import PIL
from matplotlib import pyplot as plt
from tqdm import tqdm

from codis.data import (
    InfiniteDSprites,
    InfiniteDSpritesAnalogies,
    InfiniteDSpritesTriplets,
    Latents,
)

np.random.seed(0)
repo_root = Path(__file__).parent.parent.parent

COLORS = [
    "purple",
    "maroon",
    "darkblue",
    "teal",
    "peachpuff",
    "white",
    "darkgreen",
]


def draw_batch(
    images,
    path: Path = repo_root / "img/batch_grid.png",
    fig_height: float = 10,
    n_max: int = 16,
    show=False,
):
    """Show a batch of images on a grid.
    Only the first n_max images are shown.
    Args:
        images: A numpy array of shape (N, C, H, W) or (N, H, W).
        n_max: The maximum number of images to show.
    Returns:
        None
    """
    num_images = min(images.shape[0], n_max)
    if images.ndim == 4:
        images = images.permute(0, 2, 3, 1)
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(
        ncols, nrows, figsize=(ncols / nrows * fig_height, fig_height)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, img in zip(axes.flat, images[:num_images]):
        ax.imshow(img, cmap="Greys_r", interpolation="nearest")
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def draw_batch_and_reconstructions(
    x,
    x_hat,
    fig_height: float = 10,
    n_max: int = 16,
    path: Path = None,
    show=False,
):
    """Show a batch of images and their reconstructions on a grid.
    Only the first n_max images are shown.
    Args:
        x: A numpy array of shape (N, C, H, W) or (N, H, W).
        x_hat: A numpy array of shape (N, C, H, W) or (N, H, W).
        n_max: The maximum number of images to show.
    Returns:
        None
    """
    num_images = min(x.shape[0], n_max)
    if x.ndim == 4:
        x = np.transpose(x, (0, 2, 3, 1))
        x_hat = np.transpose(x_hat, (0, 2, 3, 1))
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols / nrows * fig_height, fig_height),
    )
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, img, img_hat in zip(axes.flat, x[:num_images], x_hat[:num_images]):
        concatenated = np.concatenate([img, img_hat], axis=1)
        border_width = concatenated.shape[1] // 128 or 1
        mid = concatenated.shape[1] // 2
        concatenated[:, mid - border_width : mid + border_width] = 1.0
        ax.imshow(concatenated, cmap="Greys_r", interpolation="nearest")
        ax.axis("off")
    if show:
        plt.show()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight")
    plt.close()

    return PIL.Image.open(buffer)


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
    plt.close()


def draw_shapes(
    path: Path = repo_root / "img/shapes.png",
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
    img_size: int = 128,
    fg_color: str = "whitesmoke",
    bg_color: str = "white",
):
    """Plot an n x n grid of random shapes.
    Args:
        path: The path to save the image to.
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
        img_size: The size of the image in pixels.
        bg_color: The color of the background plot area.
    Returns:
        None
    """
    _, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols / nrows * fig_height, fig_height),
        layout="tight",
        subplot_kw={"aspect": 1.0},
        facecolor=bg_color,
    )
    dataset = InfiniteDSprites(img_size=img_size)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.flat:
        spline = dataset.generate_shape()
        ax.axis("off")
        ax.plot(spline[0], spline[1], label="spline", color=fg_color)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)


def draw_shapes_animated(
    path: Path = repo_root / "img/shapes.gif",
    nrows: int = 5,
    ncols: int = 11,
    fig_height: float = 10,
    img_size: int = 256,
    bg_color: str = "white",
    duration: int = 2,
    fps: int = 60,
    factor=None,
):
    """Create an animated GIF showing a grid of shapes undergoing transformations.
    Args:
        path: The path to save the image to.
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
        img_size: The size of the image in pixels.
        bg_color: The color of the background plot area.
        duration: The duration of the animation in seconds.
        fps: The number of frames per second.
        factor: The factor to vary. If None, all factors are varied.
    Returns:
        None
    """
    num_frames = fps * duration
    dataset = InfiniteDSprites(
        img_size=img_size,
        color_range=COLORS,
        scale_range=np.linspace(0.0, 1.0, num_frames),
        orientation_range=np.linspace(0.0, 2 * np.pi, num_frames),
        position_x_range=np.linspace(0.0, 1.0, num_frames),
        position_y_range=np.linspace(0.0, 1.0, num_frames),
    )
    shapes = [InfiniteDSprites.generate_shape() for _ in range(nrows * ncols)]
    colors = [dataset.sample_latents().color for _ in range(nrows * ncols)]
    if factor is None:
        factors = generate_multiple_factor_progression(dataset)
    else:
        path = path.with_stem(f"{path.stem}_{factor}")
        factors = generate_single_factor_progression(dataset, factor)

    frames = [
        [
            dataset.draw(
                Latents(color, shape, scale, orientation, position_x, position_y),
                channels_first=False,
            )
            for shape, color in zip(shapes, colors)
        ]
        for scale, orientation, position_x, position_y in zip(*factors)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I", fps=fps) as writer:
        for frame in tqdm(frames):
            _, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols / nrows * fig_height, fig_height),
                layout="tight",
                subplot_kw={"aspect": 1.0},
                facecolor=bg_color,
            )
            buffer = io.BytesIO()
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for ax, image in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image)
            plt.savefig(buffer, format="png")
            plt.close()
            image = imageio.imread(buffer)
            writer.append_data(image)  # type: ignore


def generate_multiple_factor_progression(dataset):
    """Generate a sequence of factors that can be used to animate a shape.
    Args:
        scale_range: The range of scales to use.
        orientation_range: The range of orientations to use.
        position_x_range: The range of x positions to use.
        position_y_range: The range of y positions to use.
    Returns:
        A tuple of factor value sequences representing a smooth animation.
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


def generate_single_factor_progression(dataset, factor):
    """Generate a smooth progression of a single factor."""
    length = 2 * len(dataset.ranges[factor])
    factors = {
        "scale": np.ones(length) * 0.5,
        "orientation": np.ones(length) * 0.0,
        "position_x": np.ones(length) * 0.5,
        "position_y": np.ones(length) * 0.5,
    }
    factors[factor][: length // 2] = dataset.ranges[factor]
    factors[factor][length // 2 :] = dataset.ranges[factor][::-1]
    return (
        factors["scale"],
        factors["orientation"],
        factors["position_x"],
        factors["position_y"],
    )


def draw_triplet(path: Path = repo_root / "img/triplet.png", fig_height: float = 10):
    """Plot a triplet of shapes form the InfiniteDSpritesTriplets.
    See Montero et al. 2020 for details of the composition task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSpritesTriplets(img_size=256)
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


def draw_smooth_shapes(
    path: Path = repo_root / "img/smooth_shapes.gif",
    nrows: int = 5,
    ncols: int = 11,
    fig_height: float = 10,
    img_size: int = 256,
    bg_color="white",
    num_shapes: int = 10,
    duration_per_shape: int = 2,
    fps: int = 60,
):
    """Smoothly interpolate between shapes and colors.
    Args:
        path: The path to save the animation to.
        fig_height: The height of the figure in inches.
        num_shapes: The number of shapes to interpolate between.
        duration: The duration of the animation in seconds.
        fps: The number of frames per second.
    """
    dataset = InfiniteDSprites(img_size=img_size, color_range=COLORS)
    colors = [
        [dataset.sample_latents().color for _ in range(num_shapes)]
        for _ in range(nrows * ncols)
    ]
    shapes = [
        [InfiniteDSprites.generate_shape() for _ in range(num_shapes)]
        for _ in range(nrows * ncols)
    ]

    # interpolate between shapes
    shape_sequences = []
    for shape in shapes:
        shape.append(shape[0])
        shape_sequence = []
        for start, end in zip(shape[:-1], shape[1:]):
            shape_sequence.extend(np.linspace(start, end, duration_per_shape * fps))
        shape_sequences.append(shape_sequence)

    # interpolate between colors
    color_sequences = []
    for color in colors:
        color.append(color[0])
        color_sequence = []
        for start, end in zip(color[:-1], color[1:]):
            color_sequence.extend(np.linspace(start, end, duration_per_shape * fps))
        color_sequences.append(color_sequence)

    shape_sequences = zip(*shape_sequences)
    color_sequences = zip(*color_sequences)

    frames = [
        [
            dataset.draw(
                Latents(
                    shape=shape,
                    color=color,
                    scale=0.7,
                    orientation=0.8,
                    position_x=0.5,
                    position_y=0.5,
                ),
                channels_first=False,
            )
            for shape, color in zip(shape_sequence, color_sequence)
        ]
        for shape_sequence, color_sequence in zip(shape_sequences, color_sequences)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I", fps=fps) as writer:
        for frame in tqdm(frames):
            _, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols / nrows * fig_height, fig_height),
                layout="tight",
                subplot_kw={"aspect": 1.0},
                facecolor=bg_color,
            )
            buffer = io.BytesIO()
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for ax, img in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(img)
            plt.savefig(buffer, format="png")
            plt.close()
            image = imageio.imread(buffer)
            writer.append_data(image)  # type: ignore


def draw_classification_task(
    path: Path = repo_root / "img/classification.png", fig_height: float = 10
):
    """Draw an example of the binary classification task.
    Args:
        fig_height: The height of the figure in inches.
    Returns:
        None
    """
    dataset = InfiniteDSprites(img_size=256)
    latents_reference = dataset.sample_latents()
    latents_same = dataset.sample_latents()
    latents_different = dataset.sample_latents()

    reference = dataset.draw(latents_reference, channels_first=False)
    for latent in ["shape", "scale", "orientation", "position_x", "position_y"]:
        same = dataset.draw(
            latents_same._replace(**{latent: latents_reference[latent]}),
            channels_first=False,
        )
        different = dataset.draw(latents_different, channels_first=False)
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
    dataset = InfiniteDSpritesAnalogies(img_size=512)
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
    dataset = InfiniteDSprites(img_size=256)
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
