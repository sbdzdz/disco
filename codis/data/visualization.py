import io

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from tqdm import tqdm

from codis.data.infinite_dsprites import InfiniteDSprites


def plot_shapes_on_grid(n=5):
    """Plot an n x n grid of random shapes.
    Args:
        n: The number of rows and columns in the grid.
    Returns:
        None
    """
    _, axes = plt.subplots(
        nrows=n, ncols=n, figsize=(20, 20), layout="tight", subplot_kw={"aspect": 1.0}
    )
    dataset = InfiniteDSprites()
    for ax in axes.flat:
        verts, spline = dataset.sample_polygon()
        ax.axis("off")
        ax.scatter(verts[0], verts[1], label="vertices", color="blue")
        ax.plot(spline[0], spline[1], label="spline", color="red")
    plt.savefig("shapes.png")


def plot_animated_shapes_on_grid(nrows=5, ncols=12, num_frames=10, height=10):
    """Create an animated GIF showing a grid of n x n InfiniteDSprites datasets."""
    datasets = [iter(InfiniteDSprites(image_size=128)) for _ in range(nrows * ncols)]
    frames = [[next(dataset) for dataset in datasets] for _ in range(num_frames)]
    aspect_ratio = ncols / nrows
    with imageio.get_writer("zoom_out.gif", mode="I") as writer:
        for frame in tqdm(frames):
            _, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(aspect_ratio * height, height),
                layout="tight",
                subplot_kw={"aspect": 1.0},
            )
            for ax, (image, _) in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image)
                buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            image = imageio.imread(buffer)
            writer.append_data(image)


if __name__ == "__main__":
    plot_animated_shapes_on_grid()
