import io
from itertools import islice

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
        verts, spline = dataset.generate_shape()
        ax.axis("off")
        ax.scatter(verts[0], verts[1], label="vertices", color="blue")
        ax.plot(spline[0], spline[1], label="spline", color="red")
    plt.savefig("shapes.png")


def zoom_out_vis(n=9, num_frames=100):
    """Create an animated GIF showing a grid of n x n InfiniteDSprites datasets."""
    datasets = [iter(InfiniteDSprites()) for _ in range(n**2)]
    frames = [[next(dataset) for dataset in datasets] for _ in range(num_frames)]
    _, axes = plt.subplots(
        nrows=n, ncols=n, figsize=(20, 20), layout="tight", subplot_kw={"aspect": 1.0}
    )
    with imageio.get_writer("zoom_out.gif", mode="I") as writer:
        for frame in tqdm(frames):
            for ax, (image, _) in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image)
                buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            image = imageio.imread(buffer)
            writer.append_data(image)
            plt.cla()


if __name__ == "__main__":
    zoom_out_vis()
