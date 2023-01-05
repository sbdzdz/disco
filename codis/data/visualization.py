from matplotlib import pyplot as plt
from codis.data.infinite_dsprites import generate_shape


def plot_shapes_on_grid(n=8):
    """Plot an n x n grid of random shapes.
    Args:
        n: The number of rows and columns in the grid.
    Returns:
        None
    """
    _, axes = plt.subplots(
        nrows=n, ncols=n, figsize=(20, 20), layout="tight", subplot_kw={"aspect": 1.0}
    )
    for ax in axes.flat:
        verts, spline = generate_shape(
            min_verts=4, max_verts=10, radius_std=0.5, angle_std=0.7
        )
        ax.axis("off")
        ax.scatter(verts[0], verts[1], label="vertices", color="blue")
        ax.plot(spline[0], spline[1], label="spline", color="red")
    plt.savefig("shapes.png")


if __name__ == "__main__":
    plot_shapes_on_grid()
