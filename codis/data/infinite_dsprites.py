import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splprep


def plot_shapes_on_grid(n=3):
    """Plot an n x n grid of random shapes."""
    _, axes = plt.subplots(
        nrows=n, ncols=n, figsize=(15, 15), layout="tight", subplot_kw={"aspect": 1.0}
    )
    for ax in axes.flat:
        verts, spline = generate_shape(
            min_verts=4, max_verts=10, radius_std=0.5, angle_std=0.5
        )
        ax.axis("off")
        ax.scatter(verts[0], verts[1], label="vertices", color="blue")
        ax.scatter([0], [0], label="origin", color="green")
        ax.plot(spline[0], spline[1], label="spline", color="red")
    plt.savefig("shapes.png")


def generate_shape(
    min_verts: int = 3,
    max_verts: int = 10,
    radius_std: float = 0.5,
    angle_std: float = 0.5,
):
    """Generate a random polygon and optionally interpolate it with a spline.
    Args:
        min_verts: Minimum number of vertices (inclusive).
        max_verts: Maximum number of vertices (inclusive).
        radius_std: Standard deviation of the polar radius when sampling the vertices.
        angle_std: Standard deviation of the polar angle when sampling the vertices.
    Returns:
        A tuple of (points, spline), where points is an array of points of shape (2, num_verts)
        and spline is an array of shape (2, num_spline_points).
    """
    n = np.random.randint(min_verts, max_verts + 1)
    verts = sample_vertices(n, radius_std, angle_std)
    spline = interpolate(verts) if np.random.rand() < 0.5 else interpolate(verts, k=1)
    return verts, spline


def sample_vertices(n: int, radius_std: float, angle_std: float):
    """Sample polar coordinates of the vertices.
    Args:
        n: Number of vertices.
        radius_std: Standard deviation of the polar radius when sampling the vertices.
        angle_std: Standard deviation of the polar angle when sampling the vertices.
    Returns:
        An array of shape (2, num_verts).
    """
    radia = np.random.normal(1.0, radius_std, n)
    radia = np.clip(radia, 0.1, 1.9)

    sector = 2 * np.pi / n
    intervals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.random.normal(0.0, sector / 2 * angle_std, n)
    angles = np.clip(angles, -sector / 2, sector / 2) + intervals

    points = [
        [radius * np.cos(angle), radius * np.sin(angle)]
        for radius, angle in zip(radia, angles)
    ]
    return np.array(points).T


def interpolate(verts, k: int = 3, num_spline_points: int = 1000):
    """Interpolate a set of vertices with a spline.
    Args:
        verts: An array of shape (2, num_verts).
        k: The degree of the spline.
        num_spline_points: The number of points to sample from the spline.
    Returns:
        An array of shape (2, num_spline_points).
    """
    verts = np.column_stack((verts, verts[:, 0]))
    spline_params, u = splprep(verts, s=0.0, per=0, k=k)
    u_new = np.linspace(u.min(), u.max(), num_spline_points)
    x, y = splev(u_new, spline_params, der=0)
    return np.array([x, y])


if __name__ == "__main__":
    plot_shapes_on_grid()
