import os
import pickle
from collections import namedtuple
from itertools import islice, product
from typing import Iterable

import imageio
import numpy as np
import pygame
import pygame.gfxdraw
from scipy.interpolate import splev, splprep
from torch.utils.data import IterableDataset

BaseLatents = namedtuple(
    "BaseLatents", "color shape scale orientation position_x, position_y"
)


class Latents(BaseLatents):
    # TODO: use a dataclass instead of a namedtuple, but make sure it can be used with a PyTorch dataloader
    """Latent variables defining a single image."""

    def to(self, device):
        return Latents(
            self.color.to(device),
            self.shape.to(device),
            self.scale.to(device),
            self.orientation.to(device),
            self.position_x.to(device),
            self.position_y.to(device),
        )


class InfiniteDSprites(IterableDataset):
    def __init__(
        self,
        image_size: int = 64,
        scale_range: Iterable = np.linspace(0.5, 1, 6),
        orientation_range: Iterable = np.linspace(0, 2 * np.pi, 40),  # TODO: apply
        position_x_range: Iterable = np.linspace(0, 1, 32),
        position_y_range: Iterable = np.linspace(0, 1, 32),
    ):
        self.image_size = image_size
        self.scale_range = scale_range
        self.orientation_range = orientation_range
        self.position_x_range = position_x_range
        self.position_y_range = position_y_range

    def __iter__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        window = pygame.display.set_mode((self.image_size, self.image_size))
        color = 0
        while True:
            _, shape = self.generate_shape()
            shape = pickle.dumps(shape)
            for scale, orientation, position_x, position_y in product(
                self.scale_range,
                self.orientation_range,
                self.position_x_range,
                self.position_y_range,
            ):
                latents = Latents(
                    color, shape, scale, orientation, position_x, position_y
                )
                image = self.draw(window, latents)
                yield image, latents

    def generate_shape(
        self,
        min_verts: int = 3,
        max_verts: int = 10,
        radius_std: float = 0.5,
        angle_std: float = 0.5,
        # TODO: fix the sorting when using a higher angle variance
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
        verts = self.sample_vertices(n, radius_std, angle_std)
        spline = (
            self.interpolate(verts)
            if np.random.rand() < 0.5
            else self.interpolate(verts, k=1)
        )
        return verts, spline

    @staticmethod
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

    @staticmethod
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
        spline_params, u = splprep(verts, s=0, per=1, k=k)
        u_new = np.linspace(u.min(), u.max(), num_spline_points)
        x, y = splev(u_new, spline_params, der=0)
        return np.array([x, y])

    @staticmethod
    def draw(window: pygame.Surface, latents: Latents):
        """Draw an image based on the values of the latents.
        Args:
            window: The pygame window to draw on.
            latents: The latents to use for drawing.
        Returns:
            The image as a numpy array.
        """
        window.fill(pygame.Color("black"))
        height, width = window.get_size()
        base_scale = 0.1 * height
        position = np.array(
            [
                0.25 * height + 0.5 * height * latents.position_y,
                0.25 * width + 0.5 * width * latents.position_x,
            ]
        ).reshape(2, 1)
        shape = base_scale * latents.scale * pickle.loads(latents.shape) + position
        pygame.gfxdraw.aapolygon(window, shape.T.tolist(), (255, 255, 255))
        pygame.draw.polygon(window, pygame.Color("white"), shape.T.tolist())
        pygame.display.update()
        return pygame.surfarray.array3d(window)


if __name__ == "__main__":
    dataset = InfiniteDSprites(image_size=512)
    writer = imageio.get_writer("infinite_dsprites.gif", mode="I")
    for image, _ in islice(dataset, 1000):
        writer.append_data(image)
    writer.close()
