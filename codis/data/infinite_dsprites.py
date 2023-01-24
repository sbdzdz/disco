import os
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
        min_verts: int = 3,
        max_verts: int = 10,
        radius_std: float = 0.8,
        angle_std: float = 0.8,
    ):
        """Create a dataset of images of random shapes.
        Args:
            image_size: The size of the images in pixels.
            scale_range: The range of scales to use.
            orientation_range: The range of orientations to use.
            position_x_range: The range of x positions to use.
            position_y_range: The range of y positions to use.
            min_verts: The minimum number of vertices in the shape.
            max_verts: The maximum number of vertices in the shape.
            radius_std: The standard deviation of the radius of the vertices.
            angle_std: The standard deviation of the angle of the vertices.
        Returns:
            None
        """
        pygame.init()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.image_size = image_size
        self.scale_range = scale_range
        self.orientation_range = orientation_range
        self.position_x_range = position_x_range
        self.position_y_range = position_y_range
        self.min_verts = min_verts
        self.max_verts = max_verts
        self.radius_std = radius_std
        self.angle_std = angle_std
        self.window = pygame.display.set_mode((self.image_size, self.image_size))

    def __del__(self):
        pygame.quit()

    def __iter__(self):
        color = 0
        while True:
            shape = self.generate_shape()
            for scale, orientation, position_x, position_y in product(
                self.scale_range,
                self.orientation_range,
                self.position_x_range,
                self.position_y_range,
            ):
                latents = Latents(
                    color, shape, scale, orientation, position_x, position_y
                )
                image = self.draw(latents)
                yield image, latents

    def generate_shape(self):
        """Generate random vertices and connect them with straight lines or a smooth curve.
        Args:
            min_verts: Minimum number of vertices (inclusive).
            max_verts: Maximum number of vertices (inclusive).
            radius_std: Standard deviation of the polar radius when sampling the vertices.
            angle_std: Standard deviation of the polar angle when sampling the vertices.
        Returns:
            A tuple of (points, spline), where points is an array of points of shape (2, num_verts)
            and spline is an array of shape (2, num_spline_points).
        """
        verts = self.sample_vertex_positions()
        shape = (
            self.interpolate(verts)
            if np.random.rand() < 0.5
            else self.interpolate(verts, k=1)
        )
        shape = shape / np.max(np.linalg.norm(shape, axis=0))  # normalize scale
        shape = shape - np.mean(shape, axis=1, keepdims=True)  # center shape
        return shape

    def sample_vertex_positions(self):
        """Sample the positions of the vertices of a polygon.
        Args:
            radius_std: Standard deviation of the polar radius when sampling the vertices.
            angle_std: Standard deviation of the polar angle when sampling the vertices.
            num_verts: Number of vertices.
        Returns:
            An array of shape (2, num_verts).
        """
        num_verts = np.random.randint(self.min_verts, self.max_verts + 1)
        rs = np.random.normal(1.0, self.radius_std, num_verts)
        rs = np.clip(rs, 0.1, 1.9)

        epsilon = 1e-6
        circle_sector = np.pi / num_verts - epsilon
        intervals = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        thetas = np.random.normal(0.0, circle_sector * self.angle_std, num_verts)
        thetas = np.clip(thetas, -circle_sector, circle_sector) + intervals

        verts = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(rs, thetas)]
        verts = np.array(verts).T
        return verts

    @staticmethod
    def interpolate(verts: np.array, k: int = 3, num_spline_points: int = 1000):
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

    def draw(self, latents: Latents):
        """Draw an image based on the values of the latents.
        Args:
            window: The pygame window to draw on.
            latents: The latents to use for drawing.
        Returns:
            The image as a numpy array.
        """
        self.window.fill(pygame.Color("black"))
        shape = self.apply_scale(latents.shape, latents.scale)
        shape = self.apply_orientation(shape, latents.orientation)
        shape = self.apply_position(shape, latents.position_x, latents.position_y)
        pygame.gfxdraw.aapolygon(self.window, shape.T.tolist(), (255, 255, 255))
        pygame.draw.polygon(self.window, pygame.Color("white"), shape.T.tolist())
        pygame.display.update()
        return pygame.surfarray.array3d(self.window)

    def apply_scale(self, shape, scale):
        """Apply a scale to a shape."""
        return 0.2 * self.image_size * scale * shape

    def apply_orientation(self, shape, orientation):
        """Rotate the shape by the given orientation."""
        rotation_matrix = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        return rotation_matrix @ shape

    def apply_position(self, shape, position_x, position_y):
        """Apply a position to a shape."""
        height, width = self.window.get_size()
        position = np.array(
            [
                0.25 * height + 0.5 * height * position_y,
                0.25 * width + 0.5 * width * position_x,
            ]
        ).reshape(2, 1)
        return shape + position


if __name__ == "__main__":
    dataset = InfiniteDSprites(image_size=512)
    writer = imageio.get_writer("infinite_dsprites.gif", mode="I")
    for image, _ in islice(dataset, 1000):
        writer.append_data(image)
    writer.close()
