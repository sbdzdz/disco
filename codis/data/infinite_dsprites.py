"""Class definitions for the infinite dSprites dataset."""
import os
from collections import namedtuple
from itertools import product

import numpy as np
import numpy.typing as npt
import pygame
import pygame.gfxdraw
from matplotlib import colors
from scipy.interpolate import splev, splprep
from shapely import geometry
from shapely.ops import unary_union
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, IterableDataset

BaseLatents = namedtuple(
    "BaseLatents", "color shape scale orientation position_x, position_y"
)


class Latents(BaseLatents):
    """Latent variables defining a single image."""

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, device):
        """Move the latents to a device."""
        return Latents(
            color=self.color.to(device),
            shape=self.shape.to(device),
            scale=self.scale.to(device),
            orientation=self.orientation.to(device),
            position_x=self.position_x.to(device),
            position_y=self.position_y.to(device),
        )


# pylint: disable=abstract-method
class InfiniteDSprites(IterableDataset):
    """Infinite dataset of procedurally generated shapes undergoing transformations."""

    def __init__(
        self,
        img_size: int = 256,
        color_range=("white",),
        scale_range=np.linspace(0.5, 1.5, 32),
        orientation_range=np.linspace(0, 2 * np.pi, 32),
        position_x_range=np.linspace(0, 1, 32),
        position_y_range=np.linspace(0, 1, 32),
        dataset_size: int = None,
        shapes: list = None,
    ):
        """Create a dataset of images of random shapes.
        Args:
            img_size: The size of the images in pixels.
            scale_range: The range of scales to use.
            orientation_range: The range of orientations to use.
            position_x_range: The range of x positions to use.
            position_y_range: The range of y positions to use.
            dataset_size: The number of images to generate. Note that `shapes` also controls
                the number of images generated.
            shapes: The number of shapes to generate or a list of shapes to use. Set
                to None to generate random shapes forever.
        Returns:
            None
        """
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        self.img_size = img_size
        self.window = pygame.display.set_mode((self.img_size, self.img_size))
        self.ranges = {
            "color": color_range,
            "scale": scale_range,
            "orientation": orientation_range,
            "position_x": position_x_range,
            "position_y": position_y_range,
        }
        self.num_latents = len(self.ranges) + 1
        self.dataset_size = dataset_size
        self.counter = 0
        self.current_shape_index = 0
        self.shapes = shapes

    @classmethod
    def from_config(cls, config: dict):
        """Create a dataset from a config."""
        for key, value in config.items():
            if isinstance(value, dict) and set(value.keys()) == {
                "start",
                "stop",
                "num",
            }:
                config[key] = np.linspace(**value)
        return cls(**config)

    def __iter__(self):
        """Generate an infinite stream of images and latent vectors.
        Args:
            None
        Returns:
            An infinite stream of (image, latents) tuples."""
        while True:
            if self.shapes is None:
                shape = self.generate_shape()  # infinite variant
            elif isinstance(self.shapes, list):
                if self.current_shape_index >= len(self.shapes):
                    return
                shape = self.shapes[self.current_shape_index]
                self.current_shape_index += 1
            elif isinstance(self.shapes, int):
                if self.current_shape_index >= self.shapes:
                    return
                shape = self.generate_shape()
                self.current_shape_index += 1

            for color, scale, orientation, position_x, position_y in product(
                *self.ranges.values()
            ):
                if self.dataset_size is not None and self.counter >= self.dataset_size:
                    return
                self.counter += 1
                color = np.array(colors.to_rgb(color))
                latents = Latents(
                    color, shape, scale, orientation, position_x, position_y
                )
                img = self.draw(latents)
                yield img, latents

    def generate_shape(self):
        """Generate random vertices and connect them with straight lines or a smooth curve.
        Args:
            None
        Returns:
            An array of shape (2, num_verts).
        """
        verts = self.sample_vertex_positions()
        shape = (
            self.interpolate(verts)
            if np.random.rand() < 0.5
            else self.interpolate(verts, k=1)
        )
        shape = self.center_and_scale(shape)
        shape = self.align(shape)

        return shape

    def center_and_scale(self, shape):
        """Center the shape and normalize scale."""
        latents = Latents(
            color=np.array([0.9, 0.9, 0.9]),
            shape=shape,
            scale=1.0,
            orientation=0.0,
            position_x=0.5,
            position_y=0.5,
        )
        img = self.draw(latents, channels_first=False)
        non_black_pixels = np.argwhere(np.any(img != [0, 0, 0], axis=2))
        center = np.mean(non_black_pixels, axis=0)
        max_dist = np.max(np.linalg.norm(non_black_pixels - center, axis=1))

        center = np.expand_dims(center - self.img_size / 2, 1) / (0.2 * self.img_size)
        max_dist = max_dist / (0.2 * self.img_size)

        shape = shape - center
        shape = shape / max_dist

        return shape

    def align(self, shape):
        """Align the principal axis of the shape with the y-axis."""
        pca = PCA(n_components=2)
        pca.fit(shape.T)

        # Get the principal components
        principal_components = pca.components_

        # Find the angle between the major axis and the y-axis
        major_axis = principal_components[0]
        angle_rad = np.arctan2(major_axis[1], major_axis[0])
        shape = self.apply_orientation(shape, -angle_rad)

        return shape

    def sample_vertex_positions(
        self,
        min_verts: int = 5,
        max_verts: int = 8,
        radius_std: float = 0.8,
        angle_std: float = 0.5,
    ):
        """Sample the positions of the vertices of a polygon.
        Args:
            min_verts: Minimum number of vertices (inclusive).
            max_verts: Maximum number of vertices (inclusive).
            radius_std: Standard deviation of the polar radius when sampling the vertices.
            angle_std: Standard deviation of the polar angle when sampling the vertices.
        Returns:
            An array of shape (2, num_verts).
        """
        num_verts = np.random.randint(min_verts, max_verts + 1)
        rs = np.random.normal(1.0, radius_std, num_verts)
        rs = np.clip(rs, 0.1, 1.9)

        epsilon = 1e-6
        circle_sector = np.pi / num_verts - epsilon
        intervals = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        thetas = np.random.normal(0.0, circle_sector * angle_std, num_verts)
        thetas = np.clip(thetas, -circle_sector, circle_sector) + intervals

        verts = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(rs, thetas)]
        verts = np.array(verts).T
        return verts

    def interpolate(
        self, verts: npt.NDArray, k: int = 3, num_spline_points: int = 1000
    ):
        """Interpolate a set of vertices with a spline.
        Args:
            verts: An array of shape (2, num_verts).
            k: The degree of the spline.
            num_spline_points: The number of points to sample from the spline.
        Returns:
            An array of shape (2, num_spline_points).
        """
        verts = np.column_stack((verts, verts[:, 0]))
        spline_params, u, *_ = splprep(verts, s=0, per=1, k=k)
        u_new = np.linspace(u.min(), u.max(), num_spline_points)
        x, y = splev(u_new, spline_params, der=0)
        return np.array([x, y])

    def draw(self, latents: Latents, channels_first=True):
        """Draw an image based on the values of the latents.
        Args:
            window: The pygame window to draw on.
            latents: The latents to use for drawing.
            channels_first: Whether to return the image with the channel dimension first.
        Returns:
            The image as a numpy array.
        """
        self.window.fill(pygame.Color("black"))
        shape = self.apply_scale(latents.shape, latents.scale)
        shape = self.apply_orientation(shape, latents.orientation)
        shape = self.apply_position(shape, latents.position_x, latents.position_y)
        color = tuple(int(255 * c) for c in latents.color)
        pygame.gfxdraw.aapolygon(self.window, shape.T.tolist(), color)
        pygame.draw.polygon(self.window, color, shape.T.tolist())

        center = np.mean(shape, axis=1)
        outline = shape - center.reshape(2, 1)
        polygon = geometry.Polygon(outline.T)

        offset_polygon = polygon.buffer(0)
        offset_polygon = offset_polygon.buffer(-2, join_style="mitre")

        if offset_polygon.geom_type == "MultiPolygon":
            offset_polygon = unary_union(offset_polygon)

        outline = np.array(offset_polygon.exterior.coords)
        print(outline.shape)
        outline = outline + center

        pygame.draw.polygon(self.window, pygame.Color("red"), (outline).tolist())

        pygame.display.update()
        image = pygame.surfarray.array3d(self.window)
        image = image.astype(np.float32) / 255.0
        if color == (255, 255, 255):
            image = image.mean(axis=2, keepdims=True)
        if channels_first:
            image = np.transpose(image, (2, 0, 1))
        return image

    def apply_scale(self, shape: npt.NDArray, scale: float):
        """Apply a scale to a shape."""
        height, _ = self.window.get_size()
        return 0.2 * height * scale * shape

    @staticmethod
    def apply_orientation(shape: npt.NDArray, orientation: float):
        """Apply an orientation to a shape.
        Args:
            shape: An array of shape (2, num_points).
            orientation: The orientation in radians.
        Returns:
            The rotated shape.
        """
        rotation_matrix = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        return rotation_matrix @ shape

    def apply_position(self, shape: npt.NDArray, position_x: float, position_y: float):
        """Apply a position to a shape.
        Args:
            shape: An array of shape (2, num_points).
            position_x: The x position of the shape.
            position_y: The y position of the shape.
        Returns:
            An array of shape (2, num_points).
        """
        height, width = self.window.get_size()
        position = np.array(
            [
                0.25 * height + 0.5 * height * position_y,
                0.25 * width + 0.5 * width * position_x,
            ]
        ).reshape(2, 1)
        return shape + position

    def sample_latents(self):
        """Sample a random set of latents."""
        return Latents(
            color=np.array(colors.to_rgb(np.random.choice(self.ranges["color"]))),
            shape=self.generate_shape(),
            scale=np.random.choice(self.ranges["scale"]),
            orientation=np.random.choice(self.ranges["orientation"]),
            position_x=np.random.choice(self.ranges["position_x"]),
            position_y=np.random.choice(self.ranges["position_y"]),
        )


class ContinualDSpritesMap(Dataset):
    """Map-style (finite) continual learning dsprites dataset."""

    def __init__(self, *args, **kwargs):
        self.dataset = InfiniteDSprites(*args, **kwargs)
        assert (
            self.dataset.dataset_size is not None or self.dataset.shapes is not None
        ), "Dataset size must be finite. Please set dataset_size or pass a list of shapes."
        self.imgs, self.latents = zip(*list(self.dataset))
        self.imgs = list(self.imgs)
        self.latents = list(self.latents)

    def __len__(self):
        if self.dataset.dataset_size is not None:
            return self.dataset.dataset_size
        return len(list(product(*self.dataset.ranges.values()))) * len(
            self.dataset.shapes
        )

    def __getitem__(self, index):
        return self.imgs[index], self.latents[index]


class RandomDSprites(InfiniteDSprites):
    """Infinite dataset of randomly transformed shapes.
    The shape is sampled from a given list or generated procedurally.
    The transformations are sampled randomly at every step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Generate an infinite stream of images.
        Args:
            None
        Yields:
            A tuple of (image, latents).
        """
        while self.dataset_size is None or self.counter < self.dataset_size:
            self.counter += 1
            if self.shapes is not None:
                shape = self.shapes[np.random.choice(len(self.shapes))]
            else:
                shape = self.generate_shape()
            latents = self.sample_latents()._replace(shape=shape)
            image = self.draw(latents)
            yield image, latents


class RandomDSpritesMap(Dataset):
    """Map-style (finite) random dsprites dataset."""

    def __init__(self, *args, **kwargs) -> None:
        self.dataset = RandomDSprites(*args, **kwargs)
        assert (
            self.dataset.dataset_size is not None
        ), "Dataset size must be finite. Please set dataset_size."
        self.imgs, self.latents = zip(*list(self.dataset))
        self.imgs = list(self.imgs)
        self.latents = list(self.latents)

    def __len__(self):
        if self.dataset.dataset_size is not None:
            return self.dataset.dataset_size
        return len(list(product(*self.dataset.ranges.values()))) * len(
            self.dataset.shapes
        )

    def __getitem__(self, index):
        return self.imgs[index], self.latents[index]


class InfiniteDSpritesTriplets(InfiniteDSprites):
    """Infinite dataset of triplets of images.
    For details see the composition task proposed by Montero et al. (2020).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Generate an infinite stream of tuples consisting of a triplet of images and an action encoding.
        Args:
            None
        Yields:
            A tuple of ((image_original, image_transform, image_target), action).
        """
        while self.dataset_size is None or self.counter < self.dataset_size:
            action = np.random.choice(list(self.ranges.keys()))
            latents_original = self.sample_latents()
            latents_transform = self.sample_latents()
            if action != "shape" and np.allclose(
                latents_original[action], latents_transform[action]
            ):
                continue
            self.counter += 1
            latents_target = latents_original._replace(
                **{action: latents_transform[action]}
            )
            image_original = self.draw(latents_original)
            image_transform = self.draw(latents_transform)
            image_target = self.draw(latents_target)
            yield ((image_original, image_transform, image_target), action)


class InfiniteDSpritesAnalogies(InfiniteDSprites):
    """Infinite dataset of image analogies."""

    def __init__(self, *args, reference_shape=None, query_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = pygame.display.set_mode((self.img_size // 2, self.img_size // 2))
        self.reference_shape = reference_shape
        self.query_shape = query_shape

    def __iter__(self):
        """Generate an infinite stream of images representing an analogy task.
        Each output array represents a 2x2 grid of images. Top row: reference
        source, reference target. Bottom row: query source, query target. The task
        is to infer a transformation between reference source and reference target
        and apply it to query source to obtain query target. Reference source and
        query source differ only in shape.
        Args:
            None
        Yields:
            An image grid as a single numpy array.
        """
        while self.dataset_size is None or self.counter < self.dataset_size:
            self.counter += 1
            source_latents = self.sample_latents()
            target_latents = self.sample_latents()

            reference_color = colors.to_rgb(np.random.choice(self.ranges["color"]))
            reference_shape = (
                self.reference_shape
                if self.reference_shape is not None
                else self.generate_shape()
            )

            query_color = colors.to_rgb(np.random.choice(self.ranges["color"]))
            query_shape = (
                self.query_shape
                if self.query_shape is not None
                else self.generate_shape()
            )

            reference_source, reference_target, query_source, query_target = (
                self.draw(
                    source_latents._replace(
                        shape=reference_shape, color=reference_color
                    )
                ),
                self.draw(
                    target_latents._replace(
                        shape=reference_shape, color=reference_color
                    )
                ),
                self.draw(
                    source_latents._replace(shape=query_shape, color=query_color)
                ),
                self.draw(
                    target_latents._replace(shape=query_shape, color=query_color)
                ),
            )
            grid = np.concatenate(
                [
                    np.concatenate([reference_source, reference_target], axis=2),
                    np.concatenate([query_source, query_target], axis=2),
                ],
                axis=1,
            )

            # add horizontal and vertical borders
            border_width = self.img_size // 128 or 1
            mid = self.img_size // 2
            grid[:, mid - border_width : mid + border_width, :] = 1.0
            grid[:, :, mid - border_width : mid + border_width] = 1.0

            yield grid
