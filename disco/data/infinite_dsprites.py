"""Class definitions for the infinite dSprites dataset."""

from collections import namedtuple
from itertools import product

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from scipy.interpolate import splev, splprep
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, IterableDataset
from numba import jit

BaseLatents = namedtuple(
    "BaseLatents", "color shape shape_id scale orientation position_x, position_y"
)


class Latents(BaseLatents):
    """Latent variables defining a single image."""

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, args, **kwargs):
        """Move the latents to a device."""
        return Latents(
            color=self.color.to(*args, **kwargs),
            shape=self.shape.to(*args, **kwargs),
            shape_id=self.shape_id.to(*args, **kwargs),
            scale=self.scale.to(*args, **kwargs),
            orientation=self.orientation.to(*args, **kwargs),
            position_x=self.position_x.to(*args, **kwargs),
            position_y=self.position_y.to(*args, **kwargs),
        )

    def replace(self, **kwargs):
        return super()._replace(**kwargs)


class InfiniteDSprites(IterableDataset):
    """Infinite dataset of procedurally generated shapes undergoing transformations."""

    def __init__(
        self,
        img_size: int = 256,
        color_range=None,
        scale_range=None,
        orientation_range=None,
        position_x_range=None,
        position_y_range=None,
        dataset_size: int = None,
        shapes: list = None,
        shape_ids: list = None,
        orientation_marker: bool = True,
        orientation_marker_color="black",
        background_color="darkgray",
        grayscale: bool = False,
        **kwargs,
    ):
        """Create a dataset of images of random shapes.
        Args:
            img_size: The size of the images in pixels.
            scale_range: The range of scales to sample from.
            orientation_range: The range of orientations to sample from.
            position_x_range: The range of x positions to sample from.
            position_y_range: The range of y positions to sample from.
            dataset_size: The number of images to generate. Note that `shapes` also controls
                the number of images generated.
            shapes: The number of shapes to generate or a list of shapes to use. Set
                to None to generate random shapes forever.
            shape_ids: The IDs of the shapes. If None, the shape ID is set to its index.
            orientation_marker: Whether to draw stripes indicating the orientation of the shape.
            background_color: The color of the canvas background.
            grayscale: If set to True, the images will have a single color channel.
        Returns:
            None
        """
        self.img_size = img_size
        self.canvas_size = img_size
        if color_range is None:
            color_range = ["white"]
        if scale_range is None:
            scale_range = np.linspace(0.5, 1.0, 32)
        if orientation_range is None:
            orientation_range = np.linspace(0, 2 * np.pi * 32 / 33, 32)
        if position_x_range is None:
            position_x_range = np.linspace(0, 1, 32)
        if position_y_range is None:
            position_y_range = np.linspace(0, 1, 32)
        self.ranges = {
            "color": color_range,
            "scale": scale_range,
            "orientation": orientation_range,
            "position_x": position_x_range,
            "position_y": position_y_range,
        }
        self.scale_factor = 0.45
        self.num_latents = len(self.ranges) + 1
        self.dataset_size = dataset_size
        self.counter = 0
        self.current_shape_index = 0
        if isinstance(shapes, list):
            self.shapes = shapes
        elif isinstance(shapes, int):
            self.shapes = [self.generate_shape() for _ in range(shapes)]
        else:
            self.shapes = None
        self.shape_ids = shape_ids
        self.orientation_marker = orientation_marker
        self.orientation_marker_color = tuple(
            int(255 * c) for c in colors.to_rgb(orientation_marker_color)
        )
        self.background_color = tuple(
            int(255 * c) for c in colors.to_rgb(background_color)
        )
        self.grayscale = grayscale

    @property
    def current_shape_id(self):
        """Return the ID of the current shape."""
        if self.shape_ids is None:
            return self.current_shape_index
        return self.shape_ids[self.current_shape_index]

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

            for color, scale, orientation, position_x, position_y in product(
                *self.ranges.values()
            ):
                if self.dataset_size is not None and self.counter >= self.dataset_size:
                    return
                self.counter += 1
                color = np.array(colors.to_rgb(color))
                latents = Latents(
                    color,
                    shape,
                    self.current_shape_id,
                    scale,
                    orientation,
                    position_x,
                    position_y,
                )
                img = self.draw(latents)
                yield img, latents
            self.current_shape_index += 1

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
        shape = self.align(shape)
        shape = self.center_and_scale(shape)

        return shape

    def center_and_scale(self, shape):
        """Center and scale a shape."""
        shape = shape - shape.mean(axis=1, keepdims=True)
        _, _, w, h = cv2.boundingRect((shape * 1000).T.astype(np.int32))
        shape[0, :] = shape[0, :] / (w / 1000)
        shape[1, :] = shape[1, :] / (h / 1000)

        transformed_shape = self.apply_scale(shape, 1)
        transformed_shape = self.apply_position(transformed_shape, 0.5, 0.5)
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3)).astype(np.int32)
        self.draw_shape(shape=transformed_shape, canvas=canvas, color=(255, 255, 255))
        center = self.get_center(canvas)[::-1]
        center = np.expand_dims(center - self.canvas_size // 2, 1) / (
            self.scale_factor * self.canvas_size
        )
        shape = shape - center

        return shape

    def align(self, shape):
        """Align the principal axis of the shape with the y-axis."""
        pca = PCA(n_components=2)
        pca.fit(shape.T)

        # Get the principal components
        principal_components = pca.components_

        # Find the angle between the major axis and the y-axis
        major_axis = principal_components[0]
        angle_rad = np.arctan2(major_axis[1], major_axis[0]) + 0.5 * np.pi
        shape = self.apply_orientation(shape, -angle_rad)

        return shape

    def sample_vertex_positions(
        self,
        min_verts: int = 5,
        max_verts: int = 8,
        radius_std: float = 0.4,
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

    def draw(self, latents: Latents, channels_first=True, debug=False):
        """Draw an image based on the values of the latents.
        Args:
            latents: The latents to use for drawing.
            channels_first: Whether to return the image with the channel dimension first.
        Returns:
            The image as a numpy array.
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.int32)
        canvas[:, :] = self.background_color
        shape = self.apply_scale(latents.shape, latents.scale)
        shape = self.apply_orientation(shape, latents.orientation)
        shape = self.apply_position(shape, latents.position_x, latents.position_y)
        color = tuple(int(255 * c) for c in latents.color)

        self.draw_shape(shape, canvas, color)
        if self.orientation_marker:
            self.draw_orientation_marker(canvas, latents)
        if debug:
            self.add_debug_info(shape, canvas)
        if self.grayscale:
            canvas = np.mean(canvas, axis=2, keepdims=True)
        if channels_first:
            canvas = np.transpose(canvas, (2, 0, 1))
        canvas = canvas.astype(np.float32) / 255.0
        return canvas

    def apply_scale(self, shape: npt.NDArray, scale: float):
        """Apply a scale to a shape."""
        height = self.canvas_size
        return self.scale_factor * height * scale * shape

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
        height, width = self.canvas_size, self.canvas_size
        position = np.array(
            [
                0.25 * width + 0.5 * width * position_x,
                0.25 * height + 0.5 * height * position_y,
            ]
        ).reshape(2, 1)
        return shape + position

    @staticmethod
    def draw_shape(shape, canvas, color):
        """Draw a shape on a canvas."""
        shape = shape.T.astype(np.int32)
        cv2.fillPoly(img=canvas, pts=[shape], color=color, lineType=cv2.LINE_AA)

    def draw_orientation_marker(self, canvas, latents):
        """Mark the right half of the shape."""
        theta = latents.orientation - np.pi / 2
        center = np.array([0, 0]).reshape(2, 1)
        y0, x0 = self.apply_position(center, latents.position_x, latents.position_y)

        def rotate_point(x, y):
            """Rotate the coordinate system by -theta around the center of the shape."""
            x_prime = (x - x0) * np.cos(-theta) + (y - y0) * np.sin(-theta) + x0
            y_prime = -(x - x0) * np.sin(-theta) + (y - y0) * np.cos(-theta) + y0
            return x_prime, y_prime

        shape_pixels = self.find_shape_pixels(canvas, self.background_color)
        x, _ = rotate_point(shape_pixels[:, 0], shape_pixels[:, 1])
        right_half = shape_pixels[x > x0]

        canvas[right_half[:, 0], right_half[:, 1]] = self.orientation_marker_color

    @staticmethod
    @jit(nopython=True)
    def find_shape_pixels(canvas, background_color):
        background_color = np.array(background_color, dtype=np.int32)

        mask = canvas == background_color
        mask = mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2]
        shape_pixels = np.argwhere(~mask)

        return shape_pixels

    def add_debug_info(self, shape, canvas):
        """Add debug info to the canvas."""
        shape_center = self.get_center(canvas)
        x, y, w, h = cv2.boundingRect(shape.T.astype(np.int32))
        cv2.rectangle(
            img=canvas, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2
        )
        cv2.circle(
            img=canvas,
            center=tuple(shape_center[::-1].astype(np.int32)),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )
        cv2.circle(
            img=canvas,
            center=(self.canvas_size // 2, self.canvas_size // 2),
            radius=5,
            color=(0, 255, 0),
            thickness=-1,
        )

    @staticmethod
    def get_center(canvas):
        """Get the center of the shape."""
        foreground_pixels = np.argwhere(np.any(canvas != [0, 0, 0], axis=2))
        return np.mean(foreground_pixels, axis=0)

    @staticmethod
    def is_monochrome(canvas):
        """Check if a canvas is monochrome (all channels are the same)."""
        return np.allclose(canvas[:, :, 0], canvas[:, :, 1]) and np.allclose(
            canvas[:, :, 1], canvas[:, :, 2]
        )

    def sample_latents(self):
        """Sample a random set of latents."""
        if self.shapes is not None:
            index = np.random.choice(len(self.shapes))
            self.current_shape_index = index
            shape = self.shapes[index]
        else:
            shape = self.generate_shape()
        return Latents(
            color=np.array(colors.to_rgb(np.random.choice(self.ranges["color"]))),
            shape=shape,
            shape_id=None,
            scale=np.random.choice(self.ranges["scale"]),
            orientation=np.random.choice(self.ranges["orientation"]),
            position_x=np.random.choice(self.ranges["position_x"]),
            position_y=np.random.choice(self.ranges["position_y"]),
        )


class InfiniteDSpritesNoImages(InfiniteDSprites):
    """Only return the latents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Generate an infinite stream of latent vectors.
        Note: We set shape to None and only return shape_id."""
        while True:
            if self.shapes is not None and self.current_shape_index >= len(self.shapes):
                return
            for color, scale, orientation, position_x, position_y in product(
                *self.ranges.values()
            ):
                if self.dataset_size is not None and self.counter >= self.dataset_size:
                    return
                self.counter += 1
                color = np.array(colors.to_rgb(color))
                yield Latents(
                    color,
                    None,
                    self.current_shape_id,
                    scale,
                    orientation,
                    position_x,
                    position_y,
                )
            self.current_shape_index += 1


class ContinualDSpritesMap(Dataset):
    """Map-style (finite) continual learning dsprites dataset."""

    def __init__(self, *args, **kwargs):
        self.dataset = InfiniteDSpritesNoImages(*args, **kwargs)
        assert (
            self.dataset.dataset_size is not None or self.dataset.shapes is not None
        ), "Dataset size must be finite. Please set dataset_size or pass a list of shapes."
        self.data = list(self.dataset)
        self.y_transform = kwargs.get("y_transform", lambda y: y)
        self.x_transform = kwargs.get("x_transform", lambda x: x)

    @property
    def targets(self):
        return [factors.shape_id for factors in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_index = self.data[index].shape_id
        if self.dataset.shape_ids is not None:
            shape_index = self.dataset.shape_ids.index(shape_index)
        shape = self.dataset.shapes[shape_index]
        factors = self.data[index]._replace(shape=shape)
        img = self.x_transform(self.dataset.draw(factors))
        factors = self.y_transform(factors)
        return img, factors


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
                index = np.random.choice(len(self.shapes))
                self.current_shape_index = index
                shape = self.shapes[index]
            else:
                shape = self.generate_shape()
            latents = self.sample_latents()._replace(
                shape=shape, shape_id=self.current_shape_id
            )
            image = self.draw(latents)
            yield image, latents


class RandomDSpritesMap(Dataset):
    """Map-style (finite) random dsprites dataset."""

    def __init__(self, *args, **kwargs) -> None:
        self.dataset = RandomDSprites(*args, **kwargs)
        assert (
            self.dataset.dataset_size is not None
        ), "Dataset size must be finite. Please set dataset_size."
        self.data, self.latents = zip(*list(self.dataset))
        self.data = list(self.data)
        self.latents = list(self.latents)

    @property
    def targets(self):
        return [latents.shape_id for latents in self.latents]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.latents[index]


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
        self.canvas_size = self.img_size // 2
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
            border_width = self.canvas_size // 128 or 1
            mid = self.canvas_size // 2
            grid[:, mid - border_width : mid + border_width, :] = 1.0
            grid[:, :, mid - border_width : mid + border_width] = 1.0

            yield grid
