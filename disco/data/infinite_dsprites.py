"""Class definitions for the infinite dSprites dataset."""
from collections import namedtuple
from itertools import product
from copy import copy, deepcopy

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from scipy.interpolate import splev, splprep
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, IterableDataset
from tensordict import TensorDict
from disco.data.polygon_functions import polygon_vectorized

# _fill_poly_vectorized = np.vectorize(lambda img,pts,color: cv2.fillPoly(img=img, pts=pts, color=color,
#                                                                         lineType=cv2.LINE_AA))
_bounding_rect_vectorized = torch.func.vmap(lambda pts: cv2.boundingRect(pts), in_dims=(0,), out_dims=0)

_rectangle_vectorized = torch.func.vmap(lambda img, x, y, w, h: cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w, y+h),
                                                                              color=(0, 255, 0), thickness=2),
                                        in_dims=(0,0,0,0,0), out_dims=0)
_circle_vectorized = torch.func.vmap(lambda img, center: cv2.circle(img=img, center=tuple(center), radius=5, color=(255, 0, 0),
                                                                    thickness=-1), in_dims=(0,0), out_dims=0)


BaseLatents = namedtuple(
    "BaseLatents", "color shape shape_id scale orientation position_x, position_y"
)


class Latents(BaseLatents):
    """Latent variables defining a single image."""

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, device, **kwargs):
        """Move the latents to a device."""
        return Latents(
            color=self.color.to(device, **kwargs),
            shape=self.shape.to(device, **kwargs),
            shape_id=self.shape_id.to(device, **kwargs),
            scale=self.scale.to(device, **kwargs),
            orientation=self.orientation.to(device, **kwargs),
            position_x=self.position_x.to(device, **kwargs),
            position_y=self.position_y.to(device, **kwargs),
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
                the number of images generated. ? Set shapes to None if you set dataset_size.
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
        self.range_means = {
            "color": np.array(colors.to_rgb(color_range[0] if len(color_range)==1 else "white")),
            "scale": np.mean(scale_range),
            "orientation": np.mean(orientation_range),
            "position_x": np.mean(position_x_range),
            "position_y": np.mean(position_y_range),
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
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3)).astype(np.int32)
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




    ###############


    ##################

    def draw_tensordict(self, latents:TensorDict, channels_first=True, debug=False):
        ''' attempt at a parallelized version of the above.'''
        b = latents.batch_size
        assert len(b) == 1
        b = b[0]
        canvas = torch.zeros((b, self.canvas_size, self.canvas_size, 3), dtype=torch.int32)
        canvas[:, :, :, :] = torch.tensor(self.background_color, device=latents["shape"].device)
        shapes = latents["shape"]
        shapes = self.apply_scale_multi(shapes, latents["scale"])
        shapes = self.apply_orientation_multi(shapes, latents["orientation"])
        shapes = self.apply_position_multi(shapes, latents["position_x"], latents["position_y"])
        color = (255 * latents["color"]).to(dtype=int)

        self.draw_shape_multiple(shapes, canvas, color)
        if self.orientation_marker:
            self.draw_orientation_marker_multi(canvas, latents)
        if debug:
            self.add_debug_info_multi(shapes, canvas)
        if self.grayscale:
            canvas = torch.mean(canvas, axis=3, keepdims=True)
        if channels_first:
            canvas = torch.transpose(canvas, (3, 1, 2)) #(2, 0, 1))
        canvas = canvas.astype(np.float32) / 255.0
        return canvas

    def apply_scale(self, shape: npt.NDArray|torch.Tensor, scale: float):
        """Apply a scale to a shape."""
        height = self.canvas_size
        return self.scale_factor * height * scale * shape

    def apply_scale_multi(self, shapes: torch.Tensor, scales: torch.Tensor):
        """Apply a scale to a shape."""
        height = self.canvas_size
        l = len(scales.shape)
        if l > len(shapes.shape):
            raise ValueError(f"Mismatching shapes of inputs: scales.shape = {scales.shape}, "
                             f"shapes.shape = {shapes.shape}")
        # add_dims = [1]*(len(shapes.shape)-l)
        n_add = len(shapes.shape) - l
        scales = scales[(..., ) + (None, ) * n_add]  # https://github.com/pytorch/pytorch/issues/9410
        scaled_shapes = scales * shapes
        # scaled_shapes = torch.einsum("b...,b...->b...", scales, shapes )
        return self.scale_factor * height * scaled_shapes


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


    @staticmethod
    def apply_orientation_multi(shapes: torch.Tensor, orientations: torch.Tensor):
        """Apply an orientation to a shape.
        Args:
            shape: An array of shape (b, 2, num_points).
            orientation: The orientation in radians.
        Returns:
            The rotated shape.
        """
        b = len(orientations)
        device = orientations.device
        assert len(orientations.squeeze().shape) <= 1, f"{orientations.shape} - expected 1d vector of orientation angles"
        rotation_matrix = torch.zeros(b, 2, 2, device=device, dtype=orientations.dtype)
        rotation_matrix[:,0,0]= torch.cos(orientations)
        rotation_matrix[:,0,1]= -torch.sin(orientations)
        rotation_matrix[:,1,0]= torch.sin(orientations)
        rotation_matrix[:,1,1]= torch.cos(orientations)

        # orientation = orientations.to("cpu").detach()
        # rotation_matrix_np = np.array(
        #     [
        #         [np.cos(orientation[5]), -np.sin(orientation[5])],
        #         [np.sin(orientation[5]), np.cos(orientation[5])],
        #     ]
        # )
        # assert np.allclose(rotation_matrix_np, rotation_matrix[5,:,:].to("cpu").detach())

        retval= torch.einsum("bik,bij->bkj", rotation_matrix, shapes) # rotation_matrix @ shape
        assert retval.shape[1:] == shapes.shape[1:]
        return retval

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


    def apply_position_multi(self, shapes: torch.Tensor, positions_x: torch.Tensor, positions_y: torch.Tensor):
        """Apply a position to a shape.
        Args:
            shapes: A tensor of shape (batchsize, 2, num_points).
            positions_x: The x position of each shape.
            positions_y: The y position of each shape.
        Returns:
            An array of shape (2, num_points).
        """
        b = len(positions_x)
        height, width = self.canvas_size, self.canvas_size
        positions = torch.stack(
            [
                0.25 * width + 0.5 * width * positions_x,
                0.25 * height + 0.5 * height * positions_y,
            ], dim=1)[:,:,None]

        return shapes + positions

    @staticmethod
    def draw_shape(shape, canvas, color):
        """Draw a shape on a canvas."""
        shape = shape.T.astype(np.int32)
        cv2.fillPoly(img=canvas, pts=[shape], color=color, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_shape_multiple(shape:torch.Tensor, canvas:torch.Tensor, color:torch.Tensor):
        """Draw a shape on a canvas."""
        shape = shape.transpose(1,2).type(torch.int32)#[:, None, ...]
        assert shape.shape[2] == 2
        assert canvas.shape[3] == 3, f"Expected 3 channels in last dimension, found: {canvas.shape}"
        b, h, w, c = canvas.shape
        mask = polygon_vectorized(shape[:,:,0], shape[:,:,1], shape=(h, w), return_mask=True)
        canvas[mask[..., None]] = color

        # _fill_poly_vectorized(canvas.to("cpu").detach().numpy(),
        #                       shape.to("cpu").detach().numpy(),
        #                       color.to("cpu").detach().numpy())

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

        # rotate shape pixel coordinates
        shape_pixels = np.argwhere(np.any(canvas != self.background_color, axis=2))
        x, _ = rotate_point(shape_pixels[:, 0], shape_pixels[:, 1])

        # select the right half of the shape
        right_half = shape_pixels[x > x0]

        canvas[right_half[:, 0], right_half[:, 1]] = self.orientation_marker_color


    def draw_orientation_marker_multi(self, canvas, latents:TensorDict):
        """Mark the right half of the shape."""
        theta = latents["orientation"] - torch.pi / 2
        device = theta.device
        b = len(theta)
        center = torch.zeros((b, 2, 1), device=device)
        yx = self.apply_position_multi(center, latents["position_x"], latents["position_y"])
        y0 = yx[:, 0]
        x0 = yx[:, 1]
        def rotate_points(x, y):
            """Rotate the coordinate system by -theta around the center of the shape."""
            x_prime = (x - x0) * np.cos(-theta) + (y - y0) * np.sin(-theta) + x0
            y_prime = -(x - x0) * np.sin(-theta) + (y - y0) * np.cos(-theta) + y0
            return x_prime, y_prime

        # rotate shape pixel coordinates
        shape_pixels = torch.argwhere(torch.any(canvas != self.background_color, axis=3))
        x, _ = rotate_points(shape_pixels[:, 1], shape_pixels[:, 2])

        # select the right half of the shape
        right_half = shape_pixels[x > x0]

        canvas[right_half[:, 0], right_half[:, 1], right_half[:, 2]] = self.orientation_marker_color

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


    def add_debug_info_multi(self, shapes:torch.Tensor, canvas:torch.Tensor):
        """Add debug info to the canvas."""
        raise NotImplementedError("You can try, but the below is likely to fail, it uses torch.func.vmap.")
        shape_center = self.get_center_multi(canvas)
        x, y, w, h = _bounding_rect_vectorized(shapes.transpose(1,2).astype(np.int32))
        _rectangle_vectorized(img=canvas, x=x, y=y, w=w, h=h)
        _circle_vectorized(img=canvas, center=shape_center[...,::-1].astype(np.int32))
        _circle_vectorized(img=canvas, center=torch.tensor((self.canvas_size // 2, self.canvas_size // 2))[None,:])


    @staticmethod
    def get_center(canvas):
        """Get the center of the shape."""
        foreground_pixels = np.argwhere(np.any(canvas != [0, 0, 0], axis=2))
        return np.mean(foreground_pixels, axis=0)

    @staticmethod
    def get_center_multi(canvas):
        """Get the center of the shape."""
        foreground_pixels = torch.argwhere(np.any(canvas != [0, 0, 0], axis=4))
        return torch.mean(foreground_pixels, axis=2)

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

    # def sample_latents_only(self):
    #     """Sample a random set of latents, _excluding_ the shape.
    #         Does not return Latents objects, but a dict of numpy arrays."""
    #     return Latents(
    #         color=np.array(colors.to_rgb(np.random.choice(self.ranges["color"]))),
    #         shape=None,
    #         shape_id=None,
    #         scale=np.random.choice(self.ranges["scale"]),
    #         orientation=np.random.choice(self.ranges["orientation"]),
    #         position_x=np.random.choice(self.ranges["position_x"]),
    #         position_y=np.random.choice(self.ranges["position_y"]),
    #     )

    def sample_controlled_from_default(self, shape, latent:str=None):
        """Sample a random value only for latent :param latent.
            latent can be one of: "color", "scale", "orientation", "position_x", "position_y".
            latent=None also works, then we return a "default" shape.
        """
        param_dict = {k:v for k, v in self.range_means.items()}
        param_dict["shape"] = shape
        param_dict["shape_id"] = -1
        if latent == "color":
            param_dict[latent] = np.array(colors.to_rgb(np.random.choice(self.ranges["color"])))
        elif latent is not None:
            param_dict[latent] = np.random.choice(self.ranges[latent])
        return Latents(**param_dict )

    # Todo: Remove the following. It's incomplete and has been moved out into group_orbit_cl.
    # ### Next: Functions that operate on batches/numpy arrays of latents, treating shapes and other latents separately.
    # def transform_controlled(self, shapes,
    #                          latents:dict[str, torch.Tensor],
    #                          latent_indicators:torch.Tensor,
    #                          latent_variable_differences:torch.Tensor|None=None,
    #                          from_valid_range=True,
    #                          latent_variables_in_order:list[str]|None=None):
    #     """
    #      TODO. We want to randomly apply different groups and different coefficients to passed latents, and
    #            return the corresponding images + latents (+ coefficients).
    #     Vectorized. latents should be: tensor/array of shape: (batch_size, num_latents).
    #     If from_valid_range=True, except allows multiple latents to vary.
    #     If from_valid_range=False, add/subtract a random value from the valid range - in that way we'll end
    #      up outside of the valid range not too rarely.
    #
    #      Sample a random value only for latents :param latents_to_vary (list of strings).
    #         latents_to_vary can each be one of: "color", "scale", "orientation", "position_x", "position_y".
    #
    #     Parameters:
    #     ----------
    #         :param latent_indicators: tensor of shape (batch_size, k), where k is the number of transforms to apply.
    #                                     Should contain integers in range(0, [5 or len(latent_variables_in_order)]).
    #         :param latent_variable_differences: None or tensor of shape (batch_size, k)
    #         :param from_valid_range: If True, sample from the valid range of each latent variable and return
    #                                 the difference to the previous latent variable value.
    #                                 If False, add or subtract a random value from the valid range, leading to
    #                                 potentially out-of-range values.
    #                                 ! For "scale" variable, need to apply an exponential to the parameter before
    #                                     transforming. (so that 0 corresponds to no transformation).
    #         :param latent_variables_in_order: list of strings, to select in what order the indices refer
    #                                             to the latent variables.
    #
    #     :return
    #         images (optional?); images of the transformed shapes.
    #         latents (lat_dict): the new latents after transformation. Again a dict, with each key mapping to a batch of
    #                  the respective latent variable values.
    #         latent_variable_differences: The "group transform parameters". Should be equal to the input variable
    #                                     of same name if it was not None.
    #
    #     """
    #     b = len(shapes)
    #     for k, v in latents.items():
    #         assert b == len(v), f"Batch size {b} does not match batch size of latents {k} {len(v)}"
    #     if latent_variables_in_order is None:
    #         latent_variables_in_order = ["position_x", "orientation", "position_y", "scale", "color"]
    #     assert torch.max(latent_indicators) < len(latent_variables_in_order), \
    #         f"Latent indicator {torch.max(latent_indicators)} exceeds the number of latent variables {len(latent_variables_in_order)}"
    #
    #     # Want to: for each "group"/indicator of a latent variable, randomly sample how much to transform
    #     #           (if latent_variable_differences is None). Then, modify the parameter by that value.
    #     # If latent_variable_differences is None, and from_valid_range=True, just resample randomly.
    #     # If a group was passed twice, allocate the total random value to the first transform.
    #
    #     if latent_variable_differences is None:
    #         color = np.array(colors.to_rgb(np.random.choice(self.ranges["color"], size=len(latents))))
    #         scale = np.random.choice(self.ranges["scale"], size=len(latents))
    #         orientation = np.random.choice(self.ranges["orientation"], size=len(latents))
    #         position_x = np.random.choice(self.ranges["position_x"], size=len(latents))
    #         position_y = np.random.choice(self.ranges["position_y"], size=len(latents))
    #         lat_dict = {"color": color, "scale": scale, "orientation": orientation,
    #                     "position_x": position_x, "position_y": position_y}
    #         lat_diffs_dict = {"color": torch.zeros_like(color), "scale": torch.zeros_like(scale),
    #                             "orientation": torch.zeros_like(orientation),
    #                             "position_x": torch.zeros_like(position_x), "position_y": torch.zeros_like(position_y)}
    #         # To return the coefficients in order:
    #         lat_diffs_tensor = torch.zeros_like(latent_indicators, dtype=torch.float32)
    #         # if from_valid_range:
    #         # lat_dict_diffs = deepcopy(lat_dict)
    #         # We can use indexing. Transform all latent variables, but only keep the change where the indicator for that group was present.
    #         for i, key in enumerate(latent_variables_in_order):
    #             mask = (latent_indicators == i)
    #             # Where this cumsum is >= 2 AND mask is 1, we have a duplicate that is not the first in that row
    #             cumsum = torch.cumsum(mask, dim=1)
    #             duplicate_ids = (cumsum > 1) and (mask)
    #             ids = torch.any(latent_indicators == i, dim=1)
    #             # preserve original latents (replace the new ones) where group should not be applied:
    #             lat_dict[key][~ids] = latents[key][~ids]
    #
    #             if from_valid_range:
    #                 # otherwise keep new latents and insert the differences into lat_dict_diffs
    #                 lat_diffs_dict[key][ids] = lat_dict[key][ids] - latents[key][ids]
    #                 lat_diffs_dict[key][duplicate_ids] = 0 # we only apply the transform once, they are commutative anyway
    #                 lat_diffs_tensor[latent_indicators == i] = lat_diffs_dict[key][ids]
    #                 assert torch.all(lat_diffs_dict[key][~ids] == 0)
    #             else:
    #                 lat_diffs_dict[key][ids] = lat_dict[key][ids]
    #                 lat_diffs_tensor[latent_indicators == i] = lat_dict[key][ids]
    #                 lat_diffs_tensor[duplicate_ids] = 0
    #
    #                 lat_dict[key][ids] += latents[key][ids]
    #                 # otherwise keep new latents and insert the differences into lat_dict_diffs
    #
    #                 assert torch.all(lat_diffs_dict[key][~ids] == 0)
    #
    #
    #
    #
    #         # postprocess: drop duplicate transforms, if any latent_indicator occurs twice, set the second
    #         #               diff to 0:
    #         for i, key in enumerate(latent_variables_in_order):
    #             ids = torch.any(latent_indicators == i, dim=1)
    #             if torch.sum(ids) > 1:
    #                 pass
    #             return imgs,
    #
    #
    #
    #
    #
    #     if from_valid_range:
    #         # return the difference between original and new latents.
    #
    #         return Latents(
    #             color=np.array(colors.to_rgb(np.random.choice(self.ranges["color"]))),
    #             shape=shape,
    #             shape_id=None,
    #             scale=np.random.choice(self.ranges["scale"]),
    #             orientation=np.random.choice(self.ranges["orientation"]),
    #             position_x=np.random.choice(self.ranges["position_x"]),
    #             position_y=np.random.choice(self.ranges["position_y"]),
    #         )
    #
    #
    #     param_dict = latents
    #     if randomize_latent == "color":
    #         param_dict[randomize_latent] = np.array(colors.to_rgb(np.random.choice(self.ranges["color"])))
    #     elif randomize_latent is not None:
    #         param_dict[randomize_latent] = np.random.choice(self.ranges[latent])
    #     return Latents(**param_dict )


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


class RandomDSpritesShapes(InfiniteDSprites):
    """Returns shapes only.
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
            yield shape


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



class RandomDSpritesMapCustomized(Dataset):
    """Stores only the shapes internally; this allows to generate new transformations
       on each iteration. Restrict the latents to vary only in one dimension, and be fixed otherwise.
       """

    def __init__(self, sample_random_latents=False,  *args, **kwargs) -> None:
        ''':param sample_random_latents: If True, sample randomly.
                                         If False, sample a prototype, with fixed default latents.
                                         Todo: Can drop that parameter? Can achieve same result by defining the
                                                randomness ranges to one element.
                                         '''
        # This class itself is not an iterable dataset; we use the iterable dataset RandomDSpritesShapes and
        #   "iterate" over it once to generate all shapes.
        self.dataset = RandomDSpritesShapes(*args, **kwargs)
        self.ranges = deepcopy(self.dataset.ranges)
        self.range_means = deepcopy(self.dataset.range_means)
        assert (
            self.dataset.dataset_size is not None
        ), "Dataset size must be finite. Please set dataset_size."
        self.dataset_size = self.dataset.dataset_size

        self.shapes = list(self.dataset)
        self.allowed_latents = ["color", "scale", "orientation", "position_x", "position_y"] # for reference
        self.sample_random_latents = sample_random_latents
        # del self.dataset # Try this, see if it works.
        #  \\ no won't work. InfiniteDSprites contains most of the logic of the dataset.

    def __len__(self):
        return self.dataset.dataset_size
        # return len(self.shapes)

    def __getitem__(self, index):
        # return a random transform of the shape at :param index .
        shape = self.shapes[index]
        if self.sample_random_latents:
            latents = self.sample_latents_only()
            latents = latents._replace(shape=shape)
        else:
            latents = self.dataset.sample_controlled_from_default(shape)
        image = self.dataset.draw(latents)
        # if self.return_numpy:
        #     # Todo
        #     return image, latents, shape
        # else:
        return image, latents

    def transform_sample(self, latents:Latents, ):
        ''''''

    # def __getitems__(self, indices):
    #   todo/can-do, for further optimization.
    #     pass

    def sample_latents_only(self):
        """Sample a random set of latents, _excluding_ the shape.
            Does not return Latents objects, but a dict of numpy arrays."""
        return Latents(
            color=np.array(colors.to_rgb(np.random.choice(self.ranges["color"]))),
            shape=None,
            shape_id=-1,
            scale=np.random.choice(self.ranges["scale"]),
            orientation=np.random.choice(self.ranges["orientation"]),
            position_x=np.random.choice(self.ranges["position_x"]),
            position_y=np.random.choice(self.ranges["position_y"]),
        )

    def sample_controlled_from_default(self, shape, latent:str=None):
        """For compatibility with earlier settings. Probably not used any more.
            Sample a random value only for latent :param latent.
            latent can be one of: "color", "scale", "orientation", "position_x", "position_y".
            latent=None also works, then we return a "default" shape.
        """
        param_dict = {k:v for k, v in self.range_means.items()}
        param_dict["shape"] = shape
        param_dict["shape_id"] = -1
        if latent == "color":
            param_dict[latent] = np.array(colors.to_rgb(np.random.choice(self.ranges["color"])))
        elif latent is not None:
            param_dict[latent] = np.random.choice(self.ranges[latent])
        return Latents(**param_dict )

    # def _getitem_numpy(self, index):
    #     '''Todo: Maybe replace __getitem__ by this function, or add a paramter to switch between the two.'''
    #     shape = self.shapes[index]
    #     assert self.sample_random_latents, "Not using this doesnt perform well, phasing it out"
    #     latents = self.dataset.sample_latents_only

    # def sample_controlled(self, latents, randomize_latent=None):
    #     '''
    #         Todo: This is not used; remove it.
    #     For drawing a predefined sample (given by latents; if :param randomize_latent==None),
    #         or randomly sampling only a single attribute but keeping the others fixed.
    #         '''
    #     if randomize_latent is not None:
    #         assert randomize_latent in self.allowed_latents, f"Can only vary one of latents {self.allowed_latents}, " \
    #                                                          f"but found {randomize_latent}"
    #         latents = self.dataset.sample_controlled(latents, randomize_latent)
    #     image = self.dataset.draw(latents)
    #     return image, latents





class InfiniteDSpritesTriplets(InfiniteDSprites):
    """Infinite dataset of triplets of images.
    For details see the composition task proposed by Montero et al. (2020).
    Note: https://openreview.net/pdf?id=qbH974jKUVy, see fig. 5
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
