"""Compare PIL to OpenCV for rendering shapes."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from disco.data import InfiniteDSprites
from disco.visualization import draw_batch_and_reconstructions


def compare(args):
    """Compare PIL to OpenCV for rendering shapes.""" ""
    dataset = InfiniteDSprites(img_size=args.img_size)
    shapes = [generate_shape(dataset) for _ in range(args.num_shapes)]

    pil_imgs = np.array([render_opencv(shape, args) for shape in shapes])
    open_imgs = np.array([render_pillow(shape, args) for shape in shapes])
    draw_batch_and_reconstructions(pil_imgs, open_imgs, path=Path("img/pil_opencv.png"))


def generate_shape(dataset):
    """Generate a random shape."""
    shape = dataset.generate_shape()
    shape = dataset.apply_scale(shape, 2)
    shape = dataset.apply_orientation(shape, 0.0)
    shape = dataset.apply_position(shape, 0.5, 0.5)
    return shape


def render_pillow(shape, args):
    """Render a shape using Pillow."""
    canvas_size = (args.img_size, args.img_size)
    background_color = (0, 0, 0)
    canvas = Image.new("RGB", canvas_size, background_color)

    # Create a draw object
    draw = ImageDraw.Draw(canvas)
    # convert shape from a (2, N) array to a list of tuples
    shape = list(zip(shape[0], shape[1]))

    polygon_color = (250, 250, 250)
    draw.polygon(shape, outline=polygon_color, fill=polygon_color, width=2)

    canvas = np.array(canvas).astype(np.float32) / 255.0
    print(canvas.min(), canvas.max(), canvas.mean())
    return canvas.transpose((2, 0, 1))


def render_opencv(shape, args):
    """Render a shape using OpenCV."""
    canvas = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
    color = (250, 250, 250)

    shape = shape.T.astype(np.int32)
    cv2.fillPoly(canvas, [shape], color)
    cv2.polylines(canvas, [shape], True, color, thickness=2)
    canvas = canvas.astype(np.float32) / 255.0
    print(canvas.min(), canvas.max(), canvas.mean())
    return canvas.transpose((2, 0, 1))


def _main():
    parser = ArgumentParser()
    parser.add_argument("--num_shapes", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    _main()
