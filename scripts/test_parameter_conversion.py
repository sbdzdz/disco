"""Test converting between a theta transformation matrix and a ground truth factor representation."""

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from codis.data.infinite_dsprites import InfiniteDSpritesRandom
from codis.visualization import draw_batch_and_reconstructions


def main(args):
    """Compare the ground truth factors to the theta transformation matrix."""
    dataset = InfiniteDSpritesRandom(
        img_size=2048, scale_range=np.linspace(0.6, 1.0, 10)
    )
    factors = [dataset.sample_latents() for _ in range(args.batch_size)]
    matrices = [
        convert_parameters_to_matrix(f.scale, f.orientation, f.position_x, f.position_y)
        for f in factors
    ]
    for i, scale in enumerate(f.scale for f in factors):
        print(i, scale)

    exemplars = [
        f._replace(
            scale=1.0,
            orientation=0.0,
            position_x=0.5,
            position_y=0.5,
        )
        for f in factors
    ]

    exemplar_images = [torch.tensor(dataset.draw(f)) for f in exemplars]
    images = [torch.tensor(dataset.draw(f)) for f in factors]
    transformed_images = np.array(
        [transform(i, m).numpy() for i, m in zip(images, matrices)]
    )
    images = np.array([img.numpy() for img in images])
    exemplar_images = np.array([img.numpy() for img in exemplar_images])
    diffs = np.abs(transformed_images - exemplar_images)

    draw_batch_and_reconstructions(
        images, exemplar_images, transformed_images, diffs, show=True
    )


def transform(img, matrix):
    """Apply the transformation matrix to the image."""
    grid = F.affine_grid(
        matrix[:2].unsqueeze(0).to(img.device),
        img.unsqueeze(0).size(),
        align_corners=False,
    )
    return F.grid_sample(img.unsqueeze(0), grid.float(), align_corners=False).squeeze(0)


def convert_parameters_to_matrix(scale, orientation, position_x, position_y):
    """Convert the ground truth factors to a transformation matrix."""
    transform_matrix = np.identity(3)

    # reverse scale
    scale = 0.8 * scale + 0.2
    transform_matrix = np.dot(
        np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]), transform_matrix
    )

    # reverse orientation
    transform_matrix = np.dot(
        np.array(
            [
                [np.cos(-orientation), -np.sin(-orientation), 0],
                [np.sin(-orientation), np.cos(-orientation), 0],
                [0, 0, 1],
            ]
        ),
        transform_matrix,
    )

    # move to 0.5, 0.5
    transform_matrix = np.dot(
        np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]),
        transform_matrix,
    )

    # move from position_x and position_y to 0, 0
    transform_matrix = np.dot(
        np.array([[1, 0, position_x], [0, 1, position_y], [0, 0, 1]]),
        transform_matrix,
    )

    return torch.tensor(transform_matrix)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=25)
    args = parser.parse_args()
    main(args)
