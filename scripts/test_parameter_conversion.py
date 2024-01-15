"""Test converting between a theta transformation matrix and a ground truth factor representation."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from disco.data.infinite_dsprites import RandomDSprites, Latents
from disco.lightning.modules import SpatialTransformer
from disco.visualization import draw_batch_and_reconstructions


def main():
    """Compare the original image with the transformed image.
    The transformation maps from the input to the canonical representation.
    """
    batch_size = 16
    dataset = RandomDSprites()
    transformer = SpatialTransformer()
    dataloader = DataLoader(dataset, batch_size, num_workers=0)
    images, factors = next(iter(dataloader))
    matrices = transformer.convert_parameters_to_matrix(factors)

    factors = Latents(
        **{k: v.numpy() for k, v in factors._asdict().items()}
    )  # convert torch tensors to numpy arrays
    factors = [
        Latents(*items) for items in zip(*factors)
    ]  # convert a batched namedtuple to a list of namedtuples

    canonical_factors = [
        f._replace(
            scale=1.0,
            orientation=0.0,
            position_x=0.5,
            position_y=0.5,
        )
        for f in factors
    ]

    canonical_images = [torch.tensor(dataset.draw(f)) for f in canonical_factors]
    transformed_images = [transform(i, m) for i, m in zip(images, matrices)]

    images = np.array([img.numpy() for img in images])
    canonical_images = np.array([img.numpy() for img in canonical_images])
    transformed_images = np.array([img.numpy() for img in transformed_images])
    diffs = np.abs(transformed_images - canonical_images)

    draw_batch_and_reconstructions(
        images, canonical_images, transformed_images, diffs, show=True
    )


def transform(img, matrix):
    """Apply the transformation matrix to the image."""
    grid = F.affine_grid(
        matrix[:2].unsqueeze(0).to(img.device),
        img.unsqueeze(0).size(),
        align_corners=False,
    )
    return F.grid_sample(
        img.unsqueeze(0), grid.float(), align_corners=False, padding_mode="border"
    ).squeeze(0)


if __name__ == "__main__":
    main()
