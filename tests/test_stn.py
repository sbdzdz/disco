"""Tests for the spatial transformer network."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from codis.data import Latents, RandomDSprites
from codis.lightning.modules import SpatialTransformer


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parameter_conversion(seed):
    """Test converting parameters to a transformation matrix."""
    np.random.seed(seed)
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
    diffs = np.mean(np.abs(transformed_images - canonical_images))
    assert diffs < 0.003


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
