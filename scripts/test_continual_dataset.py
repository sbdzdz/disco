"""Test restricting the rotation range in the continual map-style dataset."""
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader

from codis.data import ContinualDSpritesMap, InfiniteDSprites
from codis.visualization import draw_batch


def sample_and_draw(args):
    """Sample a batch from the dataset and visualize it."""
    np.random.seed(0)
    shape = InfiniteDSprites().generate_shape()
    dataset = ContinualDSpritesMap(
        shapes=[shape],
        orientation_range=np.linspace(0, 0.5 * np.pi, args.resolution),
        scale_range=np.linspace(0.5, 1.5, 5),
        position_x_range=np.linspace(0, 1, 5),
        position_y_range=np.linspace(0, 1, 5),
    )
    dataloader = DataLoader(dataset, batch_size=49, shuffle=True)
    x, _ = next(iter(dataloader))
    draw_batch(x, show=True, n_max=49)


def _main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=49)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    sample_and_draw(args)


if __name__ == "__main__":
    _main()
