"""Test restricting the rotation range in the continual map-style dataset."""
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader

from disco.data import ContinualDSpritesMap, InfiniteDSprites
from disco.visualization import draw_batch


def sample_and_draw(args):
    """Sample a batch from the dataset and visualize it."""
    np.random.seed(0)
    shapes = [InfiniteDSprites().generate_shape() for _ in range(args.num_shapes)]
    dataset = ContinualDSpritesMap(
        shapes=shapes,
        orientation_range=np.linspace(0, 0.5 * np.pi, args.resolution),
        scale_range=np.linspace(0.5, 1, args.resolution),
        position_x_range=np.linspace(0, 1, args.resolution),
        position_y_range=np.linspace(0, 1, args.resolution),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    x, _ = next(iter(dataloader))
    draw_batch(x, show=True, save=True, num_images=args.batch_size)


def _main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_shapes", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=2)
    args = parser.parse_args()
    sample_and_draw(args)


if __name__ == "__main__":
    _main()
