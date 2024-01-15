"""Visualize a random batch of images from the dataset."""
from pathlib import Path

from torch.utils.data import DataLoader

from codis.data import RandomDSprites
from codis.visualization import draw_batch
import argparse


def _main(args):
    """Sample and visualize a random batch of images from the dataset."""
    if args.canonical:
        dataset = RandomDSprites(
            img_size=args.img_size,
            scale_range=[1.0],
            orientation_range=[0.0],
            position_x_range=[0.5],
            position_y_range=[0.5],
        )
    else:
        dataset = RandomDSprites(img_size=256)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    imgs, _ = next(iter(dataloader))
    draw_batch(imgs, path=Path("img/random_dataset.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--canonical", action="store_true")
    args = parser.parse_args()
    _main()
