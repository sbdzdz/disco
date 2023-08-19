"""Visualize a random batch of images from the dataset."""
from pathlib import Path

from torch.utils.data import DataLoader

from codis.data import RandomDSprites
from codis.visualization import draw_batch
import numpy as np


def sample_and_visualize():
    """Sample and visualize a random batch of images from the dataset."""
    dataset = RandomDSprites(img_size=256, scale_range=np.array([1, 2]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    imgs, _ = next(iter(dataloader))
    draw_batch(imgs, path=Path("img/random_dataset.png"))


def _main():
    sample_and_visualize()


if __name__ == "__main__":
    _main()
