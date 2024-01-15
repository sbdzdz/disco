"""Visualize the map style dataset."""
from pathlib import Path

import numpy as np
import torch

from disco.data import ContinualDSpritesMap, InfiniteDSprites
from disco.visualization import draw_batch


def main():
    """Visualize the map style dataset."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset = InfiniteDSprites()
    shapes = [dataset.generate_shape() for _ in range(4)]

    dataset = ContinualDSpritesMap(
        scale_range=np.array([0.4, 0.9]),
        orientation_range=np.array([0.0, np.pi]),
        position_x_range=np.array([0, 1]),
        position_y_range=np.array([0, 1]),
        shapes=shapes,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for imgs, _ in dataloader:
        draw_batch(
            imgs,
            num_images=imgs.shape[0],
            path=repo_root / "img/map_dataset.png",
        )


if __name__ == "__main__":
    main()
