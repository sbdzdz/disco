"""Visualize the map style dataset."""
from pathlib import Path

import numpy as np
import torch

from codis.data import ContinualDSprites, InfiniteDSprites
from codis.visualization import draw_batch


def main():
    """Visualize the map style dataset."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset = InfiniteDSprites()
    shapes = [dataset.generate_shape() for _ in range(4)]

    dataset = ContinualDSprites(
        scale_range=np.array([0.5, 1.5]),
        orientation_range=np.array([0.0, np.pi]),
        position_x_range=np.array([0.5]),
        position_y_range=np.array([0.5]),
        shapes=shapes,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    for imgs, _ in dataloader:
        print(f"Batch shape: {imgs.shape}")
        draw_batch(
            imgs,
            path=repo_root / "img/map_dataset.png",
            show=True,
        )


if __name__ == "__main__":
    main()
