"""Visualize the map style dataset."""
import torch
import numpy as np

from codis.data import ContinualDSprites, InfiniteDSprites
from codis.visualization import draw_batch


def main():
    """Visualize the map style dataset."""
    dataset = InfiniteDSprites()
    shapes = [dataset.generate_shape() for _ in range(8)]

    dataset = ContinualDSprites(
        scale_range=np.array([0.5, 1.5]),
        orientation_range=np.array([0.0]),
        position_x_range=np.array([0.5]),
        position_y_range=np.array([0.5]),
        shapes=shapes,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    for imgs, _ in dataloader:
        print(f"Batch shape: {imgs.shape}")
        draw_batch(imgs)


if __name__ == "__main__":
    main()
