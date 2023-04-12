"""Visualize the map style dataset."""
import torch
import numpy as np

from codis.data import ContinualDSprites
from codis.visualization import draw_batch_grid


def main():
    """Visualize the map style dataset."""
    # Load the dataset
    dataset = ContinualDSprites()
    shapes = [dataset.generate_shape() for _ in range(16)]
    dataset = ContinualDSprites(
        scale_range=np.array([0.5]),
        orientation_range=np.array([0.0]),
        position_x_range=np.array([0.5]),
        position_y_range=np.array([0.5]),
        shapes=shapes,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    for batch in dataloader:
        draw_batch_grid(batch)


if __name__ == "__main__":
    main()
