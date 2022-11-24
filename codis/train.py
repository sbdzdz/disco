"""Training script."""
# import hydra
import argparse
from pathlib import Path
import numpy as np

import torch
from codis.utils.visualization import show_images_grid, show_density
from codis.data.dsprites import DSpritesDataset


def train(args):
    """Train the model."""
    dataset = DSpritesDataset(args.dsprites_path)
    for element in dataset:
        show_images_grid(element)
        break


# @hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
