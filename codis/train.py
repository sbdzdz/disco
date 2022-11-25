"""Training script."""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from codis.data.dsprites import DSpritesDataset
from codis.utils.visualization import show_images_grid


def train(args):
    """Train the model."""
    dataset = DSpritesDataset(args.dsprites_path)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for element in train_loader:
        print(element.shape)
        show_images_grid(element)
        break


# @hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
