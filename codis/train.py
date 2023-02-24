"""Training script."""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from codis.data.dsprites import DSpritesDataset
from codis.visualization import draw_batch_grid


def train(args):
    """Train the model."""
    dataset = DSpritesDataset(args.dsprites_path)
    train_loader = DataLoader(dataset, batch_size=16)
    batch = next(iter(train_loader))
    draw_batch_grid(batch)


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
