"""Training script."""
import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch

import wandb
from codis.data import InfiniteDSprites
from codis.models import CodisModel, LightningBetaVAE, LightningMLP


def train(args):
    """Train the model."""
    print(f"Cuda available {torch.cuda.is_available()}")
    wandb.init(project="codis", group=args.wandb_group, dir=args.wandb_dir, config=args)
    config = wandb.config

    train_set = InfiniteDSprites()
    val_set = InfiniteDSprites()
    test_set = InfiniteDSprites()

    backbone = LightningBetaVAE(
        img_size=train_set.img_size, latent_dim=config.latent_dim, beta=config.beta
    )
    regressor = LightningMLP(dims=[config.latent_dim, 64, 64, train_set.num_latents])
    model = CodisModel(backbone, regressor)
    trainer = pl.Trainer(default_root_dir=args.wandb_dir, accelerator="auto", devices=1)

    for _ in range(10):
        trainer.fit(backbone, train_set, val_set)
        trainer.fit(model, val_set)
        trainer.test(model, test_set)

    wandb.finish()


def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="How often to log training progress. The logs will be averaged over this number of batches.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter.")
    parser.add_argument(
        "--eval_on", type=int, default=100, help="Number of batches to evaluate on."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="Dimensionality of the latent space."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--wandb_dir",
        type=Path,
        default=repo_root / "wandb",
        help="Wandb logging directory.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb group name. If not specified, a new group will be created.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
