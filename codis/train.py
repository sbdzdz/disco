"""Training script."""
import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from codis.data import InfiniteDSprites
from codis.lightning_modules import CodisModel, LightningBetaVAE, LightningMLP


def train(args):
    """Train the model."""
    print(f"Cuda available {torch.cuda.is_available()}")
    wandb_logger = WandbLogger(
        project="codis", save_dir=args.wandb_dir, group=args.wandb_group
    )
    wandb_logger.experiment.config.update(args)
    config = wandb.config

    train_set = InfiniteDSprites()
    val_set = InfiniteDSprites()
    test_set = InfiniteDSprites()

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
    )
    test_set = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
    )

    backbone = LightningBetaVAE(
        img_size=train_set.img_size, latent_dim=config.latent_dim, beta=config.beta
    )
    trainer = pl.Trainer(
        default_root_dir=args.wandb_dir,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
    )

    for _ in range(10):
        trainer.fit(backbone, train_loader, val_loader)
        regressor = LightningMLP(
            dims=[config.latent_dim, 64, 64, train_set.num_latents]
        )
        model = CodisModel(backbone, regressor)
        trainer.fit(model, val_loader)
        trainer.test(model, val_loader)

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
