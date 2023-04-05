"""Training script."""
import argparse
from pathlib import Path

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

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

    train_loader, val_loader, test_loader = get_idsprites_loaders(args)

    backbone = LightningBetaVAE(
        img_size=args.img_size, latent_dim=config.latent_dim, beta=config.beta
    )
    trainer = Trainer(
        default_root_dir=args.wandb_dir,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        limit_train_batches=args.train_on,
        limit_val_batches=args.eval_on,
        log_every_n_steps=args.log_every_n_steps,
    )

    for _ in range(args.tasks):
        trainer.fit(backbone, train_loader, val_loader)
        regressor = LightningMLP(
            dims=[config.latent_dim, 64, 64, 7]
        )  # 7 is the number of stacked latent values
        model = CodisModel(backbone, regressor)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

    wandb.finish()


def train_vae(args):
    """Train the VAE on dSprites or idSprites."""
    print(f"Cuda available {torch.cuda.is_available()}")
    wandb_logger = WandbLogger(
        project="codis", save_dir=args.wandb_dir, group=args.wandb_group
    )
    wandb_logger.experiment.config.update(args)
    config = wandb.config

    train_loader, val_loader, test_loader = get_idsprites_loaders(args)
    vae = LightningBetaVAE(
        img_size=args.img_size, latent_dim=config.latent_dim, beta=config.beta
    )
    trainer = Trainer(
        accelerator="auto",
        default_root_dir=args.wandb_dir,
        devices=1,
        enable_checkpointing=False,
        limit_train_batches=args.train_on,
        limit_val_batches=args.eval_on,
        limit_test_batches=args.test_on,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=args.epochs,
    )
    tuner = Tuner(trainer)
    tuner.lr_find(vae)
    trainer.fit(vae, train_loader, val_loader)
    trainer.test(vae, test_loader)


def get_idsprites_loaders(args):
    """Get the data loaders for idSprites."""
    train_set = InfiniteDSprites(args.img_size)
    val_set = InfiniteDSprites(args.img_size)
    test_set = InfiniteDSprites(args.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
    )
    return train_loader, val_loader, test_loader


def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=100,
        help="How often to log training progress. The logs will be averaged over this number of batches.",
    )
    parser.add_argument(
        "--tasks", type=int, default=5, help="Number of continual learning tasks."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter.")
    parser.add_argument(
        "--img_size", type=int, default=128, help="Size of the images in the dataset."
    )
    parser.add_argument(
        "--train_on", type=int, default=10000, help="Number of batches to train on."
    )
    parser.add_argument(
        "--eval_on", type=int, default=1000, help="Number of batches to evaluate on."
    )
    parser.add_argument(
        "--test_on", type=int, default=1000, help="Number of batches to test on."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="Dimensionality of the latent space."
    )
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
    # train(args)
    train_vae(args)


if __name__ == "__main__":
    _main()
