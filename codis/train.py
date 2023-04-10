"""Training script."""
import argparse
from pathlib import Path

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from codis.data import ContinualDSprites
from codis.lightning_modules import CodisModel, LightningBetaVAE, LightningMLP

torch.set_float32_matmul_precision("medium")


def train(args):
    """Train the model in a continual learning setting."""
    print(f"Cuda available {torch.cuda.is_available()}")

    backbone = LightningBetaVAE(
        img_size=args.img_size, latent_dim=args.latent_dim, beta=args.beta
    )
    regressor = LightningMLP(
        dims=[args.latent_dim, 64, 64, 7]
    )  # 7 is the number of stacked latent values
    model = CodisModel(backbone, regressor)
    trainer = configure_trainer(args)

    dataset = ContinualDSprites(args.img_size)
    shapes = [dataset.generate_shape() for _ in range(args.tasks)]

    for i in range(args.tasks):
        train_loader, val_loader, test_loader = get_continual_loaders(args, shapes, i)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    wandb.finish()


def get_continual_loaders(args, shapes: list, task: int = 0):
    """Get the data loaders for continual learning on dSprites."""
    train_set = ContinualDSprites(shapes[task])
    val_set = ContinualDSprites(shapes[task])
    test_set = ContinualDSprites(shapes[: task + 1])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_vae(args):
    """Train the VAE on dSprites or idSprites."""
    print(f"Cuda available {torch.cuda.is_available()}")
    vae = LightningBetaVAE(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        lr=args.lr,
    )
    trainer = configure_trainer(args)
    train_loader, val_loader, test_loader = get_idsprites_loaders(args)
    trainer.fit(vae, train_loader, val_loader)
    trainer.test(vae, test_loader)


def configure_trainer(args):
    """Configure the model trainer."""
    wandb_logger = WandbLogger(
        project="codis", save_dir=args.wandb_dir, group=args.wandb_group
    )
    wandb_logger.experiment.config.update(args)
    return Trainer(
        accelerator="auto",
        default_root_dir=args.wandb_dir,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=args.train_on,
        limit_val_batches=args.eval_on,
        limit_test_batches=args.test_on,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=args.epochs,
    )


def get_idsprites_loaders(args):
    """Get the data loaders for idSprites."""
    train_set = ContinualDSprites(args.img_size)
    val_set = ContinualDSprites(args.img_size)
    test_set = ContinualDSprites(args.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=16,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=16,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=16,
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
        help="How often to log training progress. The metrics will be averaged.",
    )
    parser.add_argument(
        "--tasks", type=int, default=5, help="Number of continual learning tasks."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--img_size", type=int, default=512, help="Size of the images in the dataset."
    )
    parser.add_argument(
        "--train_on", type=int, default=1000, help="Number of batches to train on."
    )
    parser.add_argument(
        "--eval_on", type=int, default=100, help="Number of batches to evaluate on."
    )
    parser.add_argument(
        "--test_on", type=int, default=100, help="Number of batches to test on."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="Dimensionality of the latent space."
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter.")
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
    parser.add_argument(
        "--experiment", type=str, choices=["codis", "vae"], default="vae"
    )
    args = parser.parse_args()
    if args.experiment == "codis":
        train(args)
    elif args.experiment == "vae":
        train_vae(args)


if __name__ == "__main__":
    _main()
