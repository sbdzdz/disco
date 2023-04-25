"""Training script."""
import argparse
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from codis.data import ContinualDSprites, InfiniteDSprites
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
    model = CodisModel(backbone, regressor, gamma=args.gamma)
    train_loaders, val_loaders, test_loaders = configure_ci_loaders(args)
    trainer = configure_trainer(args)
    if args.watch_gradients:
        trainer.logger.watch(model)

    for task_id, (train_loader, val_loader) in enumerate(
        zip(train_loaders, val_loaders)
    ):
        print(f"Starting task {task_id}...")
        model.task_id = task_id
        trainer.fit(model, train_loader, val_loader)
        for test_loader in test_loaders:
            trainer.test(model, test_loader)
    wandb.finish()


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
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
    )


def configure_ci_loaders(args):
    """Configure data loaders for a class-incremental continual learning scenario."""
    scale_range = np.linspace(0.5, 1.5, 16)
    orientation_range = np.linspace(0, 2 * np.pi, 16)
    position_x_range = np.linspace(0, 1, 16)
    position_y_range = np.linspace(0, 1, 16)

    shapes = [InfiniteDSprites.generate_shape() for _ in range(args.tasks)]
    datasets = [
        ContinualDSprites(
            img_size=args.img_size,
            shapes=[shape],
            scale_range=scale_range,
            orientation_range=orientation_range,
            position_x_range=position_x_range,
            position_y_range=position_y_range,
        )
        for shape in shapes
    ]
    train_datasets, test_datasets = zip(
        *[random_split(d, [0.8, 0.2]) for d in datasets]
    )
    val_datasets, test_datasets = zip(
        *[random_split(d, [0.5, 0.5]) for d in test_datasets]
    )
    train_loaders = [
        DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers)
        for d in train_datasets
    ]
    val_loaders = [
        DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers)
        for d in val_datasets
    ]
    test_loaders = [
        DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers)
        for d in test_datasets
    ]

    return train_loaders, val_loaders, test_loaders


def get_idsprites_loaders(args):
    """Get the data loaders for idSprites."""
    train_set = ContinualDSprites(args.img_size)
    val_set = ContinualDSprites(args.img_size)
    test_set = ContinualDSprites(args.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        default=50,
        help="How often to log training progress. The metrics will be averaged.",
    )
    parser.add_argument(
        "--tasks", type=int, default=5, help="Number of continual learning tasks."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="Size of the images in the dataset."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="Dimensionality of the latent space."
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Beta parameter for the beta-VAE."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Relative weight of the backbone and regressor loss. 0 is only backbone loss, 1 is only regressor loss.",
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
    parser.add_argument(
        "--experiment", type=str, choices=["codis", "vae"], default="codis"
    )
    parser.add_argument(
        "--watch_gradients",
        help="Whether to log gradients in wandb.",
        action="store_true",
    )
    args = parser.parse_args()
    if args.experiment == "codis":
        train(args)
    elif args.experiment == "vae":
        train_vae(args)


if __name__ == "__main__":
    _main()
