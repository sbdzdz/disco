"""Training script."""
import argparse
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from codis.data import (
    ContinualDSpritesMap,
    InfiniteDSprites,
    Latents,
    RandomDSpritesMap,
)
from codis.lightning.callbacks import LoggingCallback, VisualizationCallback
from codis.lightning.modules import (
    LatentRegressor,
    LightningBetaVAE,
    SpatialTransformer,
    SpatialTransformerSimple,
    SupervisedVAE,
)

torch.set_float32_matmul_precision("medium")


def train(args):
    """Train the model in a continual learning setting."""
    shapes = [InfiniteDSprites().generate_shape() for _ in range(args.tasks)]
    exemplars = generate_exemplars(shapes, img_size=args.img_size)
    model, callbacks = build_model_and_callbacks(args, exemplars)
    trainer = build_trainer(args, callbacks=callbacks)

    if args.training == "continual":
        test_loaders = []
        for train_task_id, (shape, exemplar) in enumerate(zip(shapes, exemplars)):
            train_loader, val_loader, test_loader = build_continual_data_loaders(
                args, shape
            )
            test_loaders.append(test_loader)
            model.task_id = train_task_id
            if model.has_buffer:
                model.add_exemplar(exemplar)
            trainer.fit(model, train_loader, val_loader)
            trainer.fit_loop.max_epochs += args.max_epochs
            for test_task_id, test_loader in enumerate(test_loaders):
                model.task_id = test_task_id
                trainer.test(model, test_loader)
    elif args.training == "joint":
        model.task_id = 0
        if model.has_buffer:
            for exemplar in exemplars:
                model.add_exemplar(exemplar)
        train_loader, val_loader, test_loader = build_joint_data_loaders(args, shapes)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    wandb.finish()


def generate_exemplars(shapes, img_size):
    """Generate a batch of exemplars for visualization."""
    dataset = InfiniteDSprites(img_size=img_size)
    batch = [
        dataset.draw(
            Latents(
                color=(1.0, 1.0, 1.0),
                shape=shape,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            )
        )
        for shape in shapes
    ]
    return torch.stack([torch.from_numpy(img) for img in batch])


def build_model_and_callbacks(args, exemplars):
    """Prepare the appropriate model."""
    callbacks = [VisualizationCallback(exemplars), LoggingCallback()]
    if args.model == "vae":
        vae = LightningBetaVAE(
            img_size=args.img_size,
            latent_dim=args.latent_dim,
            beta=args.beta,
            lr=args.lr,
        )
        model = SupervisedVAE(
            vae=vae, gamma=args.gamma, factors_to_regress=args.factors_to_regress
        )
    elif args.model == "stn":
        model = SpatialTransformer(
            img_size=args.img_size,
            lr=args.lr,
            factors_to_regress=args.factors_to_regress,
        )
    elif args.model == "stn_simple":
        model = SpatialTransformerSimple(
            img_size=args.img_size,
            lr=args.lr,
            factors_to_regress=args.factors_to_regress,
        )
    elif args.model == "regressor":
        model = LatentRegressor(
            img_size=args.img_size,
            lr=args.lr,
            factors_to_regress=args.factors_to_regress,
        )
        callbacks = [LoggingCallback()]
    else:
        raise ValueError(f"Unknown model {args.model}.")
    return model, callbacks


def build_trainer(args, callbacks=None):
    """Configure the model trainer."""
    wandb_logger = WandbLogger(
        project="codis", save_dir=args.wandb_dir, group=args.wandb_group
    )
    wandb_logger.experiment.config.update(args)
    if callbacks is None:
        callbacks = []
    return Trainer(
        accelerator="auto",
        default_root_dir=args.wandb_dir,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
    )


def build_continual_data_loaders(args, shapes):
    """Build data loaders for a class-incremental continual learning scenario."""
    scale_range = np.linspace(0.5, 1.5, args.factor_resolution)
    orientation_range = np.linspace(0, 2 * np.pi, args.factor_resolution)
    position_x_range = np.linspace(0, 1, args.factor_resolution)
    position_y_range = np.linspace(0, 1, args.factor_resolution)

    if not isinstance(shapes, list):
        shapes = [shapes]

    dataset = ContinualDSpritesMap(
        img_size=args.img_size,
        shapes=shapes,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    if args.train_dataset_size is not None:
        dataset = torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), args.train_dataset_size)
        )

    train_dataset, test_dataset = random_split(dataset, [0.95, 0.05])
    val_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return train_loader, val_loader, test_loader


def build_joint_data_loaders(args, shapes):
    """Build data loaders for a joint training scenario."""
    scale_range = np.linspace(0.5, 1.5, args.factor_resolution)
    orientation_range = np.linspace(0, 2 * np.pi, args.factor_resolution)
    position_x_range = np.linspace(0, 1, args.factor_resolution)
    position_y_range = np.linspace(0, 1, args.factor_resolution)

    dataset = RandomDSpritesMap(
        img_size=args.img_size,
        shapes=shapes,
        dataset_size=args.train_dataset_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    test_shapes = [
        InfiniteDSprites().generate_shape() for _ in range(args.num_test_shapes)
    ]
    test_dataset = RandomDSpritesMap(
        img_size=args.img_size,
        shapes=test_shapes,
        dataset_size=args.test_dataset_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return train_loader, val_loader, test_loader


def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="How often to log training progress. The metrics will be averaged.",
    )
    parser.add_argument(
        "--training",
        type=str,
        default="continual",
        choices=["continual", "joint"],
        help="Training mode. One of 'continual' or 'joint'.",
    )
    parser.add_argument(
        "--tasks", type=int, default=5, help="Number of continual learning tasks."
    )
    parser.add_argument(
        "--num_test_shapes",
        type=int,
        default=1000,
        help="Number of shapes to use for OOD testing.",
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=None,
        help="Number of samples to use from the training dataset. If None, use the entire dataset.",
    )
    parser.add_argument(
        "--test_dataset_size",
        type=int,
        default=None,
        help="Number of samples to use from the test dataset. If None, use the entire dataset.",
    )
    parser.add_argument(
        "--factor_resolution",
        type=int,
        default=16,
        help="Resolution of the factors of variation. The dataset size is factor_resolution ** 4.",
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
        "--model",
        type=str,
        default="stn",
        choices=["vae", "stn", "stn_simple", "regressor"],
        help="Model to train. One of 'vae' or 'stn'.",
    )
    parser.add_argument(
        "--factors_to_regress",
        type=str,
        nargs="+",
        default=["orientation", "scale", "position_x", "position_y"],
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
