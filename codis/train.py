"""Training script."""
import hydra
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
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
    LightningBetaVAE,
    SpatialTransformer,
    SpatialTransformerGF,
    SupervisedVAE,
)

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train the model in a continual learning setting."""
    shapes = [InfiniteDSprites().generate_shape() for _ in range(cfg.dataset.tasks)]
    exemplars = generate_exemplars(shapes, img_size=cfg.dataset.img_size)
    model, callbacks = build_model_and_callbacks(cfg, exemplars)
    trainer = build_trainer(cfg, callbacks=callbacks)

    if cfg.training.mode == "continual":
        test_loaders = []
        for train_task_id, (shape, exemplar) in enumerate(zip(shapes, exemplars)):
            train_loader, val_loader, test_loader = build_continual_data_loaders(
                cfg, shape
            )
            test_loaders.append(test_loader)
            model.task_id = train_task_id
            if model.has_buffer:
                model.add_exemplar(exemplar)
            trainer.fit(model, train_loader, val_loader)
            trainer.fit_loop.max_epochs += cfg.training.max_epochs
            for test_task_id, test_loader in enumerate(test_loaders):
                model.task_id = test_task_id
                trainer.test(model, test_loader)
    elif cfg.training.mode == "joint":
        model.task_id = 0
        if model.has_buffer:
            for exemplar in exemplars:
                model.add_exemplar(exemplar)
        train_loader, val_loader, test_loader = build_joint_data_loaders(cfg, shapes)
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
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            )
        )
        for shape in shapes
    ]
    return torch.stack([torch.from_numpy(img) for img in batch])


def build_model_and_callbacks(cfg: DictConfig, exemplars: list):
    """Prepare the appropriate model."""
    callbacks = [VisualizationCallback(exemplars), LoggingCallback()]
    if cfg.model.name == "vae":
        vae = LightningBetaVAE(
            img_size=cfg.dataset.img_size,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            lr=cfg.training.lr,
        )
        model = SupervisedVAE(
            vae=vae,
            gamma=cfg.model.gamma,
            factors_to_regress=cfg.model.factors_to_regress,
        )
    elif cfg.model.name == "stn":
        model = SpatialTransformer(
            img_size=cfg.dataset.img_size,
            in_channels=cfg.dataset.num_channels,
            channels=cfg.model.channels,
            gamma=cfg.model.gamma,
            lr=cfg.training.lr,
            factors_to_regress=cfg.model.factors_to_regress,
        )
    elif cfg.model.name == "stn_gf":
        model = SpatialTransformerGF(
            img_size=cfg.dataset.img_size,
            in_channels=cfg.dataset.num_channels,
            channels=cfg.model.channels,
            gamma=cfg.model.gamma,
            lr=cfg.training.lr,
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.name}.")
    return model, callbacks


def build_trainer(cfg: DictConfig, callbacks=None):
    """Configure the model trainer."""
    wandb_logger = WandbLogger(
        project="codis", save_dir=cfg.wandb.dir, group=cfg.wandb.group
    )
    if callbacks is None:
        callbacks = []
    return Trainer(
        accelerator="auto",
        default_root_dir=cfg.wandb.dir,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=cfg.wandb.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
    )


def build_continual_data_loaders(cfg: DictConfig, shapes: list):
    """Build data loaders for a class-incremental continual learning scenario."""
    n = cfg.dataset.factor_resolution
    scale_range = np.linspace(0.5, 1.0, n)
    orientation_range = np.linspace(0, 2 * np.pi * (n / (n + 1)), n)
    position_x_range = np.linspace(0, 1, n)
    position_y_range = np.linspace(0, 1, n)

    if not isinstance(shapes, list):
        shapes = [shapes]

    dataset = ContinualDSpritesMap(
        img_size=cfg.dataset.img_size,
        shapes=shapes,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    train_dataset, test_dataset = random_split(dataset, [0.95, 0.05])
    val_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])

    if cfg.dataset.train_dataset_size is not None:
        dataset = torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), cfg.dataset.train_dataset_size)
        )

    if cfg.dataset.test_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(
            test_dataset,
            np.random.choice(len(test_dataset), cfg.dataset.test_dataset_size),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    return train_loader, val_loader, test_loader


def build_joint_data_loaders(cfg: DictConfig, shapes):
    """Build data loaders for a joint training scenario."""
    scale_range = np.linspace(0.5, 1.5, cfg.dataset.factor_resolution)
    orientation_range = np.linspace(0, 2 * np.pi, cfg.dataset.factor_resolution)
    position_x_range = np.linspace(0, 1, cfg.dataset.factor_resolution)
    position_y_range = np.linspace(0, 1, cfg.dataset.factor_resolution)

    dataset = RandomDSpritesMap(
        img_size=cfg.dataset.img_size,
        shapes=shapes,
        dataset_size=cfg.dataset.train_dataset_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    test_shapes = [
        InfiniteDSprites().generate_shape() for _ in range(cfg.dataset.num_test_shapes)
    ]
    test_dataset = RandomDSpritesMap(
        img_size=cfg.dataset.img_size,
        shapes=test_shapes,
        dataset_size=cfg.dataset.test_dataset_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train()
