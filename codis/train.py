"""Training script."""
from collections import defaultdict

import hydra
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from collections import Counter

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
from codis.utils import grouper

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train the model in a continual learning setting."""
    shapes = [
        InfiniteDSprites().generate_shape()
        for _ in range(cfg.dataset.tasks * cfg.dataset.shapes_per_task)
    ]
    shape_ids = range(len(shapes))
    model = build_model(cfg)
    canonical_images = generate_canonical_images(shapes, img_size=cfg.dataset.img_size)
    random_images = generate_random_images(shapes, img_size=cfg.dataset.img_size)
    callbacks = build_callbacks(cfg, canonical_images, random_images)
    trainer = build_trainer(cfg, callbacks=callbacks)

    test_dataset = None
    if cfg.training.mode == "continual":
        for task_id, (task_shapes, task_shape_ids, task_exemplars) in enumerate(
            zip(
                grouper(cfg.dataset.shapes_per_task, shapes),
                grouper(cfg.dataset.shapes_per_task, shape_ids),
                grouper(cfg.dataset.shapes_per_task, canonical_images),
            )
        ):
            train_dataset, val_dataset, task_test_dataset = build_continual_datasets(
                cfg, task_shapes, task_shape_ids
            )
            train_loader = build_dataloader(cfg, train_dataset)
            val_loader = build_dataloader(cfg, val_dataset, shuffle=False)

            test_dataset = update_test_dataset(cfg, test_dataset, task_test_dataset)
            print(
                f"Task {task_id}:",
                Counter([factors.shape_id for _, factors in test_dataset]),
            )
            test_loader = build_dataloader(cfg, test_dataset)  # shuffle for vis

            model.task_id = task_id
            for exemplar in task_exemplars:
                model.add_exemplar(exemplar)
            trainer.fit(model, train_loader, val_loader)
            trainer.fit_loop.max_epochs += cfg.training.max_epochs
            trainer.test(model, test_loader)

    elif cfg.training.mode == "joint":
        model.task_id = 0
        if model.has_buffer:
            for exemplar in canonical_images:
                model.add_exemplar(exemplar)
        train_loader, val_loader, test_loader = build_joint_data_loaders(cfg, shapes)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    wandb.finish()


def generate_canonical_images(shapes, img_size):
    """Generate a batch of exemplars for training and visualization."""
    dataset = InfiniteDSprites(img_size=img_size)
    return [
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


def generate_random_images(shapes, img_size, n=25):
    """Generate a batch of images for visualization."""
    dataset = InfiniteDSprites(img_size=img_size, shapes=shapes)
    return [dataset.draw(dataset.sample_latents()) for _ in range(n)]


def build_dataloader(cfg: DictConfig, dataset, shuffle=True):
    """Prepare a data loader."""
    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
    )


def build_model(cfg: DictConfig):
    """Prepare the appropriate model."""
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
    return model


def build_callbacks(cfg: DictConfig, canonical_images: list, random_images: list):
    """Prepare the appropriate callbacks."""
    callbacks = []
    callback_names = cfg.model.callbacks
    if "logging" in callback_names:
        callbacks.append(LoggingCallback())
    if "visualization" in callback_names:
        callbacks.append(VisualizationCallback(canonical_images, random_images))
    return callbacks


def build_trainer(cfg: DictConfig, callbacks=None):
    """Configure the model trainer."""
    wandb_logger = WandbLogger(
        project="codis",
        save_dir=cfg.wandb.dir,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
    )
    return Trainer(
        default_root_dir=cfg.wandb.dir,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=cfg.wandb.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
    )


def build_continual_datasets(cfg: DictConfig, shapes: list, shape_ids: list):
    """Build data loaders for a class-incremental continual learning scenario."""
    n = cfg.dataset.factor_resolution
    scale_range = np.linspace(0.5, 1.0, n)
    orientation_range = np.linspace(0, 2 * np.pi * (n / (n + 1)), n)
    position_x_range = np.linspace(0, 1, n)
    position_y_range = np.linspace(0, 1, n)

    dataset = ContinualDSpritesMap(
        img_size=cfg.dataset.img_size,
        shapes=shapes,
        shape_ids=shape_ids,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [
            cfg.dataset.train_split,
            cfg.dataset.val_split,
            cfg.dataset.test_split,
        ],
    )
    return train_dataset, val_dataset, test_dataset


def update_test_dataset(
    cfg: DictConfig,
    test_dataset: Dataset,
    task_test_dataset: Dataset,
    samples_per_shape: int,
):
    """Update the test dataset keeping it class-balanced."""
    samples_per_shape = cfg.dataset.test_dataset_size // (
        cfg.dataset.tasks * cfg.dataset.shapes_per_task
    )
    if test_dataset is None:
        test_dataset = task_test_dataset
    else:
        class_indices = defaultdict(list)
        for i, (_, factors) in enumerate(task_test_dataset):
            class_indices[factors.shape_id].append(i)
        subset_indices = []
        for indices in class_indices.values():
            subset_indices.extend(np.random.choice(indices, samples_per_shape))
        test_dataset = ConcatDataset([test_dataset, Subset(task_test_dataset, indices)])

    return test_dataset


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
