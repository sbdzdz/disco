"""Training script."""
import os

import hydra
import numpy as np
import torch
from avalanche.benchmarks.scenarios import ClassificationExperience
from avalanche.training.supervised import EWC, GEM, GDumb, LwF, Naive, Replay
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    LazyStreamDefinition,
    create_lazy_generic_benchmark,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from codis.data import (
    InfiniteDSprites,
    Latents,
    RandomDSpritesMap,
)
from codis.lightning.callbacks import LoggingCallback, VisualizationCallback
from codis.lightning.modules import ContinualModule
from codis.data import ContinualDataset

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model in a continual learning setting."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")

    shapes = [
        InfiniteDSprites().generate_shape()
        for _ in range(cfg.dataset.tasks * cfg.dataset.shapes_per_task)
    ]
    exemplars = generate_canonical_images(shapes, img_size=cfg.dataset.img_size)
    random_images = generate_random_images(
        shapes,
        img_size=cfg.dataset.img_size,
        factor_resolution=cfg.dataset.factor_resolution,
    )
    callbacks = build_callbacks(cfg, exemplars, random_images)

    trainer = instantiate(cfg.trainer, callbacks=callbacks)
    trainer.logger.log_hyperparams(config)

    if cfg.training.mode == "continual":
        train_continually(cfg, trainer, shapes, exemplars)
    elif cfg.training.mode == "joint":
        train_jointly(cfg, trainer, shapes, exemplars)
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}")


def train_continually(cfg, trainer, shapes, exemplars):
    """Train continually on a batch of shapes per task."""
    model = instantiate(cfg.model)
    continual_dataset = ContinualDataset(cfg, shapes=shapes, exemplars=exemplars)

    if isinstance(model, ContinualModule):  # ours
        for task_id, (loaders, task_exemplars) in enumerate(continual_dataset):
            model.task_id = task_id
            train_loader, val_loader, test_loader = loaders
            for exemplar in task_exemplars:
                model.add_exemplar(exemplar)
            trainer.fit(model, train_loader, val_loader)
            if task_id % 10 == 0:  # test every 10 tasks
                trainer.test(model, test_loader)
            trainer.fit_loop.max_epochs += cfg.trainer.max_epochs
    else:
        strategy = Naive(
            model,
            optimizer=torch.optim.Adam(model.parameters()),
            criterion=torch.nn.CrossEntropyLoss(),
        )
        train_stream = LazyStreamDefinition(
            (loaders[0] for loaders, _ in continual_dataset),
            stream_length=continual_dataset.num_tasks,
            task_labels=range(continual_dataset.num_tasks),
        )
        test_stream = LazyStreamDefinition(
            (loaders[2] for loaders, _ in continual_dataset),
            stream_length=continual_dataset.num_tasks,
            task_labels=range(continual_dataset.num_tasks),
        )
        benchmark = create_lazy_generic_benchmark(
            train_stream, test_stream, task_labels=range(continual_dataset.num_tasks)
        )
        for experience in benchmark.train_stream:
            strategy.train(experience)
            test_set = benchmark.test_stream[experience.current_experience]
            strategy.eval(test_set)


def train_jointly(cfg, trainer, shapes, exemplars):
    """Train jointly on all shapes."""
    model = instantiate(cfg.model)
    model.task_id = 0
    if model.has_buffer:
        for exemplar in exemplars:
            model.add_exemplar(exemplar)
    train_loader, val_loader, test_loader = build_joint_data_loaders(cfg, shapes)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def generate_canonical_images(shapes, img_size: int):
    """Generate a batch of exemplars for training and visualization."""
    dataset = InfiniteDSprites(
        img_size=img_size,
    )
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


def generate_random_images(
    shapes: list, img_size: int, factor_resolution: int, num_imgs: int = 25
):
    """Generate a batch of images for visualization."""
    scale_range = np.linspace(0.5, 1.0, factor_resolution)
    orientation_range = np.linspace(0, 2 * np.pi, factor_resolution)
    position_x_range = np.linspace(0, 1, factor_resolution)
    position_y_range = np.linspace(0, 1, factor_resolution)
    dataset = InfiniteDSprites(
        img_size=img_size,
        shapes=shapes,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
    )
    return [dataset.draw(dataset.sample_latents()) for _ in range(num_imgs)]


def build_callbacks(cfg: DictConfig, canonical_images: list, random_images: list):
    """Prepare the appropriate callbacks."""
    callbacks = []
    callback_names = cfg.training.callbacks
    if "logging" in callback_names:
        callbacks.append(LoggingCallback())
    if "visualization" in callback_names:
        callbacks.append(VisualizationCallback(canonical_images, random_images))
    return callbacks


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
