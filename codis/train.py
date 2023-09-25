"""Training script."""
import os

import hydra
import numpy as np
import torch
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    LazyStreamDefinition,
    create_lazy_generic_benchmark,
)
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import EWC, GEM, GDumb, LwF, Naive, Replay
from hydra.utils import call, get_object, instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split

from codis.data import (
    ContinualDataset,
    InfiniteDSprites,
    Latents,
    RandomDSpritesMap,
)
from codis.lightning.callbacks import LoggingCallback, VisualizationCallback
from codis.lightning.modules import ContinualModule

torch.set_float32_matmul_precision("high")
OmegaConf.register_new_resolver("eval", eval)


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

    if cfg.training.mode == "joint":
        train_jointly(cfg, trainer, shapes, exemplars)
    elif cfg.training.mode == "continual":
        train_continually(cfg, trainer, shapes, exemplars)
    else:
        raise ValueError(f"Unknown training mode: {cfg.training.mode}.")


def train_jointly(cfg: DictConfig, trainer, shapes, exemplars):
    """Train jointly on all shapes."""
    model = instantiate(cfg.model)
    model.task_id = 0
    for exemplar in exemplars:
        model.add_exemplar(exemplar)
    train_loader, val_loader, test_loader = build_joint_data_loaders(cfg, shapes)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def train_continually(cfg: DictConfig, trainer, shapes, exemplars):
    """Train continually with n shapes per tasks."""
    dataset = ContinualDataset(cfg, shapes=shapes, exemplars=exemplars)
    target = get_object(cfg.model._target_)
    print(target, type(target))
    if isinstance(target, ContinualModule):
        model = instantiate(cfg.model)
        train_ours(cfg, model, trainer, dataset)
    else:
        model = call(cfg.model)
        train_baseline(cfg, model, dataset)


def train_ours(cfg, model, trainer, dataset):
    """Train our model in a continual learning setting."""
    for task_id, (datasets, task_exemplars) in enumerate(dataset):
        model.task_id = task_id
        train_dataset, val_dataset, test_dataset = datasets
        train_loader = build_dataloader(train_dataset)
        val_loader = build_dataloader(val_dataset, shuffle=False)
        test_loader = build_dataloader(test_dataset)  # shuffle for vis
        for exemplar in task_exemplars:
            model.add_exemplar(exemplar)
        trainer.fit(model, train_loader, val_loader)
        if (
            not cfg.training.test_once
            and task_id % cfg.training.test_every_n_tasks == 0
        ):
            trainer.test(model, test_loader)
        trainer.fit_loop.max_epochs += cfg.trainer.max_epochs
    trainer.test(model, test_loader)


def train_baseline(cfg, model, continual_dataset):
    """Train standard continual learning baselines using Avalanche."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_generator = (
        make_classification_dataset(
            dataset=datasets[0],
            task_labels=[task_id] * len(datasets[0]),
            target_transform=lambda y: y.shape_id,
        )
        for task_id, (datasets, _) in enumerate(continual_dataset)
    )
    test_generator = (
        make_classification_dataset(
            dataset=datasets[2],
            task_labels=[task_id] * len(datasets[2]),
            target_transform=lambda y: y.shape_id,
        )
        for task_id, (datasets, _) in enumerate(continual_dataset)
    )
    train_stream = LazyStreamDefinition(
        train_generator,
        stream_length=continual_dataset.tasks,
        exps_task_labels=range(continual_dataset.tasks),
    )
    test_stream = LazyStreamDefinition(
        test_generator,
        stream_length=continual_dataset.tasks,
        exps_task_labels=range(continual_dataset.tasks),
    )
    benchmark = create_lazy_generic_benchmark(train_stream, test_stream)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")
    loggers = [
        InteractiveLogger(),
        WandBLogger(
            dir=cfg.wandb.save_dir,
            project_name=f"{cfg.wandb.project}_baselines",
            params={"group": cfg.wandb.group},
            config=config,
        ),
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, experience=True),
        loss_metrics(minibatch=True),
        forgetting_metrics(experience=True),
        # confusion_matrix_metrics(),
        loggers=loggers,
    )

    strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=cfg.training.lr),
        CrossEntropyLoss(),
        train_mb_size=cfg.dataset.batch_size,
        train_epochs=cfg.trainer.max_epochs,
        eval_mb_size=cfg.dataset.batch_size,
        evaluator=eval_plugin,
        device=device,
    )
    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        train_task = train_experience.current_experience
        print(f"Task {train_task} train: {len(train_experience.dataset)} samples.")
        print(f"Classes train: {train_experience.classes_in_this_experience}")
        strategy.train(train_experience)

        test_task = test_experience.current_experience
        print(f"Task {test_task} test: {len(test_experience.dataset)} samples.")
        print(f"Classes test: {test_experience.classes_in_this_experience}")
        strategy.eval(test_experience)


def build_dataloader(self, dataset, shuffle=True):
    """Prepare a data loader."""
    return DataLoader(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=shuffle,
    )


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
    if "checkpointing" in callback_names:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.trainer.default_root_dir,
                every_n_epochs=cfg.training.test_every_n_tasks * cfg.trainer.max_epochs,
            )
        )
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
