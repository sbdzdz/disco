"""Training script."""
import inspect
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    LazyStreamDefinition,
    create_lazy_generic_benchmark,
)
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from hydra.utils import call, get_object, instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from codis.data import (
    ContinualBenchmark,
    ContinualBenchmarkRehearsal,
    InfiniteDSprites,
    Latents,
)
from codis.lightning.callbacks import (
    LoggingCallback,
    MetricsCallback,
    VisualizationCallback,
)

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

    strategy = cfg.training.strategy
    if strategy == "naive":
        benchmark = ContinualBenchmark(cfg, shapes=shapes, exemplars=exemplars)
    elif strategy == "rehearsal":
        benchmark = ContinualBenchmarkRehearsal(cfg, shapes=shapes, exemplars=exemplars)
    else:
        raise ValueError(f"Unknown strategy: {strategy}.")

    target = get_object(cfg.model._target_)
    if inspect.isclass(target):
        train_ours_continually(cfg, benchmark, trainer)
    elif callable(target):
        train_baseline_continually(cfg, benchmark)
    else:
        raise ValueError(f"Unknown target: {target}.")


def train_ours_continually(cfg, benchmark, trainer):
    """Train our model in a continual learning setting."""
    model = instantiate(cfg.model)
    for task_id, (datasets, task_exemplars) in enumerate(benchmark):
        model.task_id = task_id
        train_loader, val_loader, test_loader = create_loaders(cfg, datasets)

        for exemplar in task_exemplars:
            model.add_exemplar(exemplar)

        if cfg.training.validate:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        if (
            not cfg.training.test_once
            and task_id % cfg.training.test_every_n_tasks == 0
        ):
            trainer.test(model, test_loader)
        trainer.fit_loop.max_epochs += cfg.trainer.max_epochs
    trainer.test(model, test_loader)


def create_loaders(cfg, datasets):
    """Create the data loaders."""
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,  # shuffle for vis
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


def train_baseline_continually(cfg, benchmark):
    """Train a standard continual learning baseline using Avalanche."""
    model = call(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.run.define_metric("*", step_metric="Step", step_sync=True)
    train_generator = (
        make_classification_dataset(
            dataset=datasets[0],
            target_transform=lambda y: y.shape_id,
        )
        for datasets, _ in benchmark
    )
    test_generator = (
        make_classification_dataset(
            dataset=datasets[2],
            target_transform=lambda y: y.shape_id,
        )
        for datasets, _ in benchmark
    )
    train_stream = LazyStreamDefinition(
        train_generator,
        stream_length=benchmark.tasks,
        exps_task_labels=[0] * benchmark.tasks,
    )
    test_stream = LazyStreamDefinition(
        test_generator,
        stream_length=benchmark.tasks,
        exps_task_labels=[0] * benchmark.tasks,
    )
    benchmark = create_lazy_generic_benchmark(train_stream, test_stream)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")
    loggers = [
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
        loggers=loggers,
    )

    strategy = instantiate(
        cfg.strategy,
        model=model,
        device=device,
        evaluator=eval_plugin,
        optimizer={"params": model.parameters()},
    )
    for train_experience, test_experience in zip(
        benchmark.train_stream, benchmark.test_stream
    ):
        train_task = train_experience.current_experience
        print(f"Task {train_task} train: {len(train_experience.dataset)} samples.")
        print(f"Classes train: {train_experience.classes_in_this_experience}")
        strategy.train(train_experience)

        test_task = test_experience.current_experience
        if (
            not cfg.training.test_once
            and test_task % cfg.training.test_every_n_tasks == 0
        ):
            print(f"Task {test_task} test: {len(test_experience.dataset)} samples.")
            min_class_id = min(test_experience.classes_in_this_experience)
            max_class_id = max(test_experience.classes_in_this_experience)
            print(f"Classes test: {min_class_id}-{max_class_id}")
            strategy.eval(test_experience)


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
    if "checkpointing" in callback_names:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(cfg.trainer.default_root_dir)
                / os.environ.get("SLURM_JOB_ID"),
                every_n_epochs=cfg.training.checkpoint_every_n_tasks
                * cfg.training.epochs_per_task,
                save_top_k=-1,
                save_weights_only=True,
            )
        )
    if "learning_rate_monitor" in callback_names:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if "logging" in callback_names:
        callbacks.append(LoggingCallback())
    if "metrics" in callback_names:
        callbacks.append(
            MetricsCallback(
                log_train_accuracy=cfg.training.log_train_accuracy,
                log_val_accuracy=cfg.training.log_val_accuracy,
                log_test_accuracy=cfg.training.log_test_accuracy,
            )
        )
    if "timer" in callback_names:
        callbacks.append(Timer())
    if "visualization" in callback_names:
        callbacks.append(VisualizationCallback(canonical_images, random_images))
    return callbacks


if __name__ == "__main__":
    train()
