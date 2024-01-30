"""Training script."""

import inspect
import os
from pathlib import Path
from typing import Union

import hydra
import idsprites as ids
import numpy as np
import torch
import wandb
from avalanche.benchmarks.generators import paths_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from hydra.utils import call, get_object, instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from disco.lightning.callbacks import (
    LoggingCallback,
    MetricsCallback,
    VisualizationCallback,
)

torch.set_float32_matmul_precision("high")
OmegaConf.register_new_resolver("eval", eval)


class FileDataset(Dataset):
    def __init__(self, path: Union[Path, str], transform=None, target_transform=None):
        self.path = Path(path)
        self.transform = transform
        self.target_transform = target_transform
        self.factors = np.load(self.path / "factors.npz", allow_pickle=True)

    def __len__(self):
        return len(self.factors["shape_id"])

    def __getitem__(self, idx):
        img_path = self.path / f"sample_{idx}.png"
        image = read_image(str(img_path))

        factors = ids.Factors(
            **{key: value[idx] for key, value in self.factors.items()}
        )
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            factors = self.target_transform(factors)
        return image, factors


class ContinualBenchmarkDisk:
    def __init__(
        self,
        path: Union[Path, str],
    ):
        """Initialize the continual learning benchmark.
        Args:
            cfg: The configuration object.
        """
        self.path = Path(path)

    def __iter__(self):
        for task_dir in self.path.glob("task_*"):
            task_exemplars = self.load_exemplars(task_dir)
            train = FileDataset(task_dir / "train")
            val = FileDataset(task_dir / "val")
            test = FileDataset(task_dir / "test")
            yield (train, val, test), task_exemplars

    def load_exemplars(self, task_dir):
        """Load the current task exemplars from a given directory."""
        paths = (task_dir / "exemplars").glob("exemplar_*.png")
        return [np.array(Image.open(path)) for path in paths]


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model in a continual learning setting."""
    target = get_object(cfg.model._target_)
    if inspect.isclass(target):
        train_ours(cfg)
    elif callable(target):
        train_baseline(cfg)
    else:
        raise ValueError(f"Unknown target: {target}.")


def train_ours(cfg):
    """Train our model in a continual learning setting."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")

    callbacks = build_callbacks(cfg)
    trainer = instantiate(cfg.trainer, callbacks=callbacks)
    trainer.logger.log_hyperparams(config)

    benchmark = ContinualBenchmarkDisk(cfg.dataset.path)
    model = instantiate(cfg.model)
    for task_id, (datasets, task_exemplars) in enumerate(benchmark):
        if cfg.training.reset_model:
            model = instantiate(cfg.model)
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


def build_callbacks(cfg: DictConfig):
    """Prepare the appropriate callbacks."""
    dataset_path = Path(cfg.dataset.path)
    exemplars = load_exemplars(dataset_path)
    random_images = load_random_images(dataset_path)

    callbacks = []
    callback_names = cfg.training.callbacks
    job_id = os.environ.get("SLURM_JOB_ID")
    if "checkpointing" in callback_names:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(cfg.trainer.default_root_dir) / job_id,
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
        callbacks.append(VisualizationCallback(exemplars, random_images))
    return callbacks


def load_exemplars(path, task=0):
    """Load the current task exemplars from a file."""
    exemplars_dir = path / f"task_{task}/exemplars"
    paths = exemplars_dir.glob("exemplar_*.png")
    return [np.array(Image.open(path)) for path in paths]


def load_random_images(path, num_imgs: int = 25):
    """Load random samples from randomly chosen tasks."""
    task_dirs = list(path.glob("task_*"))
    images = []
    for _ in range(num_imgs):
        val_dir = np.random.choice(task_dirs) / "val"
        image_path = np.random.choice(list(val_dir.glob("*.png")))
        images.append(np.array(Image.open(image_path)))
    return np.stack(images)


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


def train_baseline(cfg, benchmark):
    """Train a standard continual learning baseline using Avalanche."""
    strategy = create_strategy(cfg)
    benchmark = create_benchmark(cfg)

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


def create_benchmark(cfg):
    train_experiences = []
    test_experiences = []

    for task in range(cfg.dataset.tasks):
        task_dir = Path(cfg.dataset.path) / f"task_{task}"
        with open(task_dir / "train/labels.txt") as f:
            train_experience = [tuple(line.split()) for line in f.readlines()]

        with open(task_dir / "test/labels.txt") as f:
            test_experience = [tuple(line.split()) for line in f.readlines()]

        train_experiences.append(train_experience)
        test_experiences.append(test_experience)

    return paths_benchmark(
        train_experiences,
        test_experiences,
        task_labels=[0] * len(train_experiences),
        complete_test_set_only=True,
        train_transform=ToTensor(),
        eval_transform=ToTensor(),
    )


def create_strategy(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = create_evaluator(cfg)
    model = call(cfg.model)

    if cfg.strategy == "icarl":
        strategy = instantiate(
            cfg.strategy,
            device=device,
            evaluator=evaluator,
            optimizer={"params": model.parameters()},
            herding=True,
        )
    else:
        strategy = instantiate(
            cfg.strategy,
            model=model,
            device=device,
            evaluator=evaluator,
            optimizer={"params": model.parameters()},
        )
    return strategy


def create_evaluator(cfg):
    """Create the evaluation plugin."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")
    wandb.run.define_metric("*", step_metric="Step", step_sync=True)

    loggers = [
        WandBLogger(
            dir=cfg.wandb.save_dir,
            project_name=f"{cfg.wandb.project}_baselines",
            params={"group": cfg.wandb.group},
            config=config,
        ),
    ]

    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, experience=True),
        loss_metrics(minibatch=True),
        loggers=loggers,
    )


if __name__ == "__main__":
    train()
