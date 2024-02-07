"""Training script."""

import inspect
import os
import random
from pathlib import Path
from time import time
from typing import Union

import hydra
import idsprites as ids
import numpy as np
import torch
import wandb
from avalanche.benchmarks.generators import paths_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.training.plugins import EvaluationPlugin
from hydra.utils import call, get_object, instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

from disco.lightning.callbacks import (
    LoggingCallback,
    MetricsCallback,
    VisualizationCallback,
)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

OmegaConf.register_new_resolver("eval", eval)


class FileDataset(Dataset):
    def __init__(self, path: Union[Path, str], transform=None, target_transform=None):
        self.path = Path(path)
        self.transform = transform
        self.target_transform = target_transform
        factors = np.load(self.path / "factors.npz", allow_pickle=True)
        factors = [
            dict(zip(factors, value)) for value in zip(*factors.values())
        ]  # turn dict of lists into list of dicts
        self.data = [ids.Factors(**factors) for factors in factors]
        self.shapes = np.load(self.path / "../shapes.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.path / f"sample_{idx}.png"
        image = read_image(str(img_path)) / 255.0

        factors = self.data[idx]
        factors = factors.replace(
            shape=self.shapes[factors.shape_id % len(self.shapes)]
        )
        factors = factors.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            factors = self.target_transform(factors)
        return image, factors


class ContinualBenchmarkDisk:
    def __init__(
        self,
        path: Union[Path, str],
        accumulate_test_set: bool = True,
        test_subset: int = 100,
    ):
        """Initialize the continual learning benchmark.
        Args:
            path: The path to the dataset.
            accumulate_test_set: Whether to accumulate the test set over tasks.
        """
        self.path = Path(path)
        self.accumulate_test_set = accumulate_test_set
        self.test_subset = test_subset
        if self.accumulate_test_set:
            self.test_sets = []

    def __iter__(self):
        for task_dir in sorted(
            self.path.glob("task_*"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            task_exemplars = self.load_exemplars(task_dir)
            train = FileDataset(task_dir / "train")
            val = FileDataset(task_dir / "val")
            test = FileDataset(task_dir / "test")

            if self.accumulate_test_set:
                test = Subset(
                    test, np.random.choice(len(test), self.test_subset, replace=False)
                )
                self.test_sets.append(test)
                accumulated_test = ConcatDataset(self.test_sets)
                yield (train, val, accumulated_test), task_exemplars
            else:
                yield (train, val, test), task_exemplars

    def load_exemplars(self, task_dir):
        """Load the current task exemplars from a given directory."""
        paths = (task_dir / "exemplars").glob("exemplar_*.png")
        return [read_img_to_np(path) for path in paths]


def read_img_to_np(path: Union[Path, str]):
    """Read an image and normalize it to [0, 1].
    Args:
        path: The path to the image.
    Returns:
        The image as a numpy array.
    """
    return np.array(read_image(str(path)) / 255.0)


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

    benchmark = ContinualBenchmarkDisk(
        path=cfg.dataset.path,
        accumulate_test_set=cfg.dataset.accumulate_test_set,
        test_subset=cfg.dataset.test_subset,
    )
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
    return [read_img_to_np(path) for path in paths]


def load_random_images(path, num_imgs: int = 25):
    """Load random samples from randomly chosen tasks."""
    task_dirs = list(path.glob("task_*"))
    images = []
    for _ in range(num_imgs):
        val_dir = np.random.choice(task_dirs) / "val"
        image_path = np.random.choice(list(val_dir.glob("*.png")))
        images.append(read_img_to_np(image_path))
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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,  # shuffle for vis
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_baseline(cfg):
    """Train a standard continual learning baseline using Avalanche."""
    setup_wandb(cfg)
    benchmark = create_benchmark(cfg)
    strategy = create_strategy(cfg)

    results = []
    for task, train_experience in enumerate(benchmark.train_stream):
        log_message(train_experience, "train")
        train_start = time()
        strategy.train(train_experience, num_workers=cfg.dataset.num_workers)
        train_end = time()
        wandb.log({"train/time_per_task": (train_end - train_start) / 60})

        if not cfg.training.test_once and task % cfg.training.test_every_n_tasks == 0:
            test_start = time()
            result = strategy.eval(
                benchmark.test_stream[: task + 1],
                num_workers=cfg.dataset.num_workers,
            )
            test_end = time()
            wandb.log({"test/time_per_task": (test_end - test_start) / 60})
            results.append(result)
            log_metrics(task, result)
    wandb.finish()


def setup_wandb(cfg):
    """Set up wandb logging."""
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config["job_id"] = os.environ.get("SLURM_JOB_ID")
    wandb.init(project=cfg.wandb.project, group=cfg.wandb.group, config=config)
    wandb.define_metric("task")
    wandb.define_metric("test/accuracy", step_metric="task")
    wandb.define_metric("test/forgetting", step_metric="task")
    wandb.define_metric("train/time_per_task", step_metric="task")
    wandb.define_metric("test/time_per_task", step_metric="task")


def create_benchmark(cfg):
    train_experiences = []
    test_experiences = []

    for task in range(cfg.dataset.tasks):
        task_dir = Path(cfg.dataset.path) / f"task_{task+1}"
        with open(task_dir / "train/labels.txt") as f:
            train_experience = [
                (task_dir / "train" / parts[0], int(parts[1]))
                for parts in (line.strip().split(maxsplit=1) for line in f)
            ]

        with open(task_dir / "test/labels.txt") as f:
            test_experience = [
                (task_dir / "test" / parts[0], int(parts[1]))
                for parts in (line.strip().split(maxsplit=1) for line in f)
            ]
        if cfg.dataset.test_subset is not None:
            test_experience = random.sample(
                test_experience, cfg.dataset.test_subset
            ).tolist()

        train_experiences.append(train_experience)
        test_experiences.append(test_experience)

    return paths_benchmark(
        train_experiences,
        test_experiences,
        task_labels=[0] * len(train_experiences),
        train_transform=ToTensor(),
        eval_transform=ToTensor(),
    )


def create_strategy(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=False, stream=True),
        forgetting_metrics(experience=False, stream=True),
    )
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


def log_message(experience, stage):
    """Log the length of the dataset and the classes in the current experience."""
    print(
        f"Task {experience.current_experience} {stage}: {len(experience.dataset)} samples."
    )
    min_class_id = min(experience.classes_in_this_experience)
    max_class_id = max(experience.classes_in_this_experience)
    print(f"Classes: {min_class_id}-{max_class_id}")


def log_metrics(task, result):
    """Log the task and test accuracy to wandb."""
    wandb.log(
        {
            "task": task,
            "test/accuracy": result["Top1_Acc_Stream/eval_phase/test_stream/Task000"],
            "test/forgetting": result["StreamForgetting/eval_phase/test_stream"],
        }
    )


if __name__ == "__main__":
    train()
