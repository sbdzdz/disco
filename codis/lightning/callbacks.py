"""Lightning callbacks."""
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self, canonical_images, random_images):
        super().__init__()
        self._canonical_images = canonical_images
        self._random_images = random_images

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Show the exemplars and the corresponding reconstructions."""
        self.log_reconstructions(
            pl_module, self._canonical_images, "reconstructions_canonical"
        )
        self.log_reconstructions(
            pl_module, self._random_images, "reconstructions_random"
        )

    @staticmethod
    def log_reconstructions(pl_module, x, name, max_imgs=25):
        """Log images and reconstructions"""
        pl_module.eval()
        x_hat, *_ = pl_module(x.to(pl_module.device))
        images = draw_batch_and_reconstructions(
            to_numpy(x[:max_imgs]), to_numpy(x_hat[:max_imgs])
        )
        pl_module.logger.log_image(name, images=[images])
        pl_module.train()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            self.log_classification(pl_module, batch, "classification")

    @staticmethod
    def log_classification(pl_module, batch, name, max_imgs=25):
        x, y = batch
        pl_module.eval()
        labels = pl_module.classify(x)
        closest = torch.stack([pl_module._buffer[i] for i in labels])
        actual = torch.stack([pl_module._buffer[i] for i in y.shape_id])
        images = draw_batch_and_reconstructions(
            to_numpy(x[:max_imgs]),
            to_numpy(actual[:max_imgs]),
            to_numpy(closest[:max_imgs]),
        )
        pl_module.logger.log_image(name, images=[images])
        pl_module.train()


class LoggingCallback(Callback):
    """Callback for additional logging."""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.log("task_id", pl_module.task_id)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"Training on task {pl_module.task_id},")
        print(f"Number of batches: {len(trainer.train_dataloader)}.")
        print(f"Number of samples: {len(trainer.train_dataloader.dataset)}.")

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"Validating on task {pl_module.task_id},")
        print(f"Number of batches: {len(trainer.val_dataloaders)}.")
        print(f"Number of samples: {len(trainer.val_dataloaders.dataset)}.")

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"Testing on task {pl_module.task_id},")
        print(f"Number of batches: {len(trainer.test_dataloaders)}.")
        print(f"Number of samples: {len(trainer.test_dataloaders.dataset)}.")
