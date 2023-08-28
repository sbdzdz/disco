"""Lightning callbacks."""
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(
        self,
        canonical_images,
        random_images,
        num_reconstructions: int = 25,
        num_classifications: int = 12,
    ):
        super().__init__()
        self._canonical_images = canonical_images
        self._random_images = random_images
        self._num_reconstructions = num_reconstructions
        self._num_classifications = num_classifications

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Show the exemplars and the corresponding reconstructions."""
        self.log_reconstructions(
            pl_module,
            self._canonical_images,
            name="reconstructions_canonical",
            num_imgs=self._num_reconstructions,
        )
        self.log_reconstructions(
            pl_module,
            self._random_images,
            name="reconstructions_random",
            num_imgs=self._num_reconstructions,
        )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            self.log_classification(
                pl_module,
                batch,
                name="classification",
                num_imgs=self._num_classifications,
            )

    @staticmethod
    def log_reconstructions(pl_module, x, name, num_imgs):
        """Log images and reconstructions"""
        pl_module.eval()
        x = np.stack(x[:num_imgs])
        x_hat, *_ = pl_module(torch.from_numpy(x).to(pl_module.device))
        images = draw_batch_and_reconstructions(x, to_numpy(x_hat))
        pl_module.logger.log_image(name, images=[images])
        pl_module.train()

    @staticmethod
    def log_classification(pl_module, batch, name, num_imgs):
        pl_module.eval()
        x, y = batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        x = x[:num_imgs]
        x_hat, *_ = pl_module(x)
        labels = pl_module.classify(x)
        closest = np.stack([pl_module._buffer[i] for i in labels])
        actual = np.stack([pl_module._buffer[i] for i in y.shape_id[:num_imgs]])
        images = draw_batch_and_reconstructions(
            to_numpy(x),
            to_numpy(x_hat),
            actual,
            closest,
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
