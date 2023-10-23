"""Lightning callbacks."""
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from codis.visualization import draw_batch, draw_batch_and_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(
        self,
        canonical_images,
        random_images,
        num_reconstructions: int = 20,
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
        current_task_exemplars = pl_module.get_current_task_exemplars()
        self.log_batch(pl_module, np.array(current_task_exemplars))

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
    @torch.no_grad()
    def log_reconstructions(pl_module, batch, name, num_imgs):
        """Log images and reconstructions"""
        batch = np.stack(batch[:num_imgs])
        x_hat, *_ = pl_module(torch.from_numpy(batch).to(pl_module.device))
        images = draw_batch_and_reconstructions(
            batch, x_hat.detach().cpu().numpy(), save=False
        )
        pl_module.logger.log_image(name, images=[images])

    @staticmethod
    @torch.no_grad()
    def log_batch(pl_module, batch):
        images = draw_batch(np.array(batch), save=False)
        pl_module.logger.log_image("current_task_exemplars", images=[images])

    @staticmethod
    @torch.no_grad()
    def log_classification(pl_module, batch, name, num_imgs):
        x, y = batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        x = x[:num_imgs]
        x_hat, *_ = pl_module(x)
        labels = pl_module.classify(x)
        closest = np.stack([pl_module._buffer[i] for i in labels])
        actual = np.stack([pl_module._buffer[i] for i in y.shape_id[:num_imgs]])
        images = draw_batch_and_reconstructions(
            x.detach().cpu().numpy(),
            x_hat.detach().cpu().numpy(),
            closest,
            actual,
            save=False,
        )
        pl_module.logger.log_image(name, images=[images])


class LoggingCallback(Callback):
    """Callback for additional logging."""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.log("task_id", float(pl_module.task_id))

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(
            f"Task {pl_module.task_id} training: "
            f"{len(trainer.train_dataloader)} batches, "
            f"{len(trainer.train_dataloader.dataset)} samples."
        )

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(
            f"Task {pl_module.task_id} validation: "
            f"{len(trainer.val_dataloaders)} batches, "
            f"{len(trainer.val_dataloaders.dataset)} samples."
        )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(
            f"Task {pl_module.task_id} testing: "
            f"{len(trainer.test_dataloaders)} batches, "
            f"{len(trainer.test_dataloaders.dataset)} samples."
        )


class MetricsCallback(Callback):
    """Callback for logging metrics."""

    def __init__(
        self, log_train_accuracy: bool = False, log_val_accuracy: bool = False
    ):
        super().__init__()
        self.log_train_accuracy = log_train_accuracy
        self.log_val_accuracy = log_val_accuracy

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the training loss."""
        if self.log_train_accuracy:
            self._log_accuracy(batch, pl_module, "train")
        pl_module.log_dict({f"train/{k}": v.item() for k, v in outputs.items()})

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the validation loss."""
        if self.log_val_accuracy:
            self._log_accuracy(batch, pl_module, "val")
        pl_module.log_dict({f"val/{k}": v.item() for k, v in outputs.items()})

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the test loss."""
        self._log_accuracy(batch, pl_module, "test")
        pl_module.log_dict(
            {f"test/{k}": v.item() for k, v in outputs.items()},
        )

    def _log_accuracy(self, batch, pl_module, stage):
        x, y = batch
        accuracy = (y.shape_id == pl_module.classify(x)).float().mean().item()
        pl_module.log_dict({f"{stage}/accuracy": accuracy})
