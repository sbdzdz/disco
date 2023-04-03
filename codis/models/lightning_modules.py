"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
from torchmetrics import R2Score

import wandb
from codis.models import MLP, BaseVAE, BetaVAE
from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions

# pylint: disable=arguments-differ,unused-argument


class CodisModel(pl.LightningModule):
    """A continual disentanglement model that combines a pre-trained feature extractor and a latent regressor."""

    def __init__(
        self,
        backbone: pl.LightningModule,
        regressor: pl.LightningModule,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze()
        self.regressor = regressor
        self.r2_score = R2Score()

    def forward(self, x):
        """Perform the forward pass."""
        if not isinstance(self.backbone, BaseVAE):
            return self.regressor(self.backbone(x))
        _, mu, _ = self.backbone(x)
        return self.regressor(mu)

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss, _ = self._step(batch)
        self.log_dict({f"{k}_train": v for k, v in loss.items()})
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss, y_hat = self._step(batch)
        self.log_dict({f"{k}_val": v for k, v in loss.items()})
        self.r2_score(to_numpy(y_hat), to_numpy(batch[1]))
        self.log("R2_score", self.r2_score)
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.regressor.loss_function(y, y_hat)
        return loss, y_hat

    def configure_optimizers(self):
        """Configure the optimizers."""
        return self.regressor.configure_optimizers()


class LightningBetaVAE(pl.LightningModule):
    """The β-VAE Lightning module."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        latent_dim: int = 10,
        num_channels: Optional[list] = None,
        beta: float = 1.0,
    ):
        super().__init__()
        self.model = BetaVAE(
            img_size,
            in_channels,
            latent_dim,
            num_channels,
            beta,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Perform the forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # sourcery skip: class-extract-method
        """Perform a training step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_vae_train": v for k, v in loss.items()})
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_vae_val": v for k, v in loss.items()})
        if batch_idx == 0:
            self._log_reconstructions(batch)
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, _ = batch
        x_hat, mu, log_var = self(x)
        return self.model.loss_function(x, x_hat, mu, log_var)

    def _log_reconstructions(self, x):
        """Log reconstructions alongside original images."""
        x_hat, _, _ = self(x)
        self.logger.log_image(
            {
                "reconstruction": wandb.Image(
                    draw_batch_and_reconstructions(to_numpy(x), to_numpy(x_hat))
                )
            },
        )

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class LightningMLP(pl.LightningModule):
    """The MLP Lightning module."""

    def __init__(self, dims: List[int], dropout_rate: float = 0.0):
        super().__init__()
        self.model = MLP(dims, dropout_rate)

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        _, x = batch
        x_hat = self.model(x)
        loss = self.model.loss_function(x, x_hat)
        self.log(**{f"{k}_mlp_train": v for k, v in loss.items()})
        return loss["loss"]

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
