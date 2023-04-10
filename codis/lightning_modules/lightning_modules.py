"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
from torchmetrics import R2Score

from codis.models import MLP, BaseVAE, BetaVAE
from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions

# pylint: disable=arguments-differ,unused-argument


class CodisModel(pl.LightningModule):
    """A model that combines a backbone and a latent regressor."""

    def __init__(
        self,
        backbone: pl.LightningModule,
        regressor: pl.LightningModule,
        freeze_backbone: bool = False,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.freeze()
        self.regressor = regressor
        self.gamma = gamma
        self.r2_score = R2Score()

    def forward(self, x):
        """Perform the forward pass."""
        if not isinstance(self.backbone, BaseVAE):
            return self.regressor(self.backbone(x))
        x_hat, mu, log_var = self.backbone(x)
        return self.regressor(mu), x_hat, mu, log_var

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss, _, _ = self._step(batch)
        self.log_dict({f"{k}_train": v for k, v in loss.items()})
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        return self._step_val_test(batch, "val")

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        return self._step_val_test(batch, "test")

    def _step_val_test(self, batch, suffix):
        loss, y, y_hat = self._step(batch)
        self.log_dict({f"{k}_{suffix}": v for k, v in loss.items()}, on_epoch=True)
        self.r2_score(to_numpy(y), to_numpy(y_hat))
        self.log(f"r2_score_{suffix}", self.r2_score, on_epoch=True)
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y = self._stack_latents(y)
        y_hat, x_hat, mu, log_var = self.forward(x)
        loss = self.regressor.loss_function(y, y_hat)
        backbone_loss = self.backbone.loss_function(x, x_hat, mu, log_var)
        loss["backbone_loss"] = backbone_loss["loss"]
        if not self.freeze_backbone:
            loss["loss"] += self.gamma * backbone_loss["loss"]
        return loss, y, y_hat

    def _stack_latents(self, latents):
        """Stack the latents."""
        return torch.stack(
            [
                latents.color,
                latents.scale,
                latents.orientation,
                latents.position_x,
                latents.position_y,
            ],
            dim=-1,
        )

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
        lr: float = 1e-3,
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
        self.lr = lr

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Perform the forward pass."""
        return self.model(x)

    def loss_function(self, *args, **kwargs):
        """Calculate the loss."""
        return self.model.loss_function(*args, **kwargs)

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
            x, _ = batch
            self._log_reconstructions(x)
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        return self.model.loss_function(x, x_hat, mu, log_var)

    def _log_reconstructions(self, x):
        """Log reconstructions alongside original images."""
        x_hat, _, _ = self.forward(x)
        reconstructions = draw_batch_and_reconstructions(to_numpy(x), to_numpy(x_hat))
        self.logger.log_image("reconstructions", images=[reconstructions])

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LightningMLP(pl.LightningModule):
    """The MLP Lightning module."""

    def __init__(self, dims: List[int], dropout_rate: float = 0.0, lr: float = 1e-3):
        super().__init__()
        self.model = MLP(dims, dropout_rate)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        return self.model(x)

    def loss_function(self, *args, **kwargs):
        """Calculate the loss."""
        return self.model.loss_function(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        x_hat = self.forward(batch)
        loss = self.model.loss_function(batch, x_hat)
        self.log(**{f"{k}_mlp_train": v for k, v in loss.items()})
        return loss["loss"]

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
