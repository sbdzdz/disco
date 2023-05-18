"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
from torchmetrics import R2Score

from codis.models import MLP, BetaVAE
from codis.blocks import Encoder
from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions

# pylint: disable=arguments-differ,unused-argument


class SpatialTransformer(pl.LightningModule):
    """A model that combines a parameter regressor and differentiable affine transforms."""

    def __init__(
        self,
        regressor: pl.LightningModule,
        img_size: int = 64,
        in_channels: int = 1,
        num_channels: Optional[list] = None,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [4, 4, 8, 8, 16]
        self.encoder = Encoder(img_size, in_channels, num_channels)

    @property
    def train_task_id(self):
        """Get the current train task id."""
        return self._train_task_id

    @train_task_id.setter
    def train_task_id(self, value):
        """Set the current train task id."""
        self._train_task_id = value

    @property
    def test_task_id(self):
        """Get the current test task id."""
        return self._test_task_id

    @test_task_id.setter
    def test_task_id(self, value):
        """Set the current test task id."""
        self._test_task_id = value

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=self.encoder.lr,
        )

    def forward(self, x):
        """Perform the forward pass."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        raise NotImplementedError


class SupervisedVAE(pl.LightningModule):
    """A model that combines a VAE backbone and an MLP regressor."""

    def __init__(
        self,
        vae: pl.LightningModule,
        regressor: pl.LightningModule = None,
        gamma: float = 0.5,
        factors_to_regress: list = None,
    ):
        super().__init__()
        self.backbone = vae
        self.gamma = gamma
        if factors_to_regress is None:
            factors_to_regress = ["scale", "orientation", "position_x", "position_y"]
        self.factors_to_regress = factors_to_regress
        if regressor is None:
            regressor = LightningMLP(
                dims=[self.backbone.latent_dim, 64, 64, len(factors_to_regress)],
            )
        self.regressor = regressor
        self.r2_score = R2Score(num_outputs=self.regressor.model.dims[-1])

    @property
    def train_task_id(self):
        """Get the current train task id."""
        return self._train_task_id

    @train_task_id.setter
    def train_task_id(self, value):
        """Set the current train task id."""
        self._train_task_id = value

    @property
    def test_task_id(self):
        """Get the current test task id."""
        return self._test_task_id

    @test_task_id.setter
    def test_task_id(self, value):
        """Set the current test task id."""
        self._test_task_id = value

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": self.backbone.lr},
                {"params": self.regressor.parameters(), "lr": self.regressor.lr},
            ]
        )

    def forward(self, x):
        """Perform the forward pass."""
        x_hat, mu, log_var = self.backbone(x)
        return x_hat, mu, log_var, self.regressor(mu)

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss, _ = self._step(batch)
        self.log_dict({f"{k}_train": v for k, v in loss.items()})
        self.log("task_id", float(self.train_task_id))
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss, metrics = self._step(batch)
        self.log_dict({f"{k}_val": v for k, v in loss.items()}, on_epoch=True)
        self.log("r2_score_val", metrics["r2_score"], on_epoch=True)
        return loss["loss"]

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        loss, metrics = self._step(batch)
        self.log_dict(
            {f"{k}_test_task_{self.test_task_id}": v for k, v in loss.items()},
            on_epoch=True,
        )
        self.log_dict(
            {f"{k}_test_task_{self.test_task_id}": v for k, v in metrics.items()},
            on_epoch=True,
        )
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y = self._stack_factors(y)
        x_hat, mu, log_var, y_hat = self.forward(x)
        metrics = self._calculate_metrics(y, y_hat)
        regressor_loss = self.regressor.loss_function(y, y_hat)
        backbone_loss = self.backbone.loss_function(x, x_hat, mu, log_var)
        loss = {
            "regressor_loss": regressor_loss["loss"],
            "backbone_loss": backbone_loss["loss"],
        }
        loss["loss"] = (
            self.gamma * loss["regressor_loss"]
            + (1 - self.gamma) * loss["backbone_loss"]
        )
        return loss, metrics

    def _calculate_metrics(self, y, y_hat):
        unstacked_y = self._unstack_factors(y)
        unstacked_y_hat = self._unstack_factors(y_hat)
        return {
            "r2_score": self.r2_score(y_hat, y),
            **{
                f"r2_score_{name}": self.r2_score(
                    unstacked_y[name], unstacked_y_hat[name]
                )
                for name in self.factors_to_regress
            },
        }

    def _stack_factors(self, factors):
        """Stack the factors."""
        return torch.cat(
            [getattr(factors, name).unsqueeze(-1) for name in self.factors_to_regress],
            dim=-1,
        ).float()

    def _unstack_factors(self, stacked_factors):
        """Unstack the factors."""
        return {
            name: stacked_factors[:, i]
            for i, name in enumerate(self.factors_to_regress)
        }


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

    @property
    def latent_dim(self):
        """Dimensionality of the latent space."""
        return self.model.latent_dim

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
