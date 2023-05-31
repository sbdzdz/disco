"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import R2Score

from codis.models import MLP, BetaVAE
from codis.models.blocks import Encoder
from codis.utils import to_numpy
from codis.visualization import draw_batch_and_reconstructions

# pylint: disable=arguments-differ,unused-argument,too-many-ancestors


class ContinualModule(pl.LightningModule):
    """A base class for continual learning modules."""

    def __init__(self, lr: float = 1e-3, factors_to_regress: list = None):
        super().__init__()
        self.lr = lr
        if factors_to_regress is None:
            factors_to_regress = ["scale", "orientation", "position_x", "position_y"]
        self.factors_to_regress = factors_to_regress
        self.num_factors = len(self.factors_to_regress)
        self._task_id = None
        self.has_buffer = False

    @property
    def task_id(self):
        """Get the current train task id."""
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        """Set the current train task id."""
        self._task_id = value

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


class LatentRegressor(ContinualModule):
    """A model that directly regresses the latent factors."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        channels: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if channels is None:
            channels = [8, 8, 16, 16, 32]
        self.encoder = Encoder(channels, in_channels)
        self.encoder_output_dim = (img_size // 2 ** len(channels)) ** 2 * channels[-1]

        self.regressor = MLP(
            dims=[self.encoder_output_dim, 64, 32, self.num_factors],
        )

    def forward(self, x):
        """Perform the forward pass."""
        x = self.encoder(x).view(-1, self.encoder_output_dim)
        return self.regressor(x)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": self.lr},
                {"params": self.regressor.parameters(), "lr": self.lr},
            ]
        )

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss = self._step(batch)
        self.log("loss_train", loss)

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss = self._step(batch)
        self.log("loss_val", loss)

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        loss = self._step(batch)
        self.log(f"loss_task_{self._task_id}", loss)

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y = self._stack_factors(y)
        y_hat = self.forward(x)
        return F.mse_loss(y, y_hat)


class SpatialTransformer(ContinualModule):
    """A model that combines a parameter regressor and differentiable affine transforms."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        channels: Optional[list] = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_buffer = True

        if channels is None:
            channels = [8, 8, 16, 16, 32]
        self.encoder = Encoder(channels, in_channels)
        self.encoder_output_dim = (img_size // 2 ** len(channels)) ** 2 * channels[-1]

        # build the regressor and initialize its weights to the identity transform
        self.regressor = MLP(
            dims=[self.encoder_output_dim, 64, 32, 6],
        )
        self.regressor.model[-1].weight.data.zero_()
        self.regressor.model[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        self.mask = torch.tensor([1, 1, 1, 1, 1, 1]) if mask is None else mask

        self._buffer = []

    def add_exemplar(self, exemplar):
        """Add an exemplar to the buffer."""
        self._buffer.append(exemplar)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": self.lr},
                {"params": self.regressor.parameters(), "lr": self.lr},
            ]
        )

    def forward(self, x):
        """Perform the forward pass."""
        xs = self.encoder(x).view(-1, self.encoder_output_dim)
        theta = self.regressor(xs).view(-1, 2, 3)
        theta = theta * self.mask.view(-1, 2, 3).to(self.device)

        grid = F.affine_grid(theta, x.size())
        x_hat = F.grid_sample(x, grid)

        return x_hat, theta

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        return self._step(batch, "test")

    def _step(self, batch, stage):
        """Perform a training or validation step."""
        x, _ = batch
        x_hat, _ = self.forward(x)
        exemplar_tiled = (
            self._buffer[self.task_id].repeat(x.shape[0], 1, 1, 1).to(self.device)
        )
        loss = F.mse_loss(x_hat, exemplar_tiled)
        if stage == "train":
            self.log(f"loss_{stage}", loss, on_step=True)
        elif stage == "val":
            self.log(f"loss_{stage}", loss, on_epoch=True)
        else:
            self.log(f"loss_{stage}_task_{self.task_id}", loss, on_epoch=True)
        return loss


class SupervisedVAE(ContinualModule):
    """A model that combines a VAE backbone and an MLP regressor."""

    def __init__(
        self,
        vae: pl.LightningModule,
        regressor: pl.LightningModule = None,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.backbone = vae
        self.gamma = gamma
        if regressor is None:
            regressor = LightningMLP(
                dims=[self.backbone.latent_dim, 64, 64, len(self.factors_to_regress)],
            )
        self.regressor = regressor
        self.r2_score = R2Score(num_outputs=self.regressor.model.dims[-1])
        self.has_buffer = False

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
            {f"{k}_test_task_{self.task_id}": v for k, v in loss.items()},
            on_epoch=True,
        )
        self.log_dict(
            {f"{k}_test_task_{self.task_id}": v for k, v in metrics.items()},
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


class LightningBetaVAE(pl.LightningModule):
    """The β-VAE Lightning module."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        latent_dim: int = 10,
        channels: Optional[list] = None,
        beta: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = BetaVAE(
            img_size,
            in_channels,
            channels,
            latent_dim,
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
