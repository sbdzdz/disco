"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import R2Score

from codis.models import MLP, BetaVAE
from codis.models.blocks import Encoder
from codis.utils import to_numpy
from codis.data import Latents
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

    def classify(self, x):
        """Classify the input."""
        x_hat, *_ = self(x)
        exemplars = self._buffer
        exemplars = torch.stack(exemplars).to(self.device)
        mse = F.mse_loss(x_hat.unsqueeze(1), exemplars, reduction="none")
        mse = mse.mean(dim=(2, 3, 4))
        return mse.argmin(dim=1)

    def _stack_factors(self, factors):
        """Stack the factors."""
        return torch.cat(
            [getattr(factors, name).unsqueeze(-1) for name in self.factors_to_regress],
            dim=-1,
        ).float()

    def _unstack_factors(self, stacked_factors):
        """Unstack the factors."""
        return Latents(
            shape=None,
            shape_id=None,
            color=None,
            **{
                name: stacked_factors[:, i]
                for i, name in enumerate(self.factors_to_regress)
            },
        )


class SpatialTransformer(ContinualModule):
    """A model that combines a parameter regressor and differentiable affine transforms."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        channels: Optional[list] = None,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_buffer = True

        if channels is None:
            channels = [16, 16, 32, 32, 64]
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
        self.gamma = gamma

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

        grid = F.affine_grid(theta, x.size())
        x_hat = F.grid_sample(x, grid, padding_mode="border")

        return x_hat, theta

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_train": v for k, v in loss.items()})
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_val": v for k, v in loss.items()})
        return loss["loss"]

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_test_task_{self.task_id}": v for k, v in loss.items()})
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        x_hat, theta_hat = self.forward(x)
        exemplars = torch.stack([self._buffer[i] for i in y.shape_id])
        theta = self.convert_parameters_to_matrix(y)
        regression_loss = F.mse_loss(theta, theta_hat)
        backbone_loss = F.mse_loss(exemplars, x_hat)
        return {
            "regression_loss": regression_loss,
            "backbone_loss": backbone_loss,
            "loss": self.gamma * regression_loss + (1 - self.gamma) * backbone_loss,
        }

    def convert_parameters_to_matrix(self, factors):
        """Convert the ground truth factors to a transformation matrix.
        The matrix maps from an arbitrary image (defined by factors) to the canonical representation.
        Args:
            factors: A namedtuple of factors.
        Returns:
            A 2x3 transformation matrix.
        """
        scale, orientation, position_x, position_y = (
            factors.scale,
            factors.orientation,
            factors.position_x,
            factors.position_y,
        )
        batch_size = scale.shape[0]

        transform_matrix = self.batched_eye(batch_size)

        # scale
        scale_matrix = self.batched_eye(batch_size)
        scale_matrix[:, 0, 0] = scale
        scale_matrix[:, 1, 1] = scale
        transform_matrix = torch.bmm(scale_matrix, transform_matrix)

        # rotate
        orientation_matrix = self.batched_eye(batch_size)
        orientation_matrix[:, 0, 0] = torch.cos(orientation)
        orientation_matrix[:, 0, 1] = -torch.sin(orientation)
        orientation_matrix[:, 1, 0] = torch.sin(orientation)
        orientation_matrix[:, 1, 1] = torch.cos(orientation)
        transform_matrix = torch.bmm(orientation_matrix, transform_matrix)

        # move to the center
        translation_matrix = self.batched_eye(batch_size)
        translation_matrix[:, 0, 2] = position_x - 0.5
        translation_matrix[:, 1, 2] = position_y - 0.5
        transform_matrix = torch.bmm(translation_matrix, transform_matrix)

        return transform_matrix[:, :2, :]

    def batched_eye(self, batch_size):
        """Create a batch of identity matrices."""
        return (
            torch.eye(3, dtype=torch.float)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(self.device)
        )


class SpatialTransformerGF(SpatialTransformer):
    """A SpatialTransformer that predicts the generative factors instead of a transformation matrix."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regressor = MLP(
            dims=[self.encoder_output_dim, 64, 32, self.num_factors],
        )

    def forward(self, x):
        """Perform the forward pass."""
        xs = self.encoder(x).view(-1, self.encoder_output_dim)
        y_hat = self.regressor(xs)
        theta = self.convert_parameters_to_matrix(self._unstack_factors(y_hat))

        grid = F.affine_grid(theta, x.size())
        x_hat = F.grid_sample(x, grid, padding_mode="border")
        return x_hat, y_hat

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        x_hat, y_hat = self.forward(x)
        exemplar_tiled = (
            self._buffer[self.task_id].repeat(x.shape[0], 1, 1, 1).to(self.device)
        )
        regression_loss = F.mse_loss(self._stack_factors(y), y_hat)
        backbone_loss = F.mse_loss(exemplar_tiled, x_hat)
        y_hat = self._unstack_factors(y_hat)
        return {
            "regression_loss": regression_loss,
            "backbone_loss": backbone_loss,
            "orientation_loss": F.mse_loss(y.orientation, y_hat.orientation),
            "scale_loss": F.mse_loss(y.scale, y_hat.scale),
            "position_loss": F.mse_loss(
                torch.stack([y.position_x, y.position_y], dim=1),
                torch.stack([y_hat.position_x, y_hat.position_y], dim=1),
            ),
            "loss": self.gamma * regression_loss + (1 - self.gamma) * backbone_loss,
        }


class SupervisedVAE(ContinualModule):
    """A model that combines a VAE backbone and an MLP regressor."""

    def __init__(
        self,
        vae: pl.LightningModule,
        regressor: pl.LightningModule = None,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        regressor_loss = self.regressor.loss_function(y, y_hat)["loss"]
        backbone_loss = self.backbone.loss_function(x, x_hat, mu, log_var)["loss"]
        loss = {
            "regression_loss": regressor_loss,
            "backbone_loss": backbone_loss,
            "loss": self.gamma * regressor_loss + (1 - self.gamma) * backbone_loss,
        }
        return loss, metrics

    def _calculate_metrics(self, y, y_hat):
        y = self._unstack_factors(y)
        y_hat = self._unstack_factors(y_hat)
        return {
            "r2_score": self.r2_score(y, y_hat),
            "r2_score_orientation": self.r2_score(y.orientation, y_hat.orientation),
            "r2_score_scale": self.r2_score(y.scale, y_hat.scale),
            "r2_score_position_x": self.r2_score(y.position_x, y_hat.position_x),
            "r2_score_position_y": self.r2_score(y.position_y, y_hat.position_y),
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
