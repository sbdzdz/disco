"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import get_model, list_models

from codis.data import Latents
from codis.models import MLP, BetaVAE
from codis.models.blocks import Encoder


class ContinualModule(pl.LightningModule):
    """A base class for continual learning modules."""

    def __init__(
        self,
        lr: float = 1e-3,
        factors_to_regress: list = None,
        buffer_chunk_size: int = 64,
    ):
        super().__init__()
        self.lr = lr
        if factors_to_regress is None:
            factors_to_regress = ["scale", "orientation", "position_x", "position_y"]
        self.factors_to_regress = factors_to_regress
        self.buffer_chunk_size = buffer_chunk_size
        self.num_factors = len(self.factors_to_regress)

        self._task_id = None
        self._buffer = []

    @property
    def task_id(self):
        """Get the current train task id."""
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        """Set the current train task id."""
        self._task_id = value

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self._step(batch)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        return self._step(batch)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        return self._step(batch)

    def add_exemplar(self, exemplar):
        """Add an exemplar to the buffer."""
        self._buffer.append(exemplar)

    @torch.no_grad()
    def classify(self, x: torch.Tensor):
        """Classify the input."""
        x_hat, *_ = self(x)
        x_hat = x_hat.unsqueeze(1).detach()
        buffer = torch.stack([torch.from_numpy(img) for img in self._buffer]).to(
            self.device
        )
        buffer = buffer.unsqueeze(0).detach()

        losses = []  # classify in chunks to avoid OOM
        for chunk in torch.split(buffer, self.buffer_chunk_size, dim=1):
            chunk = chunk.repeat(x_hat.shape[0], 1, 1, 1, 1)
            loss = F.mse_loss(
                x_hat.repeat(1, chunk.shape[1], 1, 1, 1), chunk, reduction="none"
            ).mean(dim=(2, 3, 4))
            losses.append(loss)
        return torch.cat(losses, dim=1).argmin(dim=1)

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
        encoder: str = "simple_cnn",
        channels: Optional[list] = None,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if channels is None:
            channels = [16, 16, 32, 32, 64]

        if encoder == "simple_cnn":
            self.encoder = Encoder(channels, in_channels)
            self.enc_out_size = self.encoder.out_size(img_size)
        elif encoder in list_models(module=torchvision.models):
            self.encoder = get_model(encoder, weights=None, num_classes=channels[-1])
            self.enc_out_size = channels[-1]
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        # initialize the regressor to the identity transform
        self.regressor = MLP(dims=[self.enc_out_size, 64, 32, 6])
        self.regressor.model[-1].weight.data.zero_()
        self.regressor.model[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        self.gamma = gamma

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
        xs = self.encoder(x).view(-1, self.enc_out_size)
        theta = self.regressor(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_hat = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

        return x_hat, theta

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        x_hat, theta_hat = self.forward(x)
        exemplars = torch.stack(
            [torch.from_numpy(self._buffer[i]) for i in y.shape_id]
        ).to(self.device)
        theta = self.convert_parameters_to_matrix(y)
        regression_loss = F.mse_loss(theta, theta_hat)
        reconstruction_loss = F.mse_loss(exemplars, x_hat)
        return {
            "reconstruction_loss": reconstruction_loss.item(),
            "regression_loss": regression_loss.item(),
            "loss": self.gamma * regression_loss
            + (1 - self.gamma) * reconstruction_loss,
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
            dims=[self.enc_out_size, 64, 32, self.num_factors],
        )

    def forward(self, x):
        """Perform the forward pass."""
        xs = self.encoder(x).view(-1, self.enc_out_size)
        y_hat = self.regressor(xs)
        theta = self.convert_parameters_to_matrix(self._unstack_factors(y_hat))

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_hat = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
        return x_hat, y_hat

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        x_hat, y_hat = self.forward(x)
        exemplars = torch.stack(
            [torch.from_numpy(self._buffer[i]) for i in y.shape_id]
        ).to(self.device)
        regression_loss = F.mse_loss(self._stack_factors(y), y_hat)
        reconstruction_loss = F.mse_loss(exemplars, x_hat)
        y_hat = self._unstack_factors(y_hat)
        orientation_loss = F.mse_loss(y.orientation, y_hat.orientation)
        scale_loss = F.mse_loss(y.scale, y_hat.scale)
        position_loss = F.mse_loss(
            torch.stack([y.position_x, y.position_y], dim=1),
            torch.stack([y_hat.position_x, y_hat.position_y], dim=1),
        )
        accuracy = (y.shape_id == self.classify(x)).float().mean()
        return {
            "accuracy": accuracy.item(),
            "orientation_loss": orientation_loss.item(),
            "position_loss": position_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "regression_loss": regression_loss.item(),
            "scale_loss": scale_loss.item(),
            "loss": self.gamma * regression_loss
            + (1 - self.gamma) * reconstruction_loss,
        }


class SupervisedVAE(ContinualModule):
    """A model that combines a VAE backbone and an MLP regressor."""

    def __init__(
        self,
        vae: pl.LightningModule,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = vae
        self.gamma = gamma
        self.regressor = LightningMLP(
            dims=[self.backbone.latent_dim, 64, 64, len(self.factors_to_regress)],
        )

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

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y = self._stack_factors(y)
        x_hat, mu, log_var, y_hat = self.forward(x)
        regressor_loss = self.regressor.loss_function(y, y_hat)["loss"]
        vae_loss = self.backbone.loss_function(x, x_hat, mu, log_var)["loss"]
        accuracy = (y.shape_id == self.classify(x)).float().mean()
        return {
            "accuracy": accuracy.item(),
            "regression_loss": regressor_loss.item(),
            "vae_loss": vae_loss.item(),
            "loss": self.gamma * regressor_loss + (1 - self.gamma) * vae_loss,
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

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Perform the forward pass."""
        return self.model(x)

    def _step(self, batch):
        """Perform a training or validation step."""
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        return self.model.loss_function(x, x_hat, mu, log_var)


class LightningMLP(pl.LightningModule):
    """The MLP Lightning module."""

    def __init__(self, dims: List[int], dropout_rate: float = 0.0, lr: float = 1e-3):
        super().__init__()
        self.model = MLP(dims, dropout_rate)
        self.lr = lr

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
