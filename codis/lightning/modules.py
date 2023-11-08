"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models import get_model, list_models

from codis.data import Latents
from codis.models import MLP, BetaVAE


class ContinualModule(pl.LightningModule):
    """A base class for continual learning modules."""

    def __init__(
        self,
        lr: float = 1e-3,
        shapes_per_task: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.factor_names = ["scale", "orientation", "position_x", "position_y"]
        self.num_factors = len(self.factor_names)

        self._task_id = None
        self._shapes_per_task = shapes_per_task
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

    def get_current_task_exemplars(self):
        """Get the exemplars for the current task."""
        start = self._shapes_per_task * self.task_id
        end = self._shapes_per_task * (self.task_id + 1)
        return self._buffer[start:end]

    def _stack_factors(self, factors):
        """Stack the factors."""
        return torch.cat(
            [getattr(factors, name).unsqueeze(-1) for name in self.factor_names],
            dim=-1,
        ).float()

    def _unstack_factors(self, stacked_factors):
        """Unstack the factors."""
        return Latents(
            shape=None,
            shape_id=None,
            color=None,
            **{name: stacked_factors[:, i] for i, name in enumerate(self.factor_names)},
        )


class ContrastiveClassifier(ContinualModule):
    """A contrastive classifier based on SimCLR."""

    def __init__(
        self,
        train_iters_per_epoch,
        backbone: str = "resnet18",
        warmup_epochs=10,
        lr=1e-4,
        opt_weight_decay=1e-6,
        loss_temperature=0.5,
        **kwargs,
    ):
        """
        Args:
            backbone: The backbone model.
            warmup_epochs: The number of warmup epochs.
            lr: The learning rate.
            opt_weight_decay: The weight decay for the optimizer.
            loss_temperature: The temperature for the InfoNCE loss.
        """
        super().__init__(**kwargs)
        self.train_iters_per_epoch = train_iters_per_epoch
        if backbone in list_models(module=torchvision.models):
            self.backbone = get_model(backbone, weights=None)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # add a projection
        mlp_dimension = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(mlp_dimension, mlp_dimension), nn.ReLU(), self.backbone.fc
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.backbone(x)

    def info_nce_loss(self, features, labels):
        """Compute the InfoNCE loss.
        Args:
            features: A batch of features.
            labels: A batch of shape labels.
        """
        print(f"Labels before rebalancing: {labels}")
        features, labels = self.balance_batch(features, labels)
        print(f"Labels after rebalancing: {labels}")
        print(torch.bincount(labels))

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat(
            [positives, negatives], dim=1
        )  # first column are the positives
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.hparams.loss_temperature

        return F.cross_entropy(
            logits, labels
        )  # maximise the probability of the positive (class 0)

    def balance_batch(self, features, labels):
        """Balance the batch by undersampling the majority classes."""

        min_examples_per_class = torch.min(torch.bincount(labels)).item()

        indices_per_class = [
            (labels == label).nonzero(as_tuple=False).squeeze()
            for label in labels.unique()
        ]

        balanced_subset_indices = torch.cat(
            [indices[:min_examples_per_class] for indices in indices_per_class]
        )

        return features[balanced_subset_indices], labels[balanced_subset_indices]

    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        params = self.exclude_from_weight_decay(
            self.backbone.named_parameters(), weight_decay=self.hparams.opt_weight_decay
        )

        # optimizer = torch.optim.Adam(parameters, lr=self.hparams.lr)
        optimizer = LARS(params, lr=self.hparams.lr)

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = (
            self.hparams.warmup_epochs * self.train_iters_per_epoch
        )
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0,
        )

        scheduler = {
            "scheduler": linear_warmup_cosine_decay,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def exclude_from_weight_decay(self, named_params, weight_decay, skip_list=None):
        if skip_list is None:
            skip_list = ["bias", "bn"]

        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def _step(self, batch):
        x, y = batch
        y = y.shape_id

        x = self.backbone(x)
        return {"loss": self.info_nce_loss(x, y)}

    @torch.no_grad()
    def classify(self, x):
        """Classify via nearest neighbor in the backbone representation space."""
        x_hat = self.forward(x)
        buffer = torch.stack([torch.from_numpy(img) for img in self._buffer]).to(
            self.device
        )
        buffer = self.forward(buffer)
        distances = F.pairwise_distance(x_hat.unsqueeze(1), buffer.unsqueeze(0))
        return distances.argmin(dim=1)


class SupervisedClassifier(ContinualModule):
    """A supervised classification model."""

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "resnet18",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if backbone in list_models(module=torchvision.models):
            self.backbone = get_model(backbone, weights=None, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(self.backbone.parameters(), lr=self.lr)

    def forward(self, x):
        """Perform the forward pass."""
        return self.backbone(x)

    def _step(self, batch):
        """Perform a training or validation step."""
        x, y = batch
        y = y.shape_id
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, "accuracy": (y_hat.argmax(dim=1) == y).float().mean()}

    @torch.no_grad()
    def classify(self, x: torch.Tensor):
        """Classify the input."""
        return self.forward(x).argmax(dim=1)


class SpatialTransformer(ContinualModule):
    """A model that combines a parameter regressor and differentiable affine transforms."""

    def __init__(
        self,
        backbone: str = "resnet18",
        enc_out_size: int = 64,
        gamma: float = 0.5,
        buffer_chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enc_out_size = enc_out_size
        self.buffer_chunk_size = buffer_chunk_size

        if backbone not in list_models(module=torchvision.models):
            raise ValueError(f"Unknown backbone: {backbone}")

        self.encoder = get_model(backbone, weights=None, num_classes=self.enc_out_size)

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
            "reconstruction_loss": reconstruction_loss,
            "regression_loss": regression_loss,
            "loss": self.gamma * regression_loss
            + (1 - self.gamma) * reconstruction_loss,
        }

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
            dims=[self.backbone.latent_dim, 64, 64, len(self.factor_names)],
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
