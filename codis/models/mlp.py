"""A simple multi-layer perceptron.""" ""
import lightning.pytorch as pl
import torch
from torch import nn


class LigthningMLP(pl.LightningModule):
    """The MLP Lightning module."""

    def __init__(self):
        super().__init__()
        self.model = MLP()

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


class MLP(nn.Module):
    """A simple multi-layer perceptron.
    Args:
        dims: A list of dimensions for each layer, including input and output.
        dropout_rate: The dropout rate.
    """

    def __init__(self, dims, dropout_rate=0.0):
        super().__init__()
        module = [
            nn.Sequential(
                nn.Linear(n_in, n_out), nn.LeakyReLU(), nn.Dropout(dropout_rate)
            )
            for n_in, n_out in zip(dims[:-2], dims[1:-1])
        ]
        module.append(nn.Linear(dims[-2], dims[-1]))  # no activation on last layer
        self.model = nn.Sequential(*module)

    def forward(self, x):
        """Forward pass of the MLP."""
        return self.model(x)

    def loss_function(self, x, x_hat):
        """Loss function for the MLP."""
        return {"loss": nn.functional.mse_loss(x_hat, x, reduction="sum")}
