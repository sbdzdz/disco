"""A simple multi-layer perceptron.""" ""
from torch import nn


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
