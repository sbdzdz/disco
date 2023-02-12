"""A module containing an implementation of a simple autoencoder."""
from typing import Optional

from torch import nn

from codis.models.blocks import Encoder, Decoder


class AutoEncoder(nn.Module):
    """A simple autoencoder."""

    def __init__(
        self, in_channels: int, hidden_dims: Optional[list[int]] = None
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = [in_channels] + hidden_dims

        self.encoder = Encoder(in_channels, hidden_dims)
        self.decoder = Decoder(reversed(hidden_dims))

    def forward(self, x):
        """Forward pass of the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
