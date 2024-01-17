"""A module containing an implementation of a simple autoencoder."""
from typing import Optional

from torch import nn

from disco.models.blocks import Encoder, Decoder


class AutoEncoder(nn.Module):
    """A simple autoencoder."""

    def __init__(self, in_channels: int, channels: Optional[list[int]] = None) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512]
        channels = [in_channels] + channels

        self.encoder = Encoder(channels=channels, in_channels=in_channels)
        self.decoder = Decoder(reversed(channels), out_channels=in_channels)

    def forward(self, x):
        """Forward pass of the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
