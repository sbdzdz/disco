"""A module containing an implementation of a simple autoencoder."""
from typing import Optional

from torch import nn

from codis.models.blocks import build_decoder_block, build_encoder_block


class AutoEncoder(nn.Module):
    """A simple autoencoder."""

    def __init__(
        self, in_channels: int, hidden_dims: Optional[list[int]] = None
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        hidden_dims = [in_channels] + hidden_dims
        encoder_modules = [
            build_encoder_block(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.encoder = nn.Sequential(*encoder_modules)

        hidden_dims.reverse()
        decoder_modules = [
            build_decoder_block(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        """Forward pass of the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
