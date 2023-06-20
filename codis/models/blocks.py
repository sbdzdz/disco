"""Common blocks used in models."""
from torch import nn


class Encoder(nn.Module):
    """A simple encoder model."""

    def __init__(self, channels: list[int], in_channels: int = 1) -> None:
        """Initialize the encoder.
        Args:
            in_channels: The number of input channels.
            channels: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        channels = [in_channels] + channels
        module = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ]
        self.encoder = nn.Sequential(*module)

    def forward(self, x):
        """Forward pass of the encoder."""
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    """A simple decoder model."""

    def __init__(self, hidden_dims: list[int], out_channels: int = 1) -> None:
        """Initialize the decoder.
        Args:
            out_channels: The number of output channels.
            hidden_dims: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        modules = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    hidden_dims[-1], out_channels=out_channels, kernel_size=3, padding=1
                ),
                nn.Sigmoid(),
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass of the decoder."""
        x = self.model(x)
        return x
