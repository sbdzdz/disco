"""Common blocks used in models."""
from torch import nn


class Encoder(nn.Module):
    """A simple encoder model."""

    def __init__(self, hidden_dims: list[int], in_channels: int = 1) -> None:
        """Initialize the encoder.
        Args:
            in_channels: The number of input channels.
            hidden_dims: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        self.channels = hidden_dims
        hidden_dims = [in_channels] + hidden_dims
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
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.encoder = nn.Sequential(*module)

    def forward(self, x):
        """Forward pass of the encoder."""
        x = self.encoder(x)
        return x

    def out_size(self, img_size):
        """Return the output size of the encoder for a given image size."""
        out_img_size = img_size // 2 ** len(self.channels)
        return out_img_size**2 * self.channels[-1]


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
