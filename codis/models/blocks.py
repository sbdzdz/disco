"""Building blocks for the VAEs."""
from torch import nn


class EncoderBlock(nn.Module):
    """Standard encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass of the encoder block."""
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """Standard decoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward pass of the decoder block."""
        return self.relu(self.bn(self.conv(x)))
