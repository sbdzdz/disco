"""Factory functions for common blocks used in models."""
from torch import nn


def build_encoder_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
) -> nn.Sequential:
    """Build an encoder block."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def build_decoder_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    output_padding: int = 1,
) -> nn.Sequential:
    """Build a decoder block."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def build_final_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    output_padding: int = 1,
) -> nn.Sequential:
    """Build the final layer."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels=3, kernel_size=3, padding=1),
        nn.Tanh(),
    )
