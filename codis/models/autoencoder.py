from torch import nn
from codis.models.blocks import EncoderBlock, DecoderBlock


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int] = None) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        hidden_dims = [in_channels] + hidden_dims
        encoder_modules = [
            EncoderBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.encoder = nn.Sequential(*encoder_modules)

        hidden_dims.reverse()
        decoder_modules = [
            DecoderBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.decoder = nn.Sequential(decoder_modules)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
