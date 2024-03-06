"""Test the cropping mechanism of the Regressor module."""

import torch
from disco.visualization import draw_batch_and_reconstructions
from disco.lightning.modules import Regressor
import idsprites as ids
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader


def _main():
    dataset = ids.InfiniteDSprites()
    model = Regressor()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    batch, _ = next(iter(dataloader))

    width, height = batch.shape[-2:]
    cropped = model.crop(batch, batch)[0]
    print(batch.shape, cropped.shape)
    resized = torch.stack([F.resize(img, (width, height)) for img in cropped])

    draw_batch_and_reconstructions(batch, resized, save=False, show=True)


if __name__ == "__main__":
    _main()
