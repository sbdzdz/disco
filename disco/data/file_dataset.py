from pathlib import Path
from typing import Union

import idsprites as ids
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class FileDataset(Dataset):
    def __init__(self, path: Union[Path, str], transform=None, target_transform=None):
        self.path = Path(path)
        self.transform = transform
        self.target_transform = target_transform
        factors = np.load(self.path / "factors.npz", allow_pickle=True)
        factors = [
            dict(zip(factors, value)) for value in zip(*factors.values())
        ]  # turn dict of lists into list of dicts
        self.data = [ids.Factors(**factors) for factors in factors]
        self.shapes = np.load(self.path / "../shapes.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.path / f"sample_{idx}.png"
        image = read_image(str(img_path)) / 255.0

        factors = self.data[idx]
        factors = factors.replace(
            shape=self.shapes[factors.shape_id % len(self.shapes)]
        )
        factors = factors.to_tensor().to(torch.float32)
        factors = factors.replace(shape_id=factors.shape_id.to(torch.long))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            factors = self.target_transform(factors)
        return image, factors
