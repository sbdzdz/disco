from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = self.load_data(path)
        self.transform = transform

    def load_data(self, path):
        return np.load(
            path,
            encoding="latin1",
            allow_pickle=True,
        )

    def __len__(self):
        return len(self.data["imgs"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data["imgs"][idx]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        if self.transform:
            img = self.transform(img)

        latents = self.data["latents_values"][idx]

        return img, latents


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    dataset = DSpritesDataset(
        project_dir / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    )
    print(len(dataset))
