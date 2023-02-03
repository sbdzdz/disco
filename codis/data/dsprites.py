"""DSprites dataset loader."""
import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    """DSprites dataset."""

    def __init__(self, path, transform=None):
        self.data = self.load_data(path)
        self.transform = transform

    def load_data(self, path):
        """Load the data from the given path."""
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
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return img
