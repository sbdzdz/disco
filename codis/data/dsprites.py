"""DSprites dataset loader."""
import numpy as np
import torch
from torch.utils.data import Dataset


class DSprites(Dataset):
    """DSprites dataset."""

    def __init__(self, path):
        self.imgs = torch.from_numpy(
            np.load(path, encoding="latin1", allow_pickle=True)["imgs"]
        )

    def to(self, device):
        """Move the data to the given device."""
        self.imgs = self.imgs.to(device)
        return self

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.imgs[idx].float().unsqueeze(0)
