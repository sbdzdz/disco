"""DSprites dataset loader."""
import numpy as np
import torch
from torch.utils.data import Dataset


class DSprites(Dataset):
    """DSprites dataset."""

    def __init__(self, path):
        data = np.load(path, encoding="latin1", allow_pickle=True)
        self.imgs = torch.from_numpy(data["imgs"])
        self.latents = torch.from_numpy(data["latents_values"])

    def to(self, device):
        """Move the data to the given device."""
        self.imgs = self.imgs.to(device)
        self.latents = self.latents.to(device)
        return self

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = self.imgs[idx].float().unsqueeze(0)
        latents = self.latents[idx].float().unsqueeze(0)
        yield (imgs, latents)
