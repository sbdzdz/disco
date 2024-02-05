"""Train a Resnet-18 on Infinite DSprites with vanilla PyTorch."""

from argparse import ArgumentParser
from pathlib import Path
import wandb
from typing import Union

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image
import idsprites as ids
from time import time


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
        image = np.array(read_image(str(img_path)) / 255.0)

        factors = self.data[idx]
        factors = factors.replace(
            shape=self.shapes[factors.shape_id % len(self.shapes)]
        )
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            factors = self.target_transform(factors)
        return image, factors


class ContinualBenchmarkDisk:
    def __init__(
        self,
        path: Union[Path, str],
        accumulate_test_set: bool = True,
    ):
        """Initialize the continual learning benchmark.
        Args:
            path: The path to the dataset.
            accumulate_test_set: Whether to accumulate the test set over tasks.
        """
        self.path = Path(path)
        self.accumulate_test_set = accumulate_test_set
        if self.accumulate_test_set:
            self.test_sets = []

    def __iter__(self):
        for task_dir in sorted(
            self.path.glob("task_*"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            train = FileDataset(task_dir / "train")
            test = FileDataset(task_dir / "test")

            if self.accumulate_test_set:
                self.test_sets.append(test)
                test = ConcatDataset(self.test_sets)

            yield train, test


def train(args):
    model = timm.create_model("resnet18")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.init(project="disco", config=args)

    for task, (train, test) in enumerate(ContinualBenchmarkDisk(args.data_dir)):
        task_start = time()
        for epoch in range(args.epochs):
            epoch_start = time()
            for batch in DataLoader(train, batch_size=args.batch_size, shuffle=True):
                images, factors = batch
                images = images.to(args.device)
                factors = stack_factors(factors, args.device)
                output = model(images)
                loss = torch.nn.functional.MSE(output, factors)
                wandb.log({"loss": loss.item()})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_end = time()
            print(f"Task {task} epoch {epoch} done.")
            log_duration(epoch_start, epoch_end, "epoch")

        test_start = time()
        for batch in DataLoader(test, batch_size=args.batch_size):
            images, factors = batch
            images = images.to(args.device)
            factors = stack_factors(factors, args.device)
            output = model(images)
            loss = torch.nn.functional.MSE(output, factors)
            wandb.log({"test_loss": loss.item()})
        test_end = time()
        print(f"Task {task} test done.")
        log_duration(test_start, test_end, "test")
        log_duration(task_start, test_end, "task")
    wandb.finish()


def log_duration(start, end, name):
    minutes = (end - start) // 60
    seconds = (end - start) % 60
    wandb.log({f"{name}_duration": end - start})
    print(f"{name.capitalize()} took {minutes}min {seconds}s.")


def stack_factors(factors, device):
    factor_names = ["scale", "orientation", "postion_x", "position_y"]
    return (
        torch.cat(
            [getattr(factors, name).unsqueeze(-1) for name in factor_names],
            dim=-1,
        )
        .float()
        .to(device)
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=str, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train(args)
