"""Load a pre-trained ResNet-18 model and time inference on a set of images."""

import timm
import torch
from disco.data import FileDataset
from time import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


def _main(args):
    start = time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet18", pretrained=True).to(device)
    model.eval()
    datasets = [
        FileDataset(task_dir / "test")
        for task_dir in tqdm(Path(args.dataset_path).glob("task_*"))
    ]
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    loading_end = time()
    duration_min = (loading_end - start) / 60
    print(f"Loading the dataset took {duration_min:.2f} minutes.")

    training_start = time()
    for images, _ in dataloader:
        model(images.to(device))
        end = time()
    end = time()
    training_duration_min = (end - training_start) / 60
    duration_min = (end - start) / 60
    print(f"Inference took {training_duration_min:.2f} minutes.")
    print(f"Total duration: {duration_min:.2f} minutes.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks.")
    args = parser.parse_args()
    _main(args)
