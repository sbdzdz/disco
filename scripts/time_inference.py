"""Load a pre-trained ResNet-18 model and time inference on a set of images."""

import timm
import torch
from disco.data import FileDataset
from time import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader


def _main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet18", pretrained=True).to(device)
    dataset = FileDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    times = []
    for images in dataloader:
        start = time()
        model(images.to(device))
        end = time()
        times.append(end - start)
    average = sum(times) / len(times)
    print(f"Average inference time: {average:.2f} seconds.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    _main()
