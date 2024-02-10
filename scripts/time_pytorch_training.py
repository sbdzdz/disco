"""Train a Resnet-18 on Infinite DSprites with vanilla PyTorch."""

from argparse import ArgumentParser
from pathlib import Path
import wandb

import torch
import timm
from torch.utils.data import DataLoader
from time import time
from disco.data import ContinualBenchmarkDisk


def train(args):
    model = timm.create_model("resnet18", pretrained=False, num_classes=4)
    model = model.to(args.device)
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
                loss = torch.nn.functional.mse_loss(output, factors)
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
    print(f"{name} took {int(minutes)}min and {seconds:.2f}s.")


def stack_factors(factors, device):
    factor_names = ["scale", "orientation", "position_x", "position_y"]
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
