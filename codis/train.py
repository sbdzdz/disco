"""Training script."""
import argparse
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

import wandb
from codis.data import DSprites, InfiniteDSpritesRandom
from codis.models import BetaVAE
from codis.visualization import draw_batch_and_reconstructions


def train(args):
    """Train the model."""
    wandb.init(project="codis", config=args, dir=args.wandb_dir)
    config = wandb.config
    print(f"Cuda available {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set = random_split(
        DSprites(config.dsprites_path).to(device),
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(42),
    )
    test_set = InfiniteDSpritesRandom(
        image_size=64,
        radius_std=0.8,
        scale_range=np.linspace(0.5, 1.5, 10),
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    first_batch, _ = next(iter(train_loader))
    model = BetaVAE(beta=config.beta, latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # train on dsprites
    running_loss = defaultdict(list)
    for _ in range(config.epochs):
        for i, (batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch)  # pylint: disable=not-callable
            loss = model.loss_function(batch, x_hat, mu, log_var)
            loss["loss"].backward()
            optimizer.step()
            for k, v in loss.items():
                running_loss[k].append(v.item())
            if i > 0 and i % config.log_every == 0:
                with torch.no_grad():
                    x_hat, *_ = model(first_batch)  # pylint: disable=not-callable
                log_metrics(running_loss, first_batch, x_hat, suffix="_train")
                for k in running_loss:
                    running_loss[k] = []
                evaluate(model, val_set, device, config, suffix="_val")

    evaluate(model, test_set, device, config, suffix="_test")
    wandb.finish()


def evaluate(model, dataset, device, config, suffix=""):
    """Evaluate the model on the validation set."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    first_batch, _ = next(iter(dataloader))
    running_loss = defaultdict(list)
    first_batch = first_batch.to(device)

    with torch.no_grad():
        for batch, _ in islice(dataloader, config.eval_on):
            batch = batch.to(device)
            x_hat, mu, log_var = model(batch)
            loss = model.loss_function(batch, x_hat, mu, log_var)
            for k, v in loss.items():
                running_loss[k].append(v.item())
        x_hat, *_ = model(first_batch)
        log_metrics(running_loss, first_batch, x_hat, suffix=suffix)
    model.train()


def log_metrics(loss, x, x_hat, suffix=""):
    """Log the loss and the reconstructions."""
    wandb.log(
        {
            f"reconstruction{suffix}": wandb.Image(
                draw_batch_and_reconstructions(
                    x.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
                )
            ),
            **{
                f"{name}{suffix}": sum(value) / len(value)
                for name, value in loss.items()
            },
        }
    )


def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="How often to log training progress. The logs will be averaged over this number of batches.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter.")
    parser.add_argument(
        "--eval_on", type=int, default=100, help="Number of batches to evaluate on."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=10, help="Dimensionality of the latent space."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--wandb_dir",
        type=Path,
        default=repo_root / "wandb",
        help="Wandb logging directory.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
