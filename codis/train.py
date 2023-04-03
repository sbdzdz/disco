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
from codis.models import BetaVAE, MLP
from codis.visualization import draw_batch_and_reconstructions


def train(args):
    """Train the model."""
    wandb.init(project="codis", group=args.wandb_group, dir=args.wandb_dir, config=args)
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

    vae = BetaVAE(beta=config.beta, latent_dim=config.latent_dim).to(device)
    mlp = MLP(dims=[config.latent_dim, 40, 40, 40, train_set.num_latents]).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=config.lr)

    for _ in range(args.epochs):
        vae = train_vae_one_epoch(vae, optimizer, train_set, device, config)
        mlp = train_mlp_one_epoch(mlp, vae, optimizer, val_set, device, config)
        evaluate(vae, val_set, device, config, suffix="_val")

    evaluate(vae, test_set, device, config, suffix="_test")
    wandb.finish()


def train_vae_one_epoch(model, optimizer, dataset, device, config):
    """Train the model for one epoch."""
    model.train()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)

    running_loss = defaultdict(list)
    for i, (batch, _) in enumerate(dataloader):
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
            log_metrics(running_loss, suffix="vae_train")
            log_reconstructions(first_batch, x_hat, suffix="vae_train")
            for k in running_loss:
                running_loss[k] = []
    return model


def train_mlp_one_epoch(mlp, feature_extractor, optimizer, dataset, device, config):
    """Train the model for one epoch."""
    mlp.train()
    feature_extractor.eval()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)

    running_loss = defaultdict(list)
    for i, (img_batch, latent_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            x_hat = feature_extractor(img_batch)
        x_hat = mlp(x_hat)
        loss = mlp.loss_function(latent_batch, x_hat)
        loss["loss"].backward()
        optimizer.step()
        for k, v in loss.items():
            running_loss[k].append(v.item())
        if i > 0 and i % config.log_every == 0:
            with torch.no_grad():
                x_hat, *_ = mlp(first_batch)  # pylint: disable=not-callable
            log_metrics(running_loss, suffix="_train")
            for k in running_loss:
                running_loss[k] = []
    return mlp


def evaluate(model, dataset, device, config, suffix=""):
    """Evaluate the model on the validation set."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)

    running_loss = defaultdict(list)
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


def log_metrics(loss, suffix=""):
    """Log the loss."""
    wandb.log(
        {f"{name}{suffix}": sum(value) / len(value) for name, value in loss.items()}
    )


def log_reconstructions(x, x_hat, suffix=""):
    """Log the original and reconstructed images."""
    wandb.log(
        {
            f"reconstruction{suffix}": wandb.Image(
                draw_batch_and_reconstructions(x, x_hat)
            ),
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
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb group name. If not specified, a new group will be created.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
