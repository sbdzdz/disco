"""Training script."""
import argparse
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb
from codis.data import DSprites, InfiniteDSprites
from codis.models import BetaVAE
from codis.visualization import draw_batch_and_reconstructions


def train(args):
    """Train the model."""
    run = wandb.init(project="codis", config=args)
    run.log_code()
    wandb.log({"cuda_available": torch.cuda.is_available()})
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dsprites = DSprites(config.dsprites_path).to(device)
    dsprites_loader = DataLoader(dsprites, batch_size=config.batch_size, shuffle=True)
    infinite_dsprites = InfiniteDSprites()
    infinite_dsprites_loader = DataLoader(
        infinite_dsprites, batch_size=config.batch_size
    )
    model = BetaVAE(beta=config.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    first_batch = next(iter(dsprites_loader))

    # train on dsprites
    running_losses = defaultdict(float)
    for epoch in range(config.epochs):
        for i, batch in enumerate(dsprites_loader):
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch)
            losses = model.loss_function(batch, x_hat, mu, log_var)
            losses["loss"].backward()
            optimizer.step()
            for k, v in losses.items():
                running_losses[k] += v.item()
            if i > 0 and i % config.log_every == 0:
                with torch.no_grad():
                    x_hat, *_ = model(first_batch)
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": epoch * len(dsprites_loader) + i,
                        "reconstruction": wandb.Image(
                            draw_batch_and_reconstructions(
                                first_batch.detach().cpu().numpy(),
                                x_hat.detach().cpu().numpy(),
                            ),
                        ),
                        **{
                            loss_name: (loss_value / config.log_every)
                            for loss_name, loss_value in running_losses.items()
                        },
                    }
                )
                for k in running_losses:
                    running_losses[k] = 0

    # evaluate on infinite dsprites
    model.eval()
    running_losses = defaultdict(float)
    with torch.no_grad():
        for batch in islice(infinite_dsprites_loader, config.eval_on):
            x_hat, mu, log_var, _ = model(batch)
            loss = model.loss_function(batch, x_hat, mu, log_var)["loss"].item()
            losses.append(loss)
            for k, v in losses.items():
                running_losses[k] += v.item()
        wandb.log(
            {
                "idsprites_reconstruction": draw_batch_and_reconstructions(
                    batch.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
                ),
                **{
                    loss_name + "_eval": loss_value / config.eval_on
                    for loss_name, loss_value in running_losses.items()
                },
            }
        )
    wandb.finish()


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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
