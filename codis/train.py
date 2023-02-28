"""Training script."""
import argparse
import sys
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda available:", torch.cuda.is_available())

    wandb.init(project="codis")
    dsprites = DSprites(args.dsprites_path)
    dsprites_loader = DataLoader(dsprites, batch_size=128, shuffle=True)
    infinite_dsprites = InfiniteDSprites()
    infinite_dsprites_loader = DataLoader(infinite_dsprites, batch_size=16)
    model = BetaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(dsprites_loader)).to(device)
    x_hat, *_ = model(batch.to(device))
    draw_batch_and_reconstructions(
        batch.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
    )  # list has no attribute detach

    sys.exit(0)

    # train on dSprites
    running_loss = 0
    for epoch in range(10):
        for i, batch in enumerate(dsprites_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            x_hat, mu, log_var = model(batch)
            losses = model.loss_function(batch, x_hat, mu, log_var)
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"].item()
            if i % args.log_every == 0:
                with torch.no_grad():
                    x_hat, *_ = model(batch)
                print(
                    f"Epoch: {epoch}, batch: {i}, loss: {running_loss/args.log_every:2f}"
                )
                wandb.log(
                    {
                        "reconstruction": draw_batch_and_reconstructions(
                            batch.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
                        ),
                        "training_loss": running_loss / args.log_every,
                    }
                )
                running_loss = 0

    # evaluate
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in islice(infinite_dsprites_loader, 1000):
            # normalize the batch
            batch = batch / 255.0
            x_hat, mu, log_var, _ = model(batch)
            loss = model.loss_function(batch, x_hat, mu, log_var)["loss"].item()
            losses.append(loss)
        average_loss = sum(losses) / len(losses)
        print(f"Average loss on the infinite dataset: {average_loss}")

        wandb.log(
            {
                "idsprites_reconstruction": draw_batch_and_reconstructions(
                    batch.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
                ),
                "idsprites_loss": average_loss,
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
        "--log_every", type=int, default=100, help="Log every n batches"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
