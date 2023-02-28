"""Training script."""
import argparse
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
    dsprites = DSprites(args.dsprites_path).to(device)
    dsprites_loader = DataLoader(dsprites, batch_size=64, shuffle=True)
    infinite_dsprites = InfiniteDSprites()
    infinite_dsprites_loader = DataLoader(infinite_dsprites, batch_size=16)
    model = BetaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train on dsprites
    running_loss = 0
    for i in range(10):
        for i, batch in enumerate(dsprites_loader):
            optimizer.zero_grad()
            x_hat, mu, log_var = model(batch)
            losses = model.loss_function(batch, x_hat, mu, log_var)
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"].item()
            if i > 0 and i % args.log_every == 0:
                with torch.no_grad():
                    x_hat, *_ = model(batch)
                print(f"Epoch: {i}, batch: {0}, loss: {running_loss/args.log_every:2f}")
                wandb.log(
                    {
                        "reconstruction": wandb.Image(
                            draw_batch_and_reconstructions(
                                batch.detach().cpu().numpy(),
                                x_hat.detach().cpu().numpy(),
                            ),
                        ),
                        "training_loss": running_loss / args.log_every,
                    }
                )
                running_loss = 0

    # evaluate on infinite dsprites
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in islice(infinite_dsprites_loader, 1000):
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
    parser.add_argument("--log_every", type=int, default=10, help="Log every n batches")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
