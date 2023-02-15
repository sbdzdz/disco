"""Training script."""
import argparse
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from codis.data import DSprites, InfiniteDSprites
from codis.models import BetaVAE
from codis.visualization import show_images_grid


def train(args):
    """Train the model."""
    dsprites = DSprites(args.dsprites_path)
    dsprites_loader = DataLoader(dsprites, batch_size=16, shuffle=True)
    infinite_dsprites = InfiniteDSprites()
    infinite_dsprites_loader = DataLoader(infinite_dsprites, batch_size=16)
    model = BetaVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train on dSprites
    for epoch in range(10):
        running_loss = 0
        for i, batch in enumerate(dsprites_loader):
            optimizer.zero_grad()
            x_hat, mu, log_var, _ = model(batch)
            losses = model.loss_function(batch, x_hat, mu, log_var)
            losses["loss"].backward()
            optimizer.step()
            running_loss += losses["loss"]
            if i % 100 == 0:
                print(f"Epoch: {epoch}, batch: {i}, loss: {running_loss/100:2f}")
                running_loss = 0

    # evaluate
    model.eval()
    with torch.no_grad():
        for batch in islice(infinite_dsprites_loader, 20):
            x_hat, mu, log_var, _ = model(batch)
            loss = model.loss_function(batch, x_hat, mu, log_var)["loss"]
            print(loss)
        show_images_grid(batch, x_hat)


def _main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--dsprites_path",
        type=Path,
        default=repo_root / "codis/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    _main()
