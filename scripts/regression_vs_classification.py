# Download the training loss for the regression and classification runs and plot them
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import wandb


def _main(args):
    api = wandb.Api()

    classification_run = api.run("disco/disco/z9wmsw3d")
    regression_run = api.run("disco/disco/licb0eis")

    losses = {
        "classification": get_train_loss(args, classification_run),
        "regression": get_train_loss(args, regression_run),
    }
    plot(args, losses)


def get_train_loss(args, run):
    steps_per_task = get_test_steps_per_task(run)
    loss = download_metric(run, "train/loss", args.force_download)
    loss = loss[: args.num_tasks * steps_per_task]
    steps = [step / steps_per_task for step in range(len(loss))]

    return steps, loss


def download_metric(run, metric_name, force_download=False):
    path = Path(__file__).parent.parent / f"img/media/{run.name}/{metric_name}.json"
    if path.exists() and not force_download:
        print(f"Found saved run data for {run.name}.")
        with open(path, "r") as f:
            values = json.load(f)
    else:
        values = run.scan_history(keys=[metric_name])
        values = [row[metric_name] for row in values]
    if not path.exists() or force_download:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(values, f)
    return values


def get_test_steps_per_task(run):
    """Get the number of steps per task for a given run."""
    train_split = run.config["dataset"]["train_split"]
    factor_resolution = run.config["dataset"]["factor_resolution"]
    epochs_per_task = run.config["trainer"]["max_epochs"]
    batch_size = run.config["dataset"]["batch_size"]
    shapes_per_task = run.config["dataset"]["shapes_per_task"]
    if run.config["trainer"].get("devices") == -1:  # multi-gpu
        batch_size *= 2

    samples_per_epoch = math.ceil(
        train_split * shapes_per_task * factor_resolution**4
    )
    steps_per_epoch = round(samples_per_epoch / batch_size)

    print(batch_size, samples_per_epoch, steps_per_epoch)
    return int(steps_per_epoch * epochs_per_task)


def plot(args, metrics):
    """Plot the training loss for regression and classification."""
    plt.style.use(["science"])
    steps_regression, regression_train_loss = metrics["regression"]
    steps_classification, classification_train_loss = metrics["classification"]
    _, (ax1, ax2) = plt.subplots(
        2, 1, layout="tight", figsize=(args.fig_width, args.fig_height)
    )
    ax1.plot(steps_classification, classification_train_loss, label="Classification")
    ax2.plot(steps_regression, regression_train_loss, label="Regression")

    ax1.set_title("Classification")
    ax2.set_title("Regression")
    for ax in (ax1, ax2):
        ax.set_xlabel("Tasks")
        ax.set_ylabel("Training Loss")

    plt.savefig("img/regression_vs_classification.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=21)
    parser.add_argument("--fig_width", type=float, default=5.5)
    parser.add_argument("--fig_height", type=float, default=2.5)
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()
    _main(args)
