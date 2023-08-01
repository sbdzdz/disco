"""Visualize joint training results"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt

import wandb


def visualize_joint_training(args):
    """Visualize joint training results"""
    runs = get_runs(args.wandb_entity, args.wandb_group)
    shapes = [run.config["tasks"] for run in runs]
    ood_loss = [run.summary[f"{args.metric_name}_test_task_0"] for run in runs]
    val_loss = [run.summary[f"{args.metric_name}_val"] for run in runs]

    plt.style.use("ggplot")
    _, ax = plt.subplots(figsize=(20, 9), layout="tight")
    plt.plot(shapes, ood_loss, label="Novel shapes")
    plt.plot(shapes, val_loss, label="Seen shapes, novel transforms")

    ax.legend(loc="upper right")
    ax.set_xlabel("Number of shapes")
    ax.set_ylabel("Minimum loss")

    plt.savefig(args.out_path, bbox_inches="tight")


def get_runs(entity, group):
    """Get runs from wandb"""
    api = wandb.Api()
    runs = api.runs(entity, filters={"group": group})
    runs = list(sorted(runs, key=lambda run: int(run.config["tasks"])))
    return runs


def _main():
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="Wandb entity and project name, e.g. codis/codis",
        default="codis/codis",
    )
    parser.add_argument("--wandb_group", type=str, help="Wandb group name")
    parser.add_argument(
        "--metric_name", type=str, help="Metric name", default="regression_loss"
    )
    parser.add_argument(
        "--out_path", type=Path, default=repo_root / "img/joint_training.png"
    )
    args = parser.parse_args()
    visualize_joint_training(args)


if __name__ == "__main__":
    _main()
