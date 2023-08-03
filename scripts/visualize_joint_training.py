"""Visualize joint training results"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt

import wandb


def visualize_joint_training(args):
    """Visualize joint training results"""
    runs = get_runs(args.wandb_entity, args.wandb_group)
    shapes = [run.config["tasks"] for run in runs]

    plt.style.use("ggplot")
    _, ax = plt.subplots(figsize=(20, 9), layout="tight")
    if args.break_down_factors:
        for factor in ["orientation", "scale", "position_x", "position_y"]:
            loss_ood = [run.summary[f"{factor}_loss_test_task_0"] for run in runs]
            loss_val = [run.summary[f"{factor}_loss_val"] for run in runs]
            plt.plot(
                shapes,
                loss_ood,
                label=f"{factor.capitalize().replace('_', ' ')} (novel shapes)",
            )
            plt.plot(
                shapes,
                loss_val,
                label=f"{factor.capitalize().replace('_', ' ')} (seen shapes, novel factor values)",
            )
    else:
        loss_ood = [run.summary[f"{args.metric_name}_test_task_0"] for run in runs]
        loss_val = [run.summary[f"{args.metric_name}_val"] for run in runs]
        plt.plot(shapes, loss_ood, label="Novel shapes")
        plt.plot(shapes, loss_val, label="Seen shapes, novel transforms")

    ax.legend(loc="upper right")
    ax.set_xlabel("Number of shapes")
    ax.set_xlim([args.xmin, args.xmax])
    ax.set_ylabel("Minimum loss")
    ax.set_ylim([args.ymin, args.ymax])

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
    parser.add_argument("--break_down_factors", action="store_true")
    parser.add_argument("--xmin", type=float, help="X-axis min limit.")
    parser.add_argument("--xmax", type=float, help="X-axis max limit.")
    parser.add_argument("--ymin", type=float, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, help="Y-axis max limit.")
    args = parser.parse_args()
    visualize_joint_training(args)


if __name__ == "__main__":
    _main()
