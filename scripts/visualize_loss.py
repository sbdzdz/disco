"""Pull loss values from a few wandb runs and create an aggregate plot.
Example:
    python scripts/visualize_loss.py --run_ids entity/project/run_id1 entity/project/run_id2
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb


def visualize_loss(args):
    """Visualize train, validation, and optionally test losses."""
    api = wandb.Api()
    runs = api.runs(
        args.wandb_entity, filters={"group": args.wandb_group, "state": "finished"}
    )
    if len(runs) == 0:
        raise ValueError("No matching runs found.")

    stages = ["val", "train"]
    if args.include_test:
        if args.test_per_task:
            stages.extend([f"test_task_{i}" for i in range(runs[0].config["tasks"])])
        else:
            stages.extend(["test"])
    plt.style.use("ggplot")
    _, ax = plt.subplots(figsize=(20, 9), layout="tight")
    for stage in stages:
        print(f"Downloading {stage}...")
        metric = f"{args.metric_name}_{stage}"
        subsample = args.subsample if stage in ["val", "train"] else 1
        metrics = [
            download_metric(
                run,
                metric,
                subsample=subsample,
                max_samples=args.max_samples,
            )
            for run in tqdm(runs)
        ]
        steps, values = zip(*metrics)
        values_mean = np.mean(values, axis=0)

        ax.plot(steps[0], values_mean, label=f"{args.metric_name}_{stage}")
        if args.include_std:
            values_std = np.std(values, axis=0)
            ax.fill_between(
                steps[0],
                values_mean - values_std,
                values_mean + values_std,
                alpha=0.3,
            )

    for task_transition in get_task_transitions(runs[0]):
        ax.axvline(task_transition, color="gray", linestyle="dotted", linewidth=1)

    ax.set_xlabel("Steps")
    ax.set_xlim([args.xmin, args.xmax])
    ax.set_ylabel("Loss")
    ax.set_ylim([args.ymin, args.ymax])
    ax.set_title(args.plot_title)
    ax.legend(loc="upper right")

    plt.savefig(args.out_path, bbox_inches="tight")


def get_task_transitions(run):
    """Get task transitions from a wandb run."""
    task_id_steps, task_id_values = download_metric(run, "task_id")
    return [
        task_id_steps[i]
        for i in range(len(task_id_steps) - 1)
        if task_id_values[i] != task_id_values[i + 1]
    ]


def download_metric(run, name, subsample: int = 1, max_samples: int = None):
    """Download a metric from a wandb run.
    Args:
        run: Wandb run object.
        name: Name of the metric to download.
        subsample: Subsample rate.
    """
    if "test" in name:
        steps, values = zip(
            *[
                (row["_step"], row[name])
                for row in run.history(keys=["_step", name], pandas=False)
            ]
        )
    else:
        steps, values = zip(
            *[
                (row["_step"], row[name])
                for row in run.scan_history(keys=["_step", name])
            ]
        )
    steps = steps[::subsample]
    values = values[::subsample]

    if max_samples is not None:
        steps = steps[:max_samples]
        values = values[:max_samples]

    return steps, values


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
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
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate.")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max number of data points."
    )
    parser.add_argument(
        "--include_test", action="store_true", help="Include test losses in the plot."
    )
    parser.add_argument(
        "--test_per_task",
        action="store_true",
        help="Plot test losses per task instead of aggregated.",
    )
    parser.add_argument(
        "--include_std", action="store_true", help="Visualize standard deviation."
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Loss",
        help="Title of the plot.",
    )
    parser.add_argument("--xmin", type=float, help="X-axis min limit.")
    parser.add_argument("--xmax", type=float, help="X-axis max limit.")
    parser.add_argument("--ymin", type=float, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, help="Y-axis max limit.")
    args = parser.parse_args()
    visualize_loss(args)


if __name__ == "__main__":
    _main()
