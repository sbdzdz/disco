"""Pull loss values from a few wandb runs and create an aggregate plot."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import wandb

plt.style.use("ggplot")


def visualize_loss(args):
    """Visualize train, validation, and optionally test losses."""
    api = wandb.Api()
    stages = ["val", "train"]

    if args.include_test:
        stages.extend([f"test_{i}" for i in range(9)])

    _, ax = plt.subplots(figsize=(20, 9), layout="tight")
    for stage in stages:
        metric = f"{args.metric_name}_{stage}"
        metrics = [
            download_metric(
                api.run(run_id),
                metric,
                subsample=args.subsample,
                max_samples=args.max_samples,
            )
            for run_id in args.run_ids
        ]
        steps, values = zip(*metrics)
        values_mean = np.mean(values, axis=0)

        ax.plot(steps[0], values_mean, label=f"loss_{stage}")
        if args.visualize_std:
            values_std = np.std(values, axis=0)
            ax.fill_between(
                steps[0],
                values_mean - values_std,
                values_mean + values_std,
                alpha=0.3,
            )

    for task_transition in get_task_transitions(api.run(args.run_ids[0])):
        ax.axvline(task_transition, color="gray", linestyle="dotted", linewidth=1)

    ax.set_xlim([-300, 5000])
    ax.set_ylim([0, 1.0])
    ax.set_title("Loss (average of 5 runs)")
    ax.set_xlabel("Steps")
    ax.legend(loc="upper right")

    plt.savefig(args.output_dir / "train_test.png")


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
    steps, values = zip(
        *[(row["_step"], row[name]) for row in run.scan_history(keys=["_step", name])]
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
        "--run_ids",
        type=str,
        nargs="+",
        required=True,
        help="Full wandb run IDs, e.g. 'entity/project/run_id'.",
    )
    parser.add_argument("--output_dir", type=Path, default=repo_root / "img")
    parser.add_argument("--metric_name", type=str, default="loss")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate.")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max number of data points."
    )
    parser.add_argument(
        "--include_test", action="store_true", help="Include test losses in the plot."
    )
    args = parser.parse_args()

    visualize_loss(args)


if __name__ == "__main__":
    _main()
