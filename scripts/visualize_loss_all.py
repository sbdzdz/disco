"""Pull loss values from a few wandb runs and create an aggregate plot.
Example:
    python scripts/visualize_loss.py --run_ids entity/project/run_id1 entity/project/run_id2
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm


def visualize_loss(args):
    """Visualize train, validation, and optionally test losses."""
    api = wandb.Api(timeout=60)
    runs = api.runs(args.wandb_entity, filters={"group": args.wandb_group})
    if len(runs) == 0:
        raise ValueError("No matching runs found.")
    else:
        print(f"Found {len(runs)} runs.")

    stages = []
    if args.show_test:
        if args.show_test == "aggregate":
            stages.extend(["test"])
        elif args.show_test == "per_task":
            if args.num_tasks:
                num_tasks = args.num_tasks
            else:
                num_tasks = runs[0].config.get("dataset.tasks")
            if num_tasks is None and args.show_test is not None:
                raise ValueError("Couldn't infer the number of tasks.")
            stages.extend([f"test_task_{i}" for i in range(num_tasks)])

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

        # trim to the shortest run
        min_len = min(len(s) for s in steps)
        steps = [s[:min_len] for s in steps]
        values = [v[:min_len] for v in values]

        values_mean = np.mean(values, axis=0)
        ax.plot(steps[0], values_mean, label=f"{args.metric_name}_{stage}")
        if args.show_std:
            values_std = np.std(values, axis=0)
            ax.fill_between(
                steps[0],
                values_mean - values_std,
                values_mean + values_std,
                alpha=0.3,
            )

    if args.xticks == "tasks":
        task_transitions = get_task_transitions(runs[0])[:: args.x_ticks_every]
        ax.set_xticks(task_transitions)
        ax.set_xticklabels(
            range(min_len * runs[0].config.training.test_every_n_tasks)[
                :: args.x_ticks_every
            ]
        )
        ax.set_xlabel("Tasks", fontsize=args.fontsize)
    else:
        ax.set_xlabel("Steps", fontsize=args.fontsize)
    ax.set_xlim([args.xmin, args.xmax])

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    ax.yaxis.set_major_formatter("{x:.3f}")
    metric_name = args.metric_name.replace("_", " ").capitalize()
    ax.set_ylabel(metric_name, fontsize=args.fontsize)
    ax.set_ylim([args.ymin, args.ymax])

    if args.plot_title is None:
        args.plot_title = (
            f"{metric_name} (STN, 10 shapes per task, average of {len(runs)} runs.)"
        )
    ax.set_title(args.plot_title, y=1.05, fontsize=args.fontsize)
    ax.legend(loc="upper right")

    plt.savefig(args.out_path, bbox_inches="tight")


def get_task_transitions(run):
    """Get task transitions from a wandb run."""
    steps, values = download_metric(run=run, name="task_id")
    return [
        steps[i] for i, (v, v_next) in enumerate(zip(values, values[1:])) if v != v_next
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
        "--show_test",
        type=str,
        default=None,
        choices=["aggregate", "per_task"],
        help="Show test loss. Options: None, aggregate, per_task",
    )
    parser.add_argument(
        "--show_std", action="store_true", help="Visualize standard deviation."
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Title of the plot.",
    )
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--xmin", type=float, help="X-axis min limit.")
    parser.add_argument("--xmax", type=float, help="X-axis max limit.")
    parser.add_argument(
        "--xticks", type=str, help="X-axis data", choices=["steps", "tasks"]
    )
    parser.add_argument(
        "--x_ticks_every", type=int, default=10, help="Granularity of the x axis."
    )
    parser.add_argument("--ymin", type=float, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, help="Y-axis max limit.")
    parser.add_argument("--fontsize", type=int, default=20)
    args = parser.parse_args()
    visualize_loss(args)


if __name__ == "__main__":
    _main()
