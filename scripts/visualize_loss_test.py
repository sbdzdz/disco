"""Pull loss values from a few wandb run groups and create an aggregate plot.
Example:
    python scripts/visualize_loss.py --wandb_groups resnet
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm


def visualize_loss(args):
    """Visualize train, validation, and optionally test metrics."""
    api = wandb.Api(timeout=60)
    names = args.wandb_groups if args.names is None else args.names
    assert len(names) == len(args.wandb_groups), "Please provide a name for each group."
    run_dict = {
        name: [
            run
            for run in api.runs(args.wandb_entity, filters={"group": group})
            if run.name in args.wandb_names
        ]
        for name, group in zip(names, args.wandb_groups)
    }

    for name, runs in run_dict.items():
        print(f"Found {len(runs)} runs for {name}.")

    metrics = {}
    for name, runs in run_dict.items():
        if runs[0].config["model"] == "resnet":
            values = get_baseline_results(args.metric_name, runs)
        else:
            values = get_our_results(args.metric_name, runs)
        try:
            test_every_n_tasks = runs[0].config["training"]["test_every_n_tasks"]
        except KeyError:
            test_every_n_tasks = 1
        steps = [i * test_every_n_tasks for i in range(len(values[0]))]
        metrics[name] = (steps, values)

    # truncate steps and values to match the smallest step number
    max_steps = min(max(steps) for steps, _ in metrics.values())
    metrics = {
        name: (
            [step for step in steps if step <= max_steps],
            [[v for s, v in zip(steps, value) if s <= max_steps] for value in values],
        )
        for name, (steps, values) in metrics.items()
    }

    plot(args, metrics)


def get_baseline_results(metric_name, runs):
    """Get the metric values for each baseline run."""
    values = []
    for run in runs:
        task_values = []
        num_tasks = run.config["dataset"]["tasks"]
        for task in tqdm(range(num_tasks)):
            metric_name = (
                f"Top1_Acc_Exp/eval_phase/test_stream/Task{task:03d}/Exp{task:03d}"
            )
            task_values.extend(
                row[metric_name]
                for row in run.history(keys=[metric_name], pandas=False)
            )
        values.append(task_values)
    return values


def get_our_results(metric_name: str, runs: list):
    """Get the metric values for each run."""
    metric_name = f"test/{metric_name}"
    values = [
        take_last(run.scan_history(keys=["trainer/global_step", metric_name]))
        for run in tqdm(runs)
    ]
    shortest_len = min(len(v) for v in values)
    return [value[:shortest_len] for value in values]


def take_last(scan):
    """If there are multiple values for the same step, take the last one."""
    result = []
    current_step = None
    for row in reversed(list(scan)):
        step, value = row.values()
        if step != current_step:
            result.append(value)
            current_step = step
    return result[::-1]


def plot(args, metrics):
    plt.style.use("ggplot")
    _, ax = plt.subplots(figsize=(20, 9), layout="tight")
    for name, (steps, values) in metrics.items():
        values_mean = np.mean(values, axis=0)
        ax.plot(steps, values_mean, label=name)
        if args.show_std:
            values_std = np.std(values, axis=0)
            ax.fill_between(
                steps,
                values_mean - values_std,
                values_mean + values_std,
                alpha=0.3,
            )

    ax.set_xlabel("Tasks", fontsize=args.fontsize)
    ax.set_xlim([args.xmin, args.xmax])

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    ax.yaxis.set_major_formatter("{x:.3f}")
    metric_name = args.metric_name.replace("_", " ").capitalize()
    ax.set_ylabel(metric_name, fontsize=args.fontsize)
    ax.set_ylim([args.ymin, args.ymax])

    if args.plot_title is None:
        args.plot_title = f"{metric_name.capitalize()} (test)"
    ax.set_title(args.plot_title, y=1.05, fontsize=args.fontsize)
    ax.legend(loc="upper right")

    plt.savefig(args.out_path, bbox_inches="tight")


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="Wandb entity and project name, e.g. codis/codis",
        default="codis/codis",
    )
    parser.add_argument("--wandb_groups", type=str, nargs="+", help="Wandb group name")
    parser.add_argument(
        "--wandb_names",
        type=str,
        nargs="+",
        help="Wandb run names. Takes precedence over --wandb_groups",
    )
    parser.add_argument("--names", type=str, nargs="+", help="Name for each group.")
    parser.add_argument(
        "--metric_name", type=str, help="Metric name", default="accuracy"
    )
    parser.add_argument(
        "--out_path", type=Path, default=repo_root / "img/joint_training.png"
    )
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate.")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max number of data points."
    )
    parser.add_argument(
        "--show_std", action="store_true", help="Visualize standard deviation."
    )
    parser.add_argument(
        "--plot_title", type=str, default=None, help="Title of the plot."
    )
    parser.add_argument("--xmin", type=float, help="X-axis min limit.")
    parser.add_argument("--xmax", type=float, help="X-axis max limit.")
    parser.add_argument(
        "--xticks", type=str, help="X-axis data", choices=["steps", "tasks"]
    )
    parser.add_argument(
        "--xticks_every", type=int, default=10, help="Granularity of the x axis."
    )
    parser.add_argument("--ymin", type=float, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, help="Y-axis max limit.")
    parser.add_argument("--fontsize", type=int, default=20)
    args = parser.parse_args()
    visualize_loss(args)


if __name__ == "__main__":
    _main()
