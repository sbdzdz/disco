"""Pull loss values from a few wandb run groups and create an aggregate plot.
Example:
    python scripts/visualize_loss.py --wandb_groups resnet
"""
import json
from argparse import ArgumentParser
from itertools import zip_longest
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import scienceplots


def visualize_metric(args):
    """Visualize a metric across selected wandb runs."""
    api = wandb.Api(timeout=60)
    names = args.wandb_groups if args.names is None else args.names
    assert len(names) == len(
        args.wandb_groups
    ), "Please provide exactly one name for each run group."
    run_dict = {
        name: api.runs(
            args.wandb_entity, filters={"group": group if group != "null" else None}
        )
        for name, group in zip(names, args.wandb_groups)
    }

    if args.wandb_names is not None:
        run_dict = {
            name: [run for run in runs if run.name in args.wandb_names]
            for name, runs in run_dict.items()
        }

    for name, runs in run_dict.items():
        print(f"Found {len(runs)} {'run' if len(runs) == 1 else 'runs'} for {name}.")

    metrics = {}
    for name, runs in run_dict.items():
        values = [load_run(run, args, name) for run in runs]
        max_len = max(map(len, values))
        if name.startswith("test"):
            steps = [i * get_testing_frequency(runs) for i in range(max_len)]
        else:
            steps = list(range(max_len))
        metrics[name] = (steps, values)

    if args.max_steps is not None:
        metrics = truncate(metrics, args.max_steps)

    plot(args, metrics)


def truncate(metrics: dict, max_steps: int):
    """Truncate the metrics to a maximum number of steps.
    Args:
        metrics: A dictionary mapping a metric name to a tuple of (steps, values).
        max_steps: The maximum number of steps to keep.
    Returns:
        A new dictionary mapping a metric name to a tuple of (steps, values) with the
        steps truncated to the maximum number of steps and the values truncated
        accordingly.
    """
    return {
        name: (
            [step for step in steps if step <= max_steps],
            [[v for s, v in zip(steps, value) if s <= max_steps] for value in values],
        )
        for name, (steps, values) in metrics.items()
    }


def get_testing_frequency(runs):
    """Get the frequency at which the model is tested.
    Args:
        runs: A list of wandb run objects.
    Returns:
        The frequency at which the model is tested."""
    try:
        test_every_n_tasks = [
            run.config["training"]["test_every_n_tasks"] for run in runs
        ]
        assert (
            len(set(test_every_n_tasks)) == 1
        ), "All runs must have the same testing frequency."
        return test_every_n_tasks[0]
    except KeyError:
        return 1


def load_run(run, args, name):
    """Load the metric values for a single run."""
    path = Path(__file__).parent.parent / f"img/media/{run.name}/metrics.json"
    if path.exists() and not args.force_download:
        print(f"Found saved run data for {run.name}.")
        with open(path, "r") as f:
            values = json.load(f)
    elif is_baseline(name):
        values = download_baseline_results(run, args.metric_name)
    else:
        values = download_our_results(run, args.metric_name)
    if not path.exists() or args.force_download:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(values, f)
    return values


def is_baseline(name):
    """Check if the run is a baseline run."""
    return "Learning" in name or "Synaptic" in name  # TODO: fix this


def download_baseline_results(run, metric_name):
    """Get the metric values for each baseline run."""
    num_experiences = max(
        row["TrainingExperience"]
        for row in run.history(keys=["TrainingExperience"], pandas=False)
    )
    values = []
    for task in tqdm(
        range(num_experiences),
        desc=f"Downloading run data for {run.name}",
    ):
        metric_name = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{task:03d}"
        values.extend(
            row[metric_name] for row in run.history(keys=[metric_name], pandas=False)
        )
    return values


def download_our_results(run, metric_name: str):
    """Get the metric values for each run."""
    print(f"Downloading run data for {run.name}.")
    return take_last(run.scan_history(keys=["trainer/global_step", metric_name]))


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
    plt.style.use(["science", "bright"])
    _, ax = plt.subplots(layout="tight")
    for name, (steps, values) in metrics.items():
        values_mean = length_agnostic_mean(values)
        ax.plot(steps, values_mean, label=name)
        if args.show_std:
            values_std = np.std(values, axis=0)
            ax.fill_between(
                steps,
                values_mean - values_std,
                values_mean + values_std,
                alpha=0.3,
            )

    ax.set_xlabel("Tasks")
    ax.set_xlim([args.xmin, args.xmax])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.yaxis.set_major_formatter("{x:.1f}")
    metric_name = args.metric_name.split("/")[-1].replace("_", " ").capitalize()
    ax.set_ylabel(metric_name)
    ax.set_ylim([args.ymin, args.ymax])

    if args.plot_title is None:
        args.plot_title = f"{metric_name.capitalize()} (test)"
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.2),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.savefig(args.out_path, bbox_inches="tight")


def length_agnostic_mean(arrays):
    """Compute the mean of arrays of different lengths."""
    result = []
    for values in zip_longest(*arrays):
        values = [v for v in values if v is not None]
        result.append(np.mean(values))
    return result


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
    parser.add_argument("--max_steps", type=int, help="Max number of steps to plot.")
    parser.add_argument(
        "--out_path", type=Path, default=repo_root / "img/joint_training.png"
    )
    parser.add_argument(
        "--force_download", action="store_true", help="Don't use cached data."
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
    parser.add_argument("--ymin", type=float, default=-0.05, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, default=1.05, help="Y-axis max limit.")
    parser.add_argument("--fontsize", type=int, default=20)
    args = parser.parse_args()
    visualize_metric(args)


if __name__ == "__main__":
    _main()
