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
import scienceplots  # noqa: F401


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
        values = [load_run(run, args) for run in runs]
        max_len = max(map(len, values))
        if args.metric_name.startswith("test"):
            steps = [i * get_testing_frequency(runs) for i in range(max_len)]
        else:
            steps = list(range(max_len))
        metrics[name] = (steps, values)

    if args.max_steps is not None:
        metrics = truncate(metrics, args.max_steps)

    plot(args, metrics)


def truncate(metrics: dict, max_steps: int):
    """Truncate the metrics to a maximum number of steps."""
    return {
        name: (
            [step for step in steps if step <= max_steps],
            [[v for s, v in zip(steps, value) if s <= max_steps] for value in values],
        )
        for name, (steps, values) in metrics.items()
    }


def get_testing_frequency(runs):
    """Get the frequency at which the model is tested."""
    try:
        test_every_n_tasks = [
            run.config["training"]["test_every_n_tasks"] for run in runs
        ]
        assert (
            len(set(test_every_n_tasks)) == 1
        ), "All runs must have the same testing frequency."
        return test_every_n_tasks[0]
    except KeyError:
        return 10


def load_run(run, args):
    """Load the metric values for a single run."""
    path = (
        Path(__file__).parent.parent / f"img/media/{run.name}/{args.metric_name}.json"
    )
    if path.exists() and not args.force_download:
        print(f"Found saved run data for {run.name}.")
        with open(path, "r") as f:
            values = json.load(f)
    else:
        values = download_results(run, args.metric_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(values, f)
    return values


def download_results(run, metric_name: str):
    """Get the metric values for each run."""
    print(f"Downloading run data for {run.name}.")
    scan = run.scan_history(keys=["task", metric_name])
    return [row[metric_name] for row in scan]


def plot(args, metrics):
    style = ["science", "vibrant"]
    if args.grid:
        style.append("grid")
    plt.style.use(style)
    plt.rcParams["font.family"] = "Times"

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))

    left_margin = 0.1
    axes_width = 0.95 * (1 - left_margin - 0.05)
    ax = fig.add_axes([left_margin, 0.1, axes_width, 0.8])

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
    if args.include_contrastive:
        path = Path("sebastian/hostname_4570651.out")
        values = get_test_acc(path, end_task=args.max_steps)
        values = values[::10] + [values[-1]]
        steps = [10 * i for i in range(len(values))]
        ax.plot(steps, values, label="Contrastive")

    ax.set_xlabel("Tasks")
    ax.set_xlim([args.xmin, args.xmax])

    ax.yaxis.set_major_formatter("{x:.1f}")
    metric_name = args.metric_name.split("/")[-1].replace("_", " ").capitalize()
    ax.set_ylabel(metric_name)
    ax.set_ylim([args.ymin, args.ymax])

    if args.plot_title is None:
        args.plot_title = f"{metric_name.capitalize()} (test)"

    ax.legend(loc=args.legend_loc, bbox_to_anchor=(1.45, 0.5), frameon=False)

    plt.savefig(args.out_path, bbox_inches="tight")


def get_test_acc(path, end_task=-1):
    with open(path, "r") as f:
        lines = f.readlines()
        texts = ""
        for line in lines:
            texts += line[:-1] if line[-1] == "\n" else line

    texts = texts.split("task_id")[1:]
    corrects = [text.split("[")[1].split("]")[0].split(", ") for text in texts]
    counts = [text.split("[")[2].split("]")[0].split(", ") for text in texts]
    corrects = [
        np.array([float(x) for x in corrects_]) for corrects_ in corrects[:end_task]
    ]
    counts = [np.array([float(x) for x in count]) for count in counts[:end_task]]
    corrects = [corrects_.sum() for corrects_ in corrects]
    counts = [counts_.sum() for counts_ in counts]
    accs = [corrects[i] / counts[i] for i in range(len(corrects))]
    return accs


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
        help="Wandb entity and project name, e.g. disco/disco",
        default="codis/disco",
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
    parser.add_argument("--out_path", type=Path, default=repo_root / "plots/metric.pdf")
    parser.add_argument(
        "--force_download", action="store_true", help="Don't use cached data."
    )
    parser.add_argument(
        "--show_std", action="store_true", help="Visualize standard deviation."
    )
    parser.add_argument("--grid", action="store_true", help="Add grid to the plot.")
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
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    parser.add_argument(
        "--legend_loc",
        type=str,
        default="best",
        help="Location of the legend.",
    )
    parser.add_argument(
        "--include_contrastive",
        action="store_true",
        help="Include contrastive baseline.",
    )
    args = parser.parse_args()
    visualize_metric(args)


if __name__ == "__main__":
    _main()
