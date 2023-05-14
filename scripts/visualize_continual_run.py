"""Pull images and metrics from a wandb run and create custom figures."""
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import wandb

plt.style.use("ggplot")


def visualize_wandb_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(f"sebastiandziadzio/codis/{args.run_id}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = download_metrics(run, prefix=args.metric_name)

    num_tasks = run.config["tasks"]
    length = len(list(metrics.values())[0][0])
    steps, task_ids, values = list(metrics.values())[0]

    fig, ax = plt.subplots()
    xdata = [[] for _ in range(num_tasks)]
    ydata = [[] for _ in range(num_tasks)]
    lines = [ax.plot([], [])[0] for _ in range(num_tasks)]

    def init():
        """Initialize the matplotlib animation."""
        ax.set_xlim(-100, length)
        ax.set_ylim(0, 3000)
        return lines

    def update(i):
        """Update the matplotlib animation."""
        for task_id in range(num_tasks):
            if task_ids[i] == task_id:
                xdata[task_id].append(steps[i])
            if task_ids[i] == task_id:
                ydata[task_id].append(values[i])
            lines[task_id].set_data(xdata[task_id], ydata[task_id])
        return lines

    animation = FuncAnimation(
        fig=fig,
        func=update,
        interval=1,
        blit=True,
        repeat=False,
        frames=length,
        init_func=init,
    )
    plt.show()
    animation.save(
        args.output_dir / "training_matplotlib.gif",
        dpi=300,
        writer=PillowWriter(fps=60),
    )


def download_metrics(run, prefix: str = ""):
    """Download all metrics that start with the prefix."""
    assert run.state != "running", "Run is not finished yet."

    keys = [key for key in run.history().keys() if key.startswith(prefix)]

    metrics = {}
    for key in sorted(keys, key=lambda x: x.split("_")[-1]):
        values = [
            (row["_step"], row["task_id"], row[key])
            for row in run.scan_history(keys=["_step", "task_id", key])
        ]
        metrics[key] = tuple(zip(*values))

    return metrics


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=repo_root / "img")
    parser.add_argument("--metric_name", type=str, default="reconstruction_val")
    args = parser.parse_args()
    visualize_wandb_run(args)


if __name__ == "__main__":
    _main()
