"""Pull images and metrics from a wandb run and create custom figures."""
import os
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import wandb

plt.style.use("ggplot")


def visualize_wandb_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(args.run_id)

    metrics = download_metrics(run, prefix=args.metric_name, subsample=args.subsample)
    filesteps, filenames = maybe_download_media(run, output_dir=args.output_dir)

    num_tasks = run.config["tasks"]
    length = len(list(metrics.values())[0][0])
    steps, task_ids, values = list(metrics.values())[0]

    fig, axes = plt.subplots(ncols=2, figsize=(11, 3), layout="tight")
    xdata = [[] for _ in range(num_tasks)]
    ydata = [[] for _ in range(num_tasks)]
    lines = [axes[0].plot([], [], linewidth=3)[0] for _ in range(num_tasks)]
    lines.append(axes[1].imshow(plt.imread(filenames[0])))

    def init():
        """Initialize the matplotlib animation."""
        axes[0].set_xlim(-10, max(steps))
        axes[0].set_ylim(0, max(values))
        axes[0].set_xlabel("Steps")
        axes[0].set_title("VAE Loss")
        box = axes[0].get_position()
        axes[0].set_position([box.x0, box.y0, box.width, box.height * 0.9])
        axes[1].axis("off")
        return lines

    def update(i):
        """Update the matplotlib animation."""
        for task_id, (x, y) in enumerate(zip(xdata, ydata)):
            if task_ids[i] == task_id:
                x.append(steps[i])
            if task_ids[i] == task_id:
                y.append(values[i])
            lines[task_id].set_data(x, y)
        idx = min(range(len(filesteps)), key=lambda j: abs(filesteps[j] - steps[i]))
        lines[-1].set_data(plt.imread(filenames[idx]))
        return lines

    animation = FuncAnimation(
        fig=fig,
        func=update,
        interval=0.01,
        blit=True,
        repeat=False,
        frames=length,
        init_func=init,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    animation.save(
        args.output_dir / "training_matplotlib.gif",
        dpi=300,
        writer=PillowWriter(fps=30),
    )


def download_metrics(run, prefix: str = "", subsample: int = 1):
    """Download all metrics that start with the prefix."""
    assert run.state != "running", "Run is not finished yet."

    keys = [key for key in run.history().keys() if key.startswith(prefix)]
    keys = sorted(keys, key=lambda x: x.split("_")[-1])

    metrics = {}
    for key in keys:
        values = [
            (row["_step"], row["task_id"], row[key])
            for row in run.scan_history(keys=["_step", "task_id", key])
        ]
        metrics[key] = tuple(zip(*values))

    for key, (steps, task_ids, values) in metrics.items():
        metrics[key] = (
            steps[::subsample],
            task_ids[::subsample],
            values[::subsample],
        )

    return metrics


def maybe_download_media(run, output_dir: Path):
    """Download all images from a run unless they have been downloaded before."""
    assert run.state != "running", "Run is not finished yet."
    path = output_dir / f"media/{run.id}"
    for file in run.files():
        if file.name.endswith(".png"):
            file.download(root=path, exist_ok=True)

    filesteps, filenames = zip(
        *[
            (int(filename.stem.split("_")[1]), filename)
            for filename in sorted(
                path.glob("media/images/*.png"),
                key=lambda x: int(x.stem.split("_")[1]),
            )
        ]
    )
    return filesteps, filenames


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Wandb run ID, including entity and project name, e.g. 'entity/project/run_id'.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=repo_root / "img", help="Output directory."
    )
    parser.add_argument(
        "--metric_name", type=str, default="regressor_loss_train", help="Metric name."
    )
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate.")
    args = parser.parse_args()
    visualize_wandb_run(args)


if __name__ == "__main__":
    _main()
