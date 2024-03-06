"""Pull images and metrics from a wandb run and create custom figures."""
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

import wandb

plt.style.use("ggplot")


def visualize_stn_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(args.run_id)
    metrics = download_stn_metrics(run, args)
    fig, axes = plt.subplots(1, 2, figsize=(32, 9), layout="tight")
    lines = {
        "loss_val": axes[0].plot([], [], linewidth=2, label="Loss (val)")[0],
        "loss_train": axes[0].plot([], [], linewidth=2, label="Loss (train)")[0],
        "img": axes[1].imshow(plt.imread(metrics["img_paths"][0])),
    }

    def init():
        """Initialize the matplotlib animation."""
        steps, values = metrics["loss_train_steps"], metrics["loss_train_values"]
        axes[0].set_xlim(0, max(steps))
        axes[0].set_ylim(0, max(values))
        axes[0].set_title("Loss")
        axes[0].legend(loc="upper right")
        axes[0].set_xlabel("Steps")

        for task_transition in metrics["task_transitions"]:
            axes[0].axvline(task_transition, color="gray", linestyle="dotted")

        axes[1].axis("off")

        return list(lines.values())

    def update(i):
        """Update the matplotlib animation."""
        current_step = metrics["loss_train_steps"][i]
        lines["loss_train"].set_data(
            metrics["loss_train_steps"][:i], metrics["loss_train_values"][:i]
        )
        j = find_nearest_step_index(metrics["loss_val_steps"], current_step)
        lines["loss_val"].set_data(
            metrics["loss_val_steps"][:j], metrics["loss_val_values"][:j]
        )
        j = find_nearest_step_index(metrics["img_steps"], current_step)
        lines["img"].set_data(plt.imread(metrics["img_paths"][j]))

        return list(lines.values())

    length = len(metrics["loss_train_steps"])
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
    with tqdm(total=length) as progress_bar:

        def callback(current_frame: int, total_frames: int) -> None:
            # pylint: disable=unused-argument
            progress_bar.update(1)

        animation.save(
            args.output_dir / "training_stn.gif",
            dpi=args.dpi,
            writer=PillowWriter(fps=30),
            progress_callback=callback,
        )


def download_stn_metrics(run, args):
    """Download all metrics required to build the figure."""
    metrics = {}
    metrics["loss_train_steps"], metrics["loss_train_values"] = download_metric(
        run, name="loss_train", subsample=args.subsample
    )
    metrics["loss_val_steps"], metrics["loss_val_values"] = download_metric(
        run, name="loss_val", subsample=1
    )
    task_id_steps, task_id_values = download_metric(
        run, name="task_id", subsample=args.subsample
    )
    metrics["task_transitions"] = [
        task_id_steps[i]
        for i in range(len(task_id_steps) - 1)
        if task_id_values[i] != task_id_values[i + 1]
    ]

    metrics["img_steps"], metrics["img_paths"] = maybe_download_media(
        run, output_dir=args.output_dir, name="reconstructions_exemplars"
    )
    return metrics


def visualize_vae_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(args.run_id)
    metrics = download_vae_metrics(run, args)

    fig, axes = plt.subplot_mosaic("AC;BC", figsize=(32, 9), layout="tight")
    lines = {
        "vae_val": axes["A"].plot([], [], linewidth=2, label="VAE loss (val)")[0],
        "vae_train": axes["A"].plot([], [], linewidth=2, label="VAE loss (train)")[0],
        "mlp_val": axes["B"].plot([], [], linewidth=2, label="MLP loss (val)")[0],
        "mlp_train": axes["B"].plot([], [], linewidth=2, label="MLP loss (train)")[0],
        "img": axes["C"].imshow(plt.imread(metrics["img_paths"][0])),
    }

    def init():
        """Initialize the matplotlib animation."""
        for ax_name, steps, values, title in [
            ("A", metrics["vae_train_steps"], metrics["vae_train_values"], "VAE Loss"),
            ("B", metrics["mlp_train_steps"], metrics["mlp_train_values"], "MLP Loss"),
        ]:
            axes[ax_name].set_xlim(-50, max(steps))
            axes[ax_name].set_ylim(0, max(values))
            axes[ax_name].set_title(title)
            axes[ax_name].legend(loc="upper right")
            axes[ax_name].set_xlabel("Steps")

            for task_transition in metrics["task_transitions"]:
                axes[ax_name].axvline(task_transition, color="gray", linestyle="dotted")

        axes["C"].axis("off")

        return list(lines.values())

    def update(i):
        """Update the matplotlib animation."""
        current_step = metrics["vae_train_steps"][i]
        lines["vae_train"].set_data(
            metrics["vae_train_steps"][:i], metrics["vae_train_values"][:i]
        )
        j = find_nearest_step_index(metrics["vae_val_steps"], current_step)
        lines["vae_val"].set_data(
            metrics["vae_val_steps"][:j], metrics["vae_val_values"][:j]
        )

        lines["mlp_train"].set_data(
            metrics["mlp_train_steps"][:i], metrics["mlp_train_values"][:i]
        )
        j = find_nearest_step_index(metrics["mlp_val_steps"], current_step)
        lines["mlp_val"].set_data(
            metrics["mlp_val_steps"][:j], metrics["mlp_val_values"][:j]
        )

        j = find_nearest_step_index(metrics["img_steps"], current_step)
        lines["img"].set_data(plt.imread(metrics["img_paths"][j]))

        return list(lines.values())

    length = len(metrics["vae_train_steps"])
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
    with tqdm(total=length) as progress_bar:

        def callback(current_frame: int, total_frames: int) -> None:
            progress_bar.update(1)

        animation.save(
            args.output_dir / "training_vae.gif",
            dpi=args.dpi,
            writer=PillowWriter(fps=30),
            progress_callback=callback,
        )


def find_nearest_step_index(steps, step):
    """Find the index of the nearest step in a list of steps.
    Args:
        steps: List of steps.
        step: Step to find the nearest step to.
    """
    return min(range(len(steps)), key=lambda j: abs(steps[j] - step))


def download_vae_metrics(run, args):
    """Download all metrics required to build the figure.
    Args:
        run: Wandb run object.
        args: Command line arguments.
    """
    metrics = {}

    metrics["vae_train_steps"], metrics["vae_train_values"] = download_metric(
        run, name="backbone_loss_train", subsample=args.subsample
    )
    metrics["vae_val_steps"], metrics["vae_val_values"] = download_metric(
        run, name="backbone_loss_val", subsample=1
    )

    metrics["mlp_train_steps"], metrics["mlp_train_values"] = download_metric(
        run, name="regressor_loss_train", subsample=args.subsample
    )
    metrics["mlp_val_steps"], metrics["mlp_val_values"] = download_metric(
        run, name="regressor_loss_val", subsample=1
    )

    task_id_steps, task_id_values = download_metric(
        run, name="task_id", subsample=args.subsample
    )
    metrics["task_transitions"] = [
        task_id_steps[i + 1]
        for i in range(len(task_id_steps) - 1)
        if task_id_values[i] != task_id_values[i + 1]
    ]

    metrics["img_steps"], metrics["img_paths"] = maybe_download_media(
        run, output_dir=args.output_dir, name="reconstructions"
    )

    return metrics


def download_metric(run, name, subsample: int = 1):
    """Download a metric from a wandb run.
    Args:
        run: Wandb run object.
        name: Name of the metric to download.
        subsample: Subsample rate.
    """
    assert run.state != "running", "Run is not finished yet."

    steps, values = zip(
        *[(row["_step"], row[name]) for row in run.scan_history(keys=["_step", name])]
    )
    steps = steps[::subsample]
    values = values[::subsample]

    return steps, values


def maybe_download_media(run, output_dir: Path, name: str = ""):
    """Download all images from a run unless they have been downloaded before.
    Args:
        run: Wandb run object.
        output_dir: Where to download the files.
    """
    assert run.state != "running", "Run is not finished yet."
    path = output_dir / f"media/{run.id}"
    for file in run.files():
        if name in file.name and file.name.endswith(".png"):
            file.download(root=path, exist_ok=True)

    img_paths = list(
        sorted(
            path.glob(f"media/images/{name}*.png"),
            key=lambda x: int(x.stem.split("_")[-2]),
        )
    )
    img_steps = [int(img_path.stem.split("_")[-2]) for img_path in img_paths]

    return img_steps, img_paths


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
        "--dpi", type=int, default=150, help="DPI of the output figure."
    )
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate.")
    parser.add_argument(
        "--model",
        type=str,
        default="stn",
        choices=["vae", "stn"],
        help="Model type (VAE or STN).",
    )
    args = parser.parse_args()
    if args.model == "vae":
        visualize_vae_run(args)
    elif args.model == "stn":
        visualize_stn_run(args)
    else:
        raise ValueError(f"Unknown experiment type: {args.model}")


if __name__ == "__main__":
    _main()
