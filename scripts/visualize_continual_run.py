"""Pull images and metrics from a wandb run and create custom figures."""
import io
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb

plt.style.use("ggplot")


def visualize_wandb_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(f"sebastiandziadzio/codis/{args.run_id}")
    metrics = download_metrics(run, prefix=args.metric_name)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    length = len(list(metrics.values())[0][0])
    length = 5

    with imageio.get_writer(args.output_dir / "training.gif", mode="I") as writer:
        for i in tqdm(range(1, length)):
            _, ax = plt.subplots(1, 1)
            for current_task_id, color in zip(range(run.config["tasks"]), colors):
                for steps, task_ids, values in metrics.values():
                    print(steps[:i], task_ids[:i], values[:i])
                    ax.plot(
                        [
                            step
                            for step, task_id in zip(steps[:i], task_ids[:i])
                            if task_id == current_task_id
                        ],
                        [
                            value
                            for task_id, value in zip(task_ids[:i], values[:i])
                            if task_id == current_task_id
                        ],
                        color=color,
                        linewidth=1,
                    )
                    plt.show()
                    buffer = io.BytesIO()
                    plt.savefig(buffer, bbox_inches="tight")
                    plt.close()
                    writer.append_data(imageio.imread(buffer))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output_dir / "training.gif", mode="I") as writer:
        pass


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
