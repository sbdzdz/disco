"""Pull images and metrics from a wandb run and create custom figures."""
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio

import wandb


def visualize_wandb_run(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(f"sebastiandziadzio/codis/{args.run_id}")
    assert run.state != "running", "Run is not finished yet."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output_dir / "training.gif", mode="I") as writer:
        for row in run.history(pandas=False):
            if row.get(args.metric_name) is not None:
                img_path = row[args.metric_name]["path"]
                file = run.file(img_path)
                file.download(root=args.output_dir, replace=True)
                img = imageio.imread(args.output_dir / file.name)
                writer.append_data(img)


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=repo_root / "img/wandb")
    parser.add_argument("--metric_name", type=str, default="reconstruction_val")
    args = parser.parse_args()
    visualize_wandb_run(args)


if __name__ == "__main__":
    _main()
