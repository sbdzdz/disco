"""Pull images from a wandb run and create a gif."""
from argparse import ArgumentParser
from pathlib import Path

import imageio.v2 as imageio
from tqdm import tqdm

import wandb


def create_gif(args):
    """Pull images and metrics from a wandb run and create custom figures."""
    api = wandb.Api()
    run = api.run(args.run_id)
    assert run.state != "running", "Run is not finished yet."

    path = args.out_path.parent / f"media/{run.id}"
    path.mkdir(parents=True, exist_ok=True)
    for file in tqdm(run.files(), desc="Downloading files"):
        if args.media_name in file.name and file.name.endswith(".png"):
            file.download(root=path, exist_ok=True)

    paths = sorted(
        path.glob(f"media/images/{args.media_name}_*.png"),
        key=lambda x: int(x.stem.split("_")[-2]),
    )

    with imageio.get_writer(args.out_path, mode="I", fps=5) as writer:
        for img_path in tqdm(paths, desc="Creating gif"):
            writer.append_data(imageio.imread(img_path))


def _main():
    repo_root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--out_path", type=Path, default=repo_root / "img/reconstructions.gif"
    )
    parser.add_argument("--media_name", type=str, default="reconstructions")
    args = parser.parse_args()
    create_gif(args)


if __name__ == "__main__":
    _main()
