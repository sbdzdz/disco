from disco.visualization import draw_shapes
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def _main(args):
    path = Path(__file__).parent / "../img/stream"
    path.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(args.num_shapes)):
        draw_shapes(
            path=path / f"shape_{i}",
            nrows=1,
            ncols=1,
            fg_color="whitesmoke",
            img_size=512,
            background_color="darkgray",
            canonical=False,
            seed=i,
        )
        draw_shapes(
            path=path / f"shape_{i}_canonical",
            nrows=1,
            ncols=1,
            fg_color="whitesmoke",
            img_size=512,
            background_color="darkgray",
            canonical=True,
            seed=i,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_shapes", help="How many shapes to draw", default=10)
    args = parser.parse_args()
    _main(args)
