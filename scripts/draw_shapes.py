"""Draw a batch of canonical shapes."""
from argparse import ArgumentParser

from codis.visualization import draw_shapes


def _main():
    parser = ArgumentParser()
    parser.add_argument("--nrows", type=int, default=2)
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    draw_shapes(nrows=args.nrows, ncols=args.ncols, debug=args.debug)


if __name__ == "__main__":
    _main()
