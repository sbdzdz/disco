from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import scienceplots  # noqa: F401
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_gpt_benchmark(args):
    df = pd.read_csv(args.path)
    style = ["science", "vibrant", "grid"]
    plt.style.use(style)
    plt.rcParams["font.family"] = "Times"

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))

    left_margin = 0.1
    axes_width = 0.95 * (1 - left_margin - 0.05)
    ax = fig.add_axes([left_margin, 0.1, axes_width, 0.8])

    ax.plot(
        df["num_choices"],
        df["accuracy"],
        label="GPT-4 Vision",
        marker="o",
        linewidth=1,
        markersize=3,
        zorder=3,
    )
    ax.plot(
        df["num_choices"],
        [1 / n for n in df["num_choices"]],
        color="gray",
        linestyle=":",
        marker="o",
        markersize=3,
        linewidth=1,
        label="Random",
        zorder=2,
    )

    ax.set_xlabel("Number of choices")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc="right", bbox_to_anchor=(1.45, 0.5), frameon=False)

    ax.set_ylim([args.ymin, args.ymax])
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter("{x:.1f}")

    plt.savefig(args.out_path, bbox_inches="tight")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=root / "results/gpt_results.csv",
        help="Path to the .csv file with benchmark results.",
    )
    parser.add_argument("--ymin", type=float, default=-0.05)
    parser.add_argument("--ymax", type=float, default=1.05)
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    parser.add_argument("--out_path", type=Path, default=root / "plots/gpt.pdf")
    args = parser.parse_args()
    plot_gpt_benchmark(args)
