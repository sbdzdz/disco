import matplotlib.pyplot as plt
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import scienceplots  # noqa: F401


def plot_precision_recall(args):
    style = ["science", "vibrant", "grid"]
    plt.rcParams["font.family"] = "Times"
    plt.style.use(style)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))

    left_margin = 0.1
    axes_width = 0.95 * (1 - left_margin - 0.05)
    ax = fig.add_axes([left_margin, 0.1, axes_width, 0.8])

    num_true = 1000
    min_distances, true_labels, _, _ = torch.load(args.path)
    is_seen = true_labels < num_true
    dist_ratio = min_distances[:, 1] / min_distances[:, 0]

    precision, recall = [], []
    for threshold in np.arange(0, 5, 0.01):
        is_correct = (dist_ratio > threshold) == is_seen
        precision.append(is_correct[is_seen].float().mean())
        recall.append(is_correct[~is_seen].float().mean())

    ax.set_ylim([args.ymin, args.ymax])
    ax.plot(recall, precision, label="DCL")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax.legend(loc="right", bbox_to_anchor=(1.45, 0.5), frameon=False)

    plt.savefig(args.out_path, bbox_inches="tight")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--fig_height", type=float, default=3)
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument(
        "--path", type=Path, default=root / "results/precision_recall.pt"
    )
    parser.add_argument(
        "--out_path", type=Path, default=root / "plots/precision_recall.pdf"
    )
    parser.add_argument("--ymin", type=float, default=-0.05, help="Y-axis min limit.")
    parser.add_argument("--ymax", type=float, default=1.05, help="Y-axis max limit.")
    args = parser.parse_args()
    plot_precision_recall(args)
