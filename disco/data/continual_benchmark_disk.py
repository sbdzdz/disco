"""Continual benchmark from files on disk."""

from pathlib import Path
from typing import Union

from torchvision.io import read_image
from disco.data import FileDataset
from torch.utils.data import ConcatDataset
import numpy as np


class ContinualBenchmarkDisk:
    def __init__(
        self,
        path: Union[Path, str],
        accumulate_test_set: bool = True,
    ):
        """Initialize the continual learning benchmark.
        Args:
            path: The path to the dataset.
            accumulate_test_set: Whether to accumulate the test set over tasks.
        """
        self.path = Path(path)
        self.accumulate_test_set = accumulate_test_set
        if self.accumulate_test_set:
            self.test_sets = []

    def __iter__(self):
        for task_dir in sorted(
            self.path.glob("task_*"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            task_exemplars = self.load_exemplars(task_dir)
            train = FileDataset(task_dir / "train")
            val = FileDataset(task_dir / "val")
            test = FileDataset(task_dir / "test")

            if self.accumulate_test_set:
                self.test_sets.append(test)
                test = ConcatDataset(self.test_sets)

            yield (train, val, test), task_exemplars

    def load_exemplars(self, task_dir):
        """Load the current task exemplars from a given directory."""
        paths = (task_dir / "exemplars").glob("exemplar_*.png")
        return [np.array(read_image(str(path)) / 255.0) for path in paths]
