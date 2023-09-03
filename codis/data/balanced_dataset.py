from collections import Counter
import numpy as np
from torch.utils.data import ConcatDataset, Dataset


class BalancedDataset:
    """Class-balanced reservoir sampling."""

    def __init__(self, max_size: int) -> None:
        """Initialize the class-balanced reservoir sampling.
        Args:
            max_size: The maximum size of the dataset.
        """
        self.max_size = max_size
        self.dataset = None
        self.stored_class_counts = Counter()
        self.seen_class_counts = Counter()
        self.full_classes = set()

    def update(self, task_test_dataset: Dataset) -> None:
        if self.dataset is None:  # empty
            self.dataset = task_test_dataset
            self.update_counts(task_test_dataset)
        elif len(self.dataset) < self.max_size:  # not full yet
            self.dataset = ConcatDataset([self.dataset, task_test_dataset])
            self.update_counts(task_test_dataset)
        else:  # full
            for img, factors in task_test_dataset:
                self.full_classes.update(self.get_largest_classes())
                class_id = factors.shape_id
                if class_id in self.full_classes:
                    u = np.random.uniform()
                    stored = self.stored_class_counts[class_id]
                    seen = self.seen_class_counts[class_id]
                    if u <= stored / seen:
                        idx = self.get_random_instance(class_id)
                        self.dataset.data[idx] = (img, factors)
                else:
                    largest_class = np.random.choice(self.get_largest_classes())
                    idx = self.get_random_instance(largest_class)
                    self.dataset.data[idx] = (img, factors)
                    self.stored_class_counts[class_id] += 1
                    self.stored_class_counts[largest_class] -= 1
                self.seen_class_counts[class_id] += 1

    def update_counts(self, task_dataset: Dataset):
        """Update the counts of the dataset."""
        class_ids = [factors.shape_id for _, factors in task_dataset]
        self.stored_class_counts.update(class_ids)
        self.seen_class_counts.update(class_ids)

    def get_largest_classes(self):
        """Get the largest classes."""
        max_count = max(self.stored_class_counts.values())
        return [
            class_id
            for class_id, count in self.stored_class_counts.items()
            if count == max_count
        ]

    def get_random_instance(self, class_id: int):
        """Return the index of a random instance of a given class."""
        instances = [
            i
            for i, (_, factors) in enumerate(self.dataset)
            if factors.shape_id == class_id
        ]
        return np.random.choice(instances)
