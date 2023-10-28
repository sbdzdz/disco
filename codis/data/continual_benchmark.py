"""Class-incremental continual learning dataset."""
from collections import defaultdict
from itertools import zip_longest

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, random_split

from codis.data import ContinualDSpritesMap


class ContinualBenchmark:
    def __init__(self, cfg: DictConfig, shapes: list, exemplars: list):
        self.shapes = shapes
        self.shape_ids = range(len(shapes))
        self.exemplars = exemplars

        self.batch_size = cfg.dataset.batch_size
        self.factor_resolution = cfg.dataset.factor_resolution
        self.img_size = cfg.dataset.img_size
        self.num_workers = cfg.dataset.num_workers
        self.shapes_per_task = cfg.dataset.shapes_per_task
        self.tasks = cfg.dataset.tasks
        self.train_dataset_size = cfg.dataset.train_dataset_size
        self.val_dataset_size = cfg.dataset.val_dataset_size
        self.test_dataset_size = cfg.dataset.test_dataset_size
        self.test_split = cfg.dataset.test_split
        self.train_split = cfg.dataset.train_split
        self.val_split = cfg.dataset.val_split

    def __iter__(self):
        test_dataset = None
        for task_shapes, task_shape_ids, task_exemplars in zip(
            self.grouper(self.shapes, self.shapes_per_task),
            self.grouper(self.shape_ids, self.shapes_per_task),
            self.grouper(self.exemplars, self.shapes_per_task),
        ):
            train_dataset, val_dataset, task_test_dataset = self.build_datasets(
                task_shapes, task_shape_ids
            )
            test_dataset = self.update_dataset(test_dataset, task_test_dataset)
            yield (train_dataset, val_dataset, test_dataset), task_exemplars

    @staticmethod
    def grouper(iterable, n):
        """Iterate in groups of n elements, e.g. grouper(3, 'ABCDEF') --> ABC DEF.
        Args:
            n: The number of elements per group.
            iterable: The iterable to be grouped.
        Returns:
            An iterator over the groups.
        """
        args = [iter(iterable)] * n
        return (list(group) for group in zip_longest(*args))

    def build_datasets(self, shapes: list, shape_ids: list):
        """Build data loaders for a class-incremental continual learning scenario."""
        n = self.factor_resolution
        scale_range = np.linspace(0.5, 1.0, n)
        orientation_range = np.linspace(0, 2 * np.pi * (n / (n + 1)), n)
        position_x_range = np.linspace(0, 1, n)
        position_y_range = np.linspace(0, 1, n)

        dataset = ContinualDSpritesMap(
            img_size=self.img_size,
            shapes=shapes,
            shape_ids=shape_ids,
            scale_range=scale_range,
            orientation_range=orientation_range,
            position_x_range=position_x_range,
            position_y_range=position_y_range,
        )
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [
                self.train_split,
                self.val_split,
                self.test_split,
            ],
        )
        return train_dataset, val_dataset, test_dataset

    def update_dataset(
        self, dataset: Dataset | None, task_dataset: Subset, total_size: int
    ):
        """Update the dataset keeping it class-balanced.
        Args:
            dataset: The cumulative dataset.
            task_dataset: The test dataset for the current task.
            total_size: The total size of the cumulative dataset.
        """
        samples_per_shape = total_size // (self.tasks * self.shapes_per_task)

        # collect indices per shape and choose samples_per_shape samples randomly
        task_data = [task_dataset.dataset.data[idx] for idx in task_dataset.indices]
        shape_indices = defaultdict(list)
        for i, factors in enumerate(task_data):
            shape_indices[factors.shape_id].append(i)
        subset_indices = []
        for indices in shape_indices.values():
            subset_indices.extend(np.random.choice(indices, samples_per_shape))

        task_data = [task_data[i] for i in subset_indices]

        if dataset is None:
            dataset = ContinualDSpritesMap(
                img_size=self.img_size,
                dataset_size=1,
                shapes=self.shapes,
                shape_ids=self.shape_ids,
            )  # dummy dataset
            dataset.data = task_data
        else:
            dataset.data.extend(task_data)

        return dataset


class ContinualBenchmarkRehearsal(ContinualBenchmark):
    def __init__(self, cfg: DictConfig, shapes: list, exemplars: list):
        super().__init__(cfg, shapes, exemplars)
        self.rehearsal_dataset_size = cfg.dataset.rehearsal_dataset_size

    def __iter__(self):
        train_dataset = None
        val_dataset = None
        test_dataset = None
        for task_shapes, task_shape_ids, task_exemplars in zip(
            self.grouper(self.shapes, self.shapes_per_task),
            self.grouper(self.shape_ids, self.shapes_per_task),
            self.grouper(self.exemplars, self.shapes_per_task),
        ):
            task_train_dataset, val_dataset, task_test_dataset = self.build_datasets(
                task_shapes, task_shape_ids
            )
            train_dataset = self.update_dataset(
                train_dataset, task_train_dataset, self.train_dataset_size
            )
            val_dataset = self.update_dataset(
                val_dataset, task_train_dataset, self.val_dataset_size
            )
            test_dataset = self.update_dataset(
                test_dataset, task_test_dataset, self.test_dataset_size
            )
            yield (train_dataset, val_dataset, test_dataset), task_exemplars
