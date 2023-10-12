"""Class-incremental continual learning dataset."""
from collections import defaultdict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split, Subset

from codis.data import ContinualDSpritesMap
from codis.utils import grouper


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
        self.test_dataset_size = cfg.dataset.test_dataset_size
        self.test_split = cfg.dataset.test_split
        self.train_split = cfg.dataset.train_split
        self.val_split = cfg.dataset.val_split

    def __iter__(self):
        test_dataset = None
        for task_shapes, task_shape_ids, task_exemplars in zip(
            grouper(self.shapes_per_task, self.shapes),
            grouper(self.shapes_per_task, self.shape_ids),
            grouper(self.shapes_per_task, self.exemplars),
        ):
            train_dataset, val_dataset, task_test_dataset = self.build_datasets(
                task_shapes, task_shape_ids
            )

            test_dataset = self.update_test_dataset(test_dataset, task_test_dataset)

            yield (train_dataset, val_dataset, test_dataset), task_exemplars

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

    def update_test_dataset(
        self, test_dataset: Dataset | None, task_test_dataset: Subset
    ):
        """Update the test dataset keeping it class-balanced."""
        samples_per_shape = self.test_dataset_size // (
            self.tasks * self.shapes_per_task
        )

        task_data = [
            task_test_dataset.dataset.data[idx] for idx in task_test_dataset.indices
        ]

        # collect indices per shape and choose samples_per_shape samples randomly
        shape_indices = defaultdict(list)
        for i, factors in enumerate(task_data):
            shape_indices[factors.shape_id].append(i)
        subset_indices = []
        for indices in shape_indices.values():
            subset_indices.extend(np.random.choice(indices, samples_per_shape))

        task_data = [task_data[i] for i in subset_indices]

        if test_dataset is None:
            test_dataset = ContinualDSpritesMap(
                dataset_size=1,
                shapes=self.shapes,
                shape_ids=self.shape_ids,
            )  # dummy dataset
            test_dataset.data = task_data
        else:
            test_dataset.data.extend(task_data)

        return test_dataset
