"""Test for the balanced dataset."""
from collections import Counter

import numpy as np
import pytest

from codis.data import BalancedDataset, ContinualDSpritesMap, InfiniteDSprites
from codis.utils import grouper


@pytest.mark.parametrize("num_tasks", [1, 2, 5])
# @pytest.mark.parametrize("shapes_per_task", [10, 20, 50])
# @pytest.mark.parametrixe("max_size", [500])
def test_balanced_dataset(
    num_tasks: int, shapes_per_task: int = 10, max_size: int = 100
):
    """Test the balanced dataset."""
    dataset = InfiniteDSprites()
    balanced_dataset = BalancedDataset(max_size=max_size)
    shapes = [dataset.generate_shape() for _ in range(num_tasks * shapes_per_task)]
    resolution = 4
    datasets = [
        ContinualDSpritesMap(
            shapes=shapes,
            scale_range=np.linspace(0.5, 1.0, resolution),
            orientation_range=np.linspace(0, 1, resolution),
            position_x_range=np.linspace(0, 1, resolution),
            position_y_range=np.linspace(0, 1, resolution),
        )
        for shapes in grouper(shapes_per_task, shapes)
    ]
    for task_dataset in datasets:
        balanced_dataset.update(task_dataset)

    class_ids = [factors.shape_id for _, factors in balanced_dataset.dataset]
    class_counts = Counter(class_ids)
    samples_per_class = max_size // (num_tasks * shapes_per_task)

    assert len(balanced_dataset.dataset) <= max_size
    assert len(class_counts) == num_tasks * shapes_per_task
    assert all(count == samples_per_class for count in class_counts.values())
