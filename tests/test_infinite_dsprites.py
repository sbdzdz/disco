"""Test the infinite dSprites dataset."""
import pytest
import numpy as np

from codis.data import InfiniteDSprites, InfiniteDSpritesAnalogies


@pytest.mark.parametrize("dataset_class", [InfiniteDSprites, InfiniteDSpritesAnalogies])
def test_idsprites_instantiation_with_no_parameters(dataset_class):
    """Test that the dataset can be instantiated with no parameters."""
    dataset = dataset_class()
    assert dataset.img_size == 256
    assert dataset.ranges["color"] == ("white",)
    assert np.allclose(dataset.ranges["scale"], np.linspace(0.5, 1.0, 6))
    assert np.allclose(dataset.ranges["orientation"], np.linspace(0.0, 2 * np.pi, 40))
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0.0, 1.0, 32))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0.0, 1.0, 32))


@pytest.mark.parametrize("dataset_class", [InfiniteDSprites, InfiniteDSpritesAnalogies])
def test_instantiation_from_config(dataset_class):
    """Test that the dataset can be instantiated from a config."""
    config = {
        "img_size": 64,
        "color_range": ["red", "green", "blue"],
        "scale_range": [0.5, 1.0],
        "orientation_range": {
            "start": 0.0,
            "stop": 2 * np.pi,
            "num": 10,
        },
        "position_x_range": np.linspace(0.1, 0.9, 3),
        "position_y_range": np.linspace(0.1, 0.9, 3),
    }
    dataset = dataset_class.from_config(config)
    assert dataset.img_size == 64
    assert dataset.ranges["color"] == ["red", "green", "blue"]
    assert dataset.ranges["scale"] == [0.5, 1.0]
    assert np.allclose(dataset.ranges["orientation"], np.linspace(0.0, 2 * np.pi, 10))
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0.1, 0.9, 3))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0.1, 0.9, 3))


@pytest.mark.parametrize(
    "color_range,shape",
    [(["white"], (1, 256, 256)), (["red"], (3, 256, 256))],
)
def test_idsprites_iteration(color_range, shape):
    """Test that the dataset can be iterated over."""
    dataset = InfiniteDSprites(color_range=color_range)
    images, _ = next(iter(dataset))
    assert images.shape == shape
    assert images.min() >= 0.0
    assert images.max() <= 1.0


@pytest.mark.parametrize(
    "color_range,shape",
    [(["white"], (1, 256, 256)), (["red"], (3, 256, 256))],
)
def test_idsprites_analogies_iteration(color_range, shape):
    """Test that the dataset can be iterated over."""
    dataset = InfiniteDSpritesAnalogies(color_range=color_range)
    images = next(iter(dataset))
    assert images.shape == shape
    assert images.min() >= 0.0
    assert images.max() <= 1.0
