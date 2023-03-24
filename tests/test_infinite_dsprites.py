"""Test the infinite dSprites dataset."""
import numpy as np

from codis.data import InfiniteDSprites


def test_instantiation_with_no_parameters():
    """Test that the dataset can be instantiated with no parameters."""
    dataset = InfiniteDSprites()
    assert dataset.min_verts == 3
    assert dataset.max_verts == 10
    assert dataset.radius_std == 0.6
    assert dataset.angle_std == 0.8
    assert dataset.image_size == 64
    assert dataset.ranges["color"] == ("white",)
    assert np.allclose(dataset.ranges["scale"], np.linspace(0.5, 1.0, 6))
    assert np.allclose(dataset.ranges["orientation"], np.linspace(0.0, 2 * np.pi, 40))
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0.0, 1.0, 32))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0.0, 1.0, 32))


def test_instantiation_from_config():
    """Test that the dataset can be instantiated from a config."""
    config = {
        "image_size": 64,
        "color_range": ["red", "green", "blue"],
        "scale_range": [0.5, 1.0],
        "orientation_range": {
            "start": 0.0,
            "stop": 2 * np.pi,
            "num": 10,
        },
        "position_x_range": np.linspace(0.1, 0.9, 3),
        "positon_y_range": np.linspace(0.1, 0.9, 3),
    }
    dataset = InfiniteDSprites.from_config(config)
    assert dataset.image_size == 64
    assert dataset.ranges["color"] == ["red", "green", "blue"]
    assert dataset.ranges["scale"] == [0.5, 1.0]
    assert np.allclose(dataset.ranges["orientation"], np.linspace(0.0, 2 * np.pi, 10))
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0.1, 0.9, 3))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0.1, 0.9, 3))
