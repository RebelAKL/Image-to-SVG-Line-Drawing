import pytest
import numpy as np

from utils import validate_image_array, is_blank_image


def test_validate_image_array_none():
    with pytest.raises(ValueError):
        validate_image_array(None)


def test_validate_image_array_too_small():
    arr = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_image_array(arr, min_size=10)


def test_is_blank_image_true():
    arr = np.zeros((100, 100), dtype=np.uint8)
    assert is_blank_image(arr, threshold=0.5)


def test_is_blank_image_false():
    # Create a clearly noisy image (variance >> threshold)
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(100, 100), dtype=np.uint8)
    assert not is_blank_image(arr, threshold=0.5)
