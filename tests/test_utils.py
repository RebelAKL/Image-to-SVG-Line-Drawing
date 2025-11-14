import importlib.util
import os
import numpy as np
import pytest

# load utils by path to avoid import path issues in test environments
spec = importlib.util.spec_from_file_location('utils', os.path.join(os.path.dirname(__file__), '..', 'utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def test_is_blank_image():
    img = np.ones((50, 50), dtype=np.uint8) * 128
    assert utils.is_blank_image(img, threshold=1.0)
    rng = np.random.RandomState(1)
    noisy = img.copy().astype(float)
    noisy += rng.randint(0, 50, size=img.shape)
    assert not utils.is_blank_image(noisy, threshold=1.0)


def test_safe_decode_image_invalid():
    assert utils.safe_decode_image(b'') is None


def test_validate_image_array_errors():
    with pytest.raises(ValueError):
        utils.validate_image_array(None)
