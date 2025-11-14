import io
import os
import pytest
import numpy as np

from PIL import Image

# load utils by path to avoid import path issues in test environments
import importlib.util
spec = importlib.util.spec_from_file_location('utils', os.path.join(os.path.dirname(__file__), '..', 'utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def _create_png_bytes(color=(128, 128, 128), size=(50, 50)):
    buf = io.BytesIO()
    img = Image.new('RGB', size, color=color)
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_corrupted_image_bytes_returns_none():
    b = _create_png_bytes()
    # truncate bytes to simulate corruption
    corrupted = b[:50]
    assert utils.safe_decode_image(corrupted) is None


def test_unsupported_file_format_returns_none():
    txt = b"This is not an image"
    assert utils.safe_decode_image(txt) is None


def test_truncated_file_handling():
    b = _create_png_bytes()
    truncated = b[:-10]
    assert utils.safe_decode_image(truncated) is None


def test_extreme_aspect_ratios_validation():
    # Very wide image: height below min_size should be rejected by validate_image_array
    arr = np.zeros((5, 5000), dtype=np.uint8)
    with pytest.raises(ValueError):
        utils.validate_image_array(arr, min_size=10)

    # Very tall image: width below min_size
    arr2 = np.zeros((5000, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        utils.validate_image_array(arr2, min_size=10)


def test_numerical_anomalies_nan_inf():
    arr = np.ones((100, 100), dtype=float)
    arr[0, 0] = float('nan')
    with pytest.raises(ValueError):
        utils.validate_image_array(arr)

    arr2 = np.ones((100, 100), dtype=float)
    arr2[0, 1] = float('inf')
    with pytest.raises(ValueError):
        utils.validate_image_array(arr2)
import pytest
import numpy as np

from streamlit_app import process_image


def test_none_input():
    with pytest.raises(ValueError):
        process_image(None, 0.33, 0.0001)


def test_empty_array():
    arr = np.array([])
    with pytest.raises(ValueError):
        process_image(arr, 0.33, 0.0001)


def test_blank_white_image():
    arr = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res = process_image(arr, 0.33, 0.0001)
    assert res is None


def test_too_small_image():
    arr = np.ones((5, 5, 3), dtype=np.uint8) * 255
    with pytest.raises(ValueError):
        process_image(arr, 0.33, 0.0001)
