import os
import importlib.util
import pytest
import numpy as np

# load utils and streamlit_app by path to avoid import path issues
base = os.path.dirname(__file__)
spec_u = importlib.util.spec_from_file_location('utils', os.path.join(base, '..', 'utils.py'))
utils = importlib.util.module_from_spec(spec_u)
spec_u.loader.exec_module(utils)

spec_s = importlib.util.spec_from_file_location('streamlit_app', os.path.join(base, '..', 'streamlit_app.py'))
stream = importlib.util.module_from_spec(spec_s)
spec_s.loader.exec_module(stream)


def test_validate_image_array_dimensions_and_dtype():
    # 1D array
    arr1 = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.validate_image_array(arr1)

    # non-numeric dtype
    arr2 = np.array([["a"] * 10] * 10, dtype=object)
    with pytest.raises(TypeError):
        utils.validate_image_array(arr2)


def test_validate_min_max_size():
    small = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        utils.validate_image_array(small, min_size=10)

    large = np.zeros((20000, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        utils.validate_image_array(large, max_size=10000)


def test_process_image_blank_and_invalid(monkeypatch):
    # Blank image should return None
    blank = np.zeros((100, 100), dtype=np.uint8)
    res = stream.process_image(blank, canny_sigma=0.33, min_contour_frac=0.0001)
    assert res is None

    # Invalid image type should raise when validate_image_array present
    with pytest.raises(ValueError):
        stream.process_image(None, canny_sigma=0.33, min_contour_frac=0.0001)

    # Simulate utils unavailable: monkeypatch validate_image_array to None
    monkeypatch.setattr(stream, 'validate_image_array', None)
    # Now pass a 1D array - fallback should reject consistently
    with pytest.raises(ValueError):
        stream.process_image(np.array([1, 2, 3]), canny_sigma=0.33, min_contour_frac=0.0001)
