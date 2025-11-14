import pytest
import numpy as np


@pytest.fixture
def valid_test_image():
    # Create a simple 800x600 RGB image with a circle and rectangle
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # white background
    img[:] = 255
    # draw a black rectangle and circle using simple numpy operations
    img[100:200, 100:300] = 0
    rr, cc = 300, 400
    for r in range(-50, 51):
        c = int(cc + r * 0.5)
        if 0 <= rr + r < h:
            img[rr + r, max(0, c - 2):min(w, c + 2)] = 0
    return img


@pytest.fixture
def temp_image_file(tmp_path):
    from PIL import Image
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    p = tmp_path / 'temp.png'
    img.save(str(p))
    return str(p)
