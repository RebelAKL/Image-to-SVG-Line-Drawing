import os
import importlib.util
import pytest
import numpy as np

# load streamlit_app by path
base = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('streamlit_app', os.path.join(base, '..', 'streamlit_app.py'))
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


def test_end_to_end_pipeline_simple(monkeypatch):
    # Skip if potrace/svgwrite/scour not available
    pytest.importorskip('potrace')
    pytest.importorskip('svgwrite')
    pytest.importorskip('scour')

    # Create a simple image with a black rectangle on white background
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2 = pytest.importorskip('cv2')
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), thickness=-1)

    # Process image
    paths = stream.process_image(img, canny_sigma=0.33, min_contour_frac=0.0001)
    assert paths is not None

    svg = stream.generate_svg_from_paths(paths, width=200, height=200)
    assert '<svg' in svg and 'path' in svg

    opt = stream.optimize_svg_string(svg)
    assert '<svg' in opt
