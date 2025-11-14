import pytest
import io


def test_download_image_from_url_smoke(monkeypatch):
    """Smoke test for download_image_from_url.

    This test is skipped automatically if optional image libraries (requests, numpy, cv2)
    are not installed in the environment.
    """
    requests = pytest.importorskip("requests")
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")

    # import the function under test
    from utils import download_image_from_url

    # Create a small valid PNG in-memory
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (20, 20), color=(10, 20, 30)).save(buf, format='PNG')
    sample_png = buf.getvalue()

    class DummyResp:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.HTTPError("bad status")

    def fake_get(url, timeout=10, **kwargs):
        return DummyResp(sample_png)

    monkeypatch.setattr(requests, "get", fake_get)

    arr = download_image_from_url("http://example.com/foo.png")
    # If decoding succeeded we expect an array-like with shape
    assert hasattr(arr, "shape")
