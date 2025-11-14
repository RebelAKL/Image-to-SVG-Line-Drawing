import io
import os
import importlib.util
import pytest

from PIL import Image

# load utils module by path
base = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('utils', os.path.join(base, '..', 'utils.py'))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def _png_bytes():
    buf = io.BytesIO()
    Image.new('RGB', (20, 20), color=(255, 0, 0)).save(buf, format='PNG')
    return buf.getvalue()


def test_download_success(monkeypatch, tmp_path):
    class DummyResp:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise Exception('HTTP error')

    def fake_get(url, timeout=10, headers=None):
        return DummyResp(_png_bytes(), status_code=200)

    monkeypatch.setattr(utils, 'requests', utils.requests)
    monkeypatch.setattr(utils.requests, 'get', fake_get)

    arr = utils.download_image_from_url('http://example.com/image.png')
    assert hasattr(arr, 'shape')


def test_http_error_raises(monkeypatch):
    class DummyResp:
        def raise_for_status(self):
            raise utils.requests.HTTPError('404')

    def fake_get(url, timeout=10, headers=None):
        return DummyResp()

    monkeypatch.setattr(utils, 'requests', utils.requests)
    monkeypatch.setattr(utils.requests, 'get', fake_get)

    with pytest.raises(utils.requests.HTTPError):
        utils.download_image_from_url('http://example.com/notfound.png')


def test_timeout_and_connection_errors(monkeypatch):
    def raise_timeout(url, timeout=10, headers=None):
        raise utils.requests.Timeout('timeout')

    def raise_conn(url, timeout=10, headers=None):
        raise utils.requests.ConnectionError('conn')

    monkeypatch.setattr(utils, 'requests', utils.requests)
    monkeypatch.setattr(utils.requests, 'get', raise_timeout)
    with pytest.raises(utils.requests.Timeout):
        utils.download_image_from_url('http://example.com/timeout.png')

    monkeypatch.setattr(utils.requests, 'get', raise_conn)
    with pytest.raises(utils.requests.ConnectionError):
        utils.download_image_from_url('http://example.com/conn.png')


def test_non_image_content_raises_valueerror(monkeypatch):
    class DummyResp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=10, headers=None):
        return DummyResp(b'not an image')

    monkeypatch.setattr(utils, 'requests', utils.requests)
    monkeypatch.setattr(utils.requests, 'get', fake_get)

    with pytest.raises(ValueError):
        utils.download_image_from_url('http://example.com/notimage')
