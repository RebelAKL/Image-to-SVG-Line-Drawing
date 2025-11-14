"""Utility helpers for image download, decoding, and validation.

This module centralizes image-related helpers used by the Streamlit app and
future Lambda handler. Functions are defensive and document the exceptions
they raise so callers can handle them consistently.
"""
from typing import Optional
import io
import requests
import numpy as np
import cv2


def safe_decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Safely decode image bytes to an OpenCV numpy array.

    Returns None on decode failure instead of raising.
    """
    if not image_bytes:
        return None
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def download_image_from_url(url: str, timeout: int = 10) -> np.ndarray:
    """Download an image from URL and return as an OpenCV numpy array.

    Raises:
        requests.HTTPError: on non-2xx status codes
        requests.Timeout: on timeout
        ValueError: when downloaded content cannot be decoded to an image
    """
    headers = {'User-Agent': 'ImageToSVG/1.0 (+https://example)'}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    img = safe_decode_image(resp.content)
    if img is None:
        raise ValueError('Downloaded content is not a valid image')
    return img


def is_blank_image(image_array: np.ndarray, threshold: float = 1.0) -> bool:
    """Return True if the image appears blank/solid color.

    Uses standard deviation on the grayscale image. Returns False for None input.
    """
    if image_array is None:
        return False
    try:
        if image_array.ndim == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        return float(gray.std()) < float(threshold)
    except Exception:
        return False


def validate_image_array(image_array: np.ndarray, min_size: int = 10, max_size: int = 10000) -> None:
    """Validate image array and raise descriptive exceptions on failure.

    Raises ValueError or TypeError with clear messages for callers to present to users.
    """
    if image_array is None:
        raise ValueError('image_array is None')

    # Ensure it's a numpy array
    if not hasattr(image_array, 'shape'):
        raise TypeError('image_array must be a numpy array-like with .shape')

    # Size check
    try:
        dims = image_array.shape
    except Exception:
        raise TypeError('Could not access image_array.shape')

    if len(dims) not in (2, 3):
        raise ValueError(f'Invalid image dimensions: expected 2 or 3, got {len(dims)}')

    h, w = int(dims[0]), int(dims[1])
    if h <= 0 or w <= 0:
        raise ValueError('Image has non-positive dimensions')
    if h < min_size or w < min_size:
        raise ValueError(f'Image too small: minimum dimension is {min_size}px')
    if h > max_size or w > max_size:
        raise ValueError(f'Image too large: maximum dimension is {max_size}px')

    # dtype validation
    if not np.issubdtype(image_array.dtype, np.integer) and not np.issubdtype(image_array.dtype, np.floating):
        raise TypeError(f'Image dtype must be numeric, got {image_array.dtype}')

    # Check for NaN/Inf
    if np.isnan(image_array).any() or np.isinf(image_array).any():
        raise ValueError('Image array contains NaN or Inf values')
