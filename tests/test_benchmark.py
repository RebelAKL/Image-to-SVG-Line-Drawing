import os
import json
import pytest

from benchmark import (
    measure_svg_compression,
    calculate_throughput,
    format_bytes,
    format_time,
    create_synthetic_test_image,
)


def test_measure_svg_compression_basic():
    svg = '<svg><path d="M0,0 L10,10"/></svg>'
    opt = '<svg><path d="M0,0 L10,10"/></svg>'
    orig, opt_sz, ratio = measure_svg_compression(svg, opt)
    assert orig == opt_sz
    assert ratio == 0.0


def test_calculate_throughput():
    t = calculate_throughput((800, 600), 1.0)
    assert t == 480000


def test_format_helpers():
    assert 'KB' in format_bytes(1024)
    assert 'ms' in format_time(0.001)


def test_create_synthetic_image():
    img = create_synthetic_test_image(200, 100, complexity='complex')
    assert img.shape[0] == 100 and img.shape[1] == 200
