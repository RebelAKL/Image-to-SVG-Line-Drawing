"""Generate small binary fixtures used by the test suite.

Usage:
    python generate_fixtures.py --outdir ./output

This script creates a few small images and sample corrupted files suitable
for unit tests. It is intentionally conservative about sizes to keep the
repo small while allowing reproducible test runs.
"""
import os
import argparse
from PIL import Image
import numpy as np


def make_png(path, size=(50, 50), color=(128, 128, 128)):
    img = Image.new('RGB', size, color=color)
    img.save(path, format='PNG')


def make_corrupted_png(valid_path, out_path, truncate_bytes=50):
    with open(valid_path, 'rb') as f:
        data = f.read()
    with open(out_path, 'wb') as f:
        f.write(data[:truncate_bytes])


def make_text_file(path, text='not an image'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def make_numpy_image(path, shape=(100, 100), dtype='uint8'):
    arr = np.zeros(shape, dtype=dtype)
    np.save(path, arr)


def create_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    valid = os.path.join(outdir, 'valid_small.png')
    make_png(valid)
    make_corrupted_png(valid, os.path.join(outdir, 'corrupted_small.png'))
    make_text_file(os.path.join(outdir, 'not_image.txt'))
    make_numpy_image(os.path.join(outdir, 'blank.npy'))
    return outdir


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', '-o', default='output')
    args = p.parse_args()
    print('Creating fixtures in', args.outdir)
    create_all(args.outdir)
