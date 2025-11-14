Fixtures for tests
==================

This directory contains a small generator script `generate_fixtures.py` which
creates lightweight binary fixtures used by the unit tests. The generator
produces:

- `valid_small.png` — a small 50x50 PNG for decoding tests
- `corrupted_small.png` — a truncated version of the PNG to test decode failures
- `not_image.txt` — plain text file to test unsupported-content handling
- `blank.npy` — a saved NumPy blank array useful for array-based tests

To regenerate fixtures locally:

```
python tests/fixtures/generate_fixtures.py --outdir tests/fixtures/output
```

Tests should call the generator or use the generated files as needed. The
generator is intentionally small and avoids producing very large images to
keep the repository lightweight.
