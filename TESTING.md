Testing guide
=============

This project includes a pytest-based test suite and a small test runner to execute it.

Quick local run
---------------

1. (Optional) Create and activate a Python virtual environment.
2. Install dependencies:

   - pip install -r requirements.txt

3. Run the tests with coverage (default):

   - python run_tests.py

Run script flags
----------------

- --html: produce HTML coverage in test_reports/coverage_html
- --xml: produce coverage XML at test_reports/coverage.xml
- --edge-cases-only: run only tests marked with pytest.mark.edge (useful for quick edge-case runs)
- --integration-only: run only tests marked with pytest.mark.integration
- --ci: run in CI mode and produce JUnit XML at test_reports/junit.xml

Examples
--------

Run all tests and produce HTML and XML coverage reports:

```powershell
python run_tests.py --html --xml
```

Run only integration tests in CI mode:

```powershell
python run_tests.py --integration-only --ci
```

Notes
-----

- Some integration tests require optional image-processing packages (numpy, opencv-python, pillow, matplotlib, potrace, cairosvg, svgpathtools). Tests that need those packages use pytest.importorskip and will be skipped when the packages are not installed. This keeps the core test suite runnable in minimal CI images.
- Use the GitHub Actions workflow in `.github/workflows/pytest.yml` to run the test suite on push and pull requests. The workflow installs dependencies listed in `requirements.txt`.

Adding tests
------------

- Tests live under `tests/` and follow standard pytest conventions.
- Add new tests and run them locally using `python run_tests.py`.

Coverage
--------

- Coverage is configured via `.coveragerc`. The CI workflow produces a coverage report printed to the log and an XML report for integrations that need it.

README note
-----------

Add a short pointer to this runner in your README under a "Testing" heading to help contributors discover the runner quickly.
