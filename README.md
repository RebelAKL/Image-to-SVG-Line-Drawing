# Image-to-SVG-Line-Drawing

This repository contains a Streamlit-based local test app that converts photos and technical images into line-art SVGs suitable for engineering drawing and documentation.

## New: Metrics & Validation

This project now includes a metrics and validation framework to quantitatively and visually compare generated SVGs against ground-truth expected outputs.

New modules:
- `metrics.py` — core metric calculations (edge precision/recall/F1 with tolerance, SVG statistics, Hausdorff distances).
- `visualization.py` — visual comparison helpers (overlay, comparison, difference map, side-by-side views).
- `validation.py` — batch validation CLI and report generation (JSON/CSV/Markdown).

Validation reports are written to `validation_reports/` by default.

## Streamlit App Enhancements

The Streamlit app (`streamlit_app.py`) was updated to optionally display computed metrics and visual comparisons when ground-truth SVGs are available in `expected_outputs/`. These features are optional and do not break the original upload-and-convert workflow.

## Dependencies

## Parameter Optimization

This project now includes an automated parameter optimization feature to help find good values for the `canny_sigma` and `min_contour_frac` parameters used by the image->SVG pipeline.

- Three search strategies are supported:
	- **Quick Search**: 5×5 grid (25 combinations), fast and suitable for interactive use.
	- **Standard Search**: 10×10 grid (100 combinations), more thorough.
	- **Fine-Tuning**: 7×7 grid centered on current parameters, for local refinement.

- Scoring:
	- When a reference SVG is available, the optimizer uses a composite score combining F1 (50%), inverse Hausdorff (30%), and inverse path count (20%).
	- When no reference is available, a heuristic score is used (edge consistency, path count, file size).

Usage in Streamlit:

1. Open the Streamlit app (`streamlit_app.py`).
2. Expand the "🔍 Auto-Optimize Parameters" panel in the sidebar.
3. Select a search strategy and click "Run Optimization". You may choose to auto-apply the best parameters.

Batch usage:

You can run optimization across the test dataset with the helper in `parameter_optimizer.py`:

```powershell
python -c "from parameter_optimizer import optimize_test_dataset; optimize_test_dataset()"
```

Reports are written to `optimization_reports/` (JSON, CSV, Markdown, PNG heatmaps).


Install all dependencies with:

```powershell
pip install -r requirements.txt
```

New dependencies added:
- scipy
- scikit-image
- cairosvg (system dependency: cairo / libcairo)
- svgpathtools
- matplotlib
- Pillow

Note: On Linux/macOS you may need to install the system Cairo library (e.g., `sudo apt-get install libcairo2` or `brew install cairo`).

## Running Batch Validation

Use the CLI to run validation over test cases defined in `test_images/test_metadata.json`:

```powershell
python validation.py --metadata test_images/test_metadata.json --output-dir validation_reports
```

You can validate a single test case with:

```powershell
python validation.py --single mechanical_parts/gear_spur_01.jpg
```

Reports are generated in JSON, CSV, and Markdown formats.

## Testing

See `TESTING.md` for instructions on running the pytest suite locally and in CI, including the `run_tests.py` helper.

## Notes

- The validation suite is tolerant to missing ground-truth: when a reference SVG is not found, metrics that require it will show as N/A and processing will continue.
- For reproducible results, populate `test_images/` with input images and `expected_outputs/` with manually-verified SVGs, and update `test_images/test_metadata.json` accordingly.

# Image-to-SVG-Line-Drawing