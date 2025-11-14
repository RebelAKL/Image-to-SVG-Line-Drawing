## Test Images Dataset — Overview

This dataset provides a structured, manually-curated set of input images and corresponding expected SVG outputs to validate the Image-to-SVG conversion pipeline implemented in `streamlit_app.py`.

The test dataset aims to:
- Validate conversion quality across categories (mechanical, architectural, electronics, tools).
- Stress edge detection and contour filtering with targeted edge cases.
- Provide ground truth SVGs for automated comparisons and parameter optimization.

## Directory Structure

The dataset is organized under two parallel root directories (mirrored structure):

- `test_images/`
  - `mechanical_parts/`
  - `architectural_elements/`
  - `circuit_boards/`
  - `tools/`
  - `edge_cases/`
    - `high_contrast/`
    - `low_contrast/`
    - `complex_geometry/`
    - `simple_geometry/`
    - `high_resolution/`
    - `low_resolution/`
    - `noisy_background/`
    - `minimal_features/`

- `expected_outputs/`  (mirrors `test_images/` exactly; each input image should have a corresponding `.svg` here)

## Test Categories

mechanical_parts/
- Includes gears, bolts, screws, bearings, springs. Tests circular and threaded geometries and fine repetitive details.

architectural_elements/
- Building facades, floor plan fragments, window/door details. Tests straight lines, right angles, and repeating geometric patterns.

circuit_boards/
- PCB photos, IC outlines, fine traces and pads. Tests precision on very thin parallel features.

tools/
- Wrenches, screwdrivers, hammers, measuring tools. Tests mixed curves and straight edges common in manuals.

edge_cases/ (each subdir focuses on a specific challenge)
- `high_contrast/`: Black-on-white or extreme contrast conditions.
- `low_contrast/`: Subtle intensity differences; must test sensitivity.
- `complex_geometry/`: Intricate overlapping shapes and fine details.
- `simple_geometry/`: Basic shapes to ensure no overfitting.
- `high_resolution/`: Very large images to test performance/memory.
- `low_resolution/`: Small images to test robustness with limited pixels.
- `noisy_background/`: Cluttered backgrounds to evaluate contour filtering.
- `minimal_features/`: Very few edges to test lower detection limits and `None` returns.

## Naming Conventions

Files should follow: `{category}_{type}_{number}.{ext}`
Examples:
- `gear_spur_01.jpg`
- `facade_modern_02.png`
- `pcb_traces_03.jpg`

Supported formats: JPG, JPEG, PNG (matching `streamlit_app.py` uploader).

## Metadata Schema

All test cases are described in `test_images/test_metadata.json`. Each key is the relative path to the image (from `test_images/`). Each value is an object with these fields:

- `category`: String, main category or path-like category (e.g., "edge_cases/low_contrast").
- `description`: Short human description.
- `image_type`: Specific classification (e.g., "gear", "bolt", "facade").
- `resolution`: `{width: number, height: number}`.
- `contrast_level`: "high" | "medium" | "low".
- `geometry_complexity`: "simple" | "moderate" | "complex".
- `optimal_parameters`: `{canny_sigma: number, min_contour_frac: number}`.
- `known_challenges`: Array of strings describing difficulties.
- `expected_output`: Relative path to corresponding SVG in `expected_outputs/`.
- `notes`: Free-form notes.

Example entry (see `test_metadata.json` for live examples).

## Ground Truth Creation Process

1. Upload the test image to `streamlit_app.py`.
2. Adjust the `canny_sigma` and `min_contour_frac` sliders until the vector result matches the photo’s key geometric features.
3. Download the optimized SVG from the app.
4. Manually verify the SVG for geometric fidelity (measurements, straightness, preserved fine features).
5. Save the final SVG to `expected_outputs/<category>/` with a matching name.
6. Update `test_images/test_metadata.json` with `optimal_parameters` and notes.

## Adding New Test Cases

1. Add image to the appropriate directory under `test_images/`.
2. Create the expected SVG using the pipeline and save to the mirrored location in `expected_outputs/`.
3. Add an entry to `test_images/test_metadata.json` documenting parameters and any challenges.

## Usage in Validation

This dataset will be used by automated tests to:
- Re-run `streamlit_app.py` (or a headless core processing function) for each image using the `optimal_parameters` in metadata.
- Compare generated SVGs to `expected_outputs/` using geometry-aware diffing (e.g., path comparison tolerances) in future phases.


## Contact / Process Notes

- Keep the `expected_outputs/` SVGs strictly geometric (no styling) for easier automated comparison.
- When in doubt, err on preserving geometry over aesthetics.
