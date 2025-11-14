# Sample Images Guide — sourcing and creation

This guide explains where to find or how to create high-quality test images for the test suite.

## 1. Image Sourcing Recommendations

Free stock photo sources (search examples in parentheses):
- Unsplash ("gear", "bolt", "circuit board macro")
- Pexels ("window frame", "building facade", "tool flat lay")
- Pixabay ("screwdriver", "wrench", "PCB closeup")

Technical documentation and CAD screenshots:
- Export orthographic CAD renderings (PNG) from CAD software for mechanical parts.
- Capture screenshot exports from PCB design tools (EAGLE, KiCAD) for circuit boards.

Synthetic creation:
- Use vector editors (Inkscape, Illustrator) to render synthetic photographs or shapes for simple/controlled edge cases.

## 2. Image Requirements

Supported formats: JPG, JPEG, PNG.

Recommended resolution ranges:
- Low-res tests: 320x240 or smaller.
- Medium-res tests: 800x600 to 1600x1200.
- High-res tests: 3000x2000 and above.

Background: Plain neutral backgrounds are preferred for baseline tests. Edge-case sets should include cluttered or textured backgrounds.

Lighting and contrast:
- Even lighting with minimal specular highlights for mechanical parts.
- Soft lighting for low-contrast tests; strong, directional lighting for high-contrast tests.

## 3. Minimum Test Set

- 3 images per main category (mechanical_parts, architectural_elements, circuit_boards, tools) = 12 images
- 2 images per edge case subdirectory (8 subdirs x 2 = 16 images)
- Minimum total ≈ 28 (round to 30 for buffer)

## 4. Comprehensive Test Set

- 10+ images per main category = 40+
- 5+ per edge-case subdir = 40+
- Comprehensive total ≈ 100+ images

## 5. Specific Image Suggestions

Mechanical parts:
- Clock gear (top-down), hex bolt, compression spring, bearing race, threaded rod segment.

Architectural elements:
- Modern facade with grid windows, floor plan section (cropped), detailed door/jamb photo.

Circuit boards:
- Macro of traces and pads, IC with legible outline, connector edge with solder pads.

Tools:
- Flat lay of hand tools (wrench, screwdriver), caliper close-up, hammer head on neutral background.

Photography tips:
- Flat-lay for tools and parts: camera perpendicular to plane.
- Use tripod and uniform lighting to reduce motion blur.
- Isolate object on a neutral background when possible.

## 6. Edge Case Creation

How to create artificial edge cases using image editors:
- Low contrast: Decrease contrast/brightness in editor or overlay a translucent layer.
- Add noise: Use Gaussian or salt-and-pepper noise filters to simulate texture.
- Resize down/up to create low and high resolution variants.
- Blur: Apply slight Gaussian blur to test tolerance of edge detectors.

## Quick checklist before adding to `test_images/`
- Filename follows naming convention.
- Image format is supported (png/jpg).
- Metadata entry added or prepared for `test_images/test_metadata.json`.
- Expected output placeholder path created in `expected_outputs/`.


Good luck building the dataset — this guide should make it straightforward to assemble a robust suite that exercises the `streamlit_app.py` pipeline thoroughly.