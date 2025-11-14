import io
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    import cairosvg
    import matplotlib.pyplot as plt
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    cairosvg = None
    plt = None

from metrics import rasterize_svg_to_edges, MetricsResult


def recolor_svg_raster(svg_rgba: Image.Image, target_color: Tuple[int, int, int]) -> Image.Image:
    """Replace dark stroke pixels in an RGBA raster with target_color while preserving alpha."""
    if Image is None:
        raise ImportError("Pillow required for recoloring SVG raster")
    arr = np.array(svg_rgba)
    if arr.shape[2] == 4:
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        mask = a > 0
        arr[..., 0][mask] = target_color[0]
        arr[..., 1][mask] = target_color[1]
        arr[..., 2][mask] = target_color[2]
        return Image.fromarray(arr)
    else:
        return svg_rgba.convert('RGBA')


def create_svg_overlay(base_image: np.ndarray, svg_string: str, opacity: float = 0.7, svg_color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    if Image is None or cairosvg is None:
        raise ImportError("Pillow and cairosvg required for overlay generation")
    h, w = base_image.shape[0], base_image.shape[1]
    base = Image.fromarray(base_image).convert('RGBA')
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=w, output_height=h)
    svg_img = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    svg_img = recolor_svg_raster(svg_img, svg_color)
    # Apply opacity
    alpha = svg_img.split()[-1].point(lambda v: int(v * opacity))
    svg_img.putalpha(alpha)
    base.paste(svg_img, (0, 0), svg_img)
    return np.array(base)


def create_comparison_overlay(base_image: np.ndarray, generated_svg: str, reference_svg: str, generated_color: Tuple[int, int, int] = (0, 255, 0), reference_color: Tuple[int, int, int] = (255, 0, 0), opacity: float = 0.6) -> np.ndarray:
    if Image is None or cairosvg is None:
        raise ImportError("Pillow and cairosvg required for comparison overlay")
    h, w = base_image.shape[0], base_image.shape[1]
    base = Image.fromarray(base_image).convert('RGBA')
    png_gen = cairosvg.svg2png(bytestring=generated_svg.encode('utf-8'), output_width=w, output_height=h)
    png_ref = cairosvg.svg2png(bytestring=reference_svg.encode('utf-8'), output_width=w, output_height=h)
    gen_img = Image.open(io.BytesIO(png_gen)).convert('RGBA')
    ref_img = Image.open(io.BytesIO(png_ref)).convert('RGBA')
    gen_img = recolor_svg_raster(gen_img, generated_color)
    ref_img = recolor_svg_raster(ref_img, reference_color)
    gen_img.putalpha(gen_img.split()[-1].point(lambda v: int(v * opacity)))
    ref_img.putalpha(ref_img.split()[-1].point(lambda v: int(v * opacity)))
    base.paste(ref_img, (0, 0), ref_img)
    base.paste(gen_img, (0, 0), gen_img)
    return np.array(base)


def create_edge_difference_map(predicted_edges: np.ndarray, ground_truth_edges: np.ndarray) -> np.ndarray:
    # Ensure boolean
    pred = predicted_edges.astype(bool)
    gt = ground_truth_edges.astype(bool)
    h, w = pred.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(np.logical_not(pred), gt)
    result[tp] = [255, 255, 255]  # white
    result[fp] = [255, 0, 0]      # red
    result[fn] = [0, 0, 255]      # blue
    return result


def create_side_by_side_comparison(original_image: np.ndarray, generated_svg: str, reference_svg: Optional[str] = None) -> np.ndarray:
    if Image is None or cairosvg is None:
        raise ImportError("Pillow and cairosvg required for side-by-side comparison")
    h, w = original_image.shape[0], original_image.shape[1]
    orig = Image.fromarray(original_image).convert('RGB')
    png_gen = cairosvg.svg2png(bytestring=generated_svg.encode('utf-8'), output_width=w, output_height=h)
    gen_img = Image.open(io.BytesIO(png_gen)).convert('RGB')
    if reference_svg:
        png_ref = cairosvg.svg2png(bytestring=reference_svg.encode('utf-8'), output_width=w, output_height=h)
        ref_img = Image.open(io.BytesIO(png_ref)).convert('RGB')
        combined = Image.new('RGB', (w * 3, h), (255, 255, 255))
        combined.paste(orig, (0, 0))
        combined.paste(gen_img, (w, 0))
        combined.paste(ref_img, (w * 2, 0))
    else:
        combined = Image.new('RGB', (w * 2, h), (255, 255, 255))
        combined.paste(orig, (0, 0))
        combined.paste(gen_img, (w, 0))
    return np.array(combined)


def create_metrics_visualization(metrics_result: MetricsResult):
    if plt is None:
        raise ImportError("matplotlib required for metrics visualization")
    # Simple bar chart for precision/recall/f1 and file sizes
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    names = []
    vals = []
    if metrics_result.edge_precision is not None:
        names.extend(['Precision', 'Recall', 'F1'])
        vals.extend([metrics_result.edge_precision or 0.0, metrics_result.edge_recall or 0.0, metrics_result.edge_f1 or 0.0])
        axes[0].bar(names, vals, color=['#2ca02c', '#ff7f0e', '#1f77b4'])
        axes[0].set_ylim(0, 1)
        axes[0].set_title('Edge Metrics')
    else:
        axes[0].text(0.5, 0.5, 'No edge metrics available', ha='center')

    # File sizes
    sizes = [metrics_result.svg_file_size_bytes or 0, metrics_result.svg_optimized_size_bytes or 0]
    axes[1].bar(['original', 'optimized'], sizes, color=['#7f7f7f', '#17becf'])
    axes[1].set_title('SVG Size (bytes)')
    plt.tight_layout()
    return fig
