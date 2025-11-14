import io
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np

# Optional heavy dependencies; functions will handle ImportError gracefully
try:
    from PIL import Image
    import cairosvg
    from skimage.morphology import dilation, disk, thin
    from skimage.feature import canny
    from scipy.spatial.distance import directed_hausdorff
    from scipy.spatial import distance as sp_distance
    import svgpathtools as spt
except Exception:
    # We'll raise later when functions requiring these libs are used
    Image = None
    cairosvg = None
    dilation = None
    disk = None
    thin = None
    canny = None
    directed_hausdorff = None
    sp_distance = None
    spt = None


@dataclass
class MetricsResult:
    edge_precision: Optional[float] = None
    edge_recall: Optional[float] = None
    edge_f1: Optional[float] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    pred_edge_count: Optional[int] = None
    gt_edge_count: Optional[int] = None
    path_count: Optional[int] = None
    svg_file_size_bytes: Optional[int] = None
    svg_optimized_size_bytes: Optional[int] = None
    compression_ratio_percent: Optional[float] = None
    hausdorff_distance: Optional[float] = None
    modified_hausdorff_distance: Optional[float] = None
    processing_time_seconds: Optional[float] = None


def parse_svg_paths(svg_string: str) -> List[str]:
    """Extract `d` attributes from <path> elements in an SVG string.
    Returns a list of path `d` strings."""
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(svg_string)
        # Handle namespaces
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        paths = []
        for p in root.findall('.//svg:path', ns):
            d = p.get('d')
            if d:
                paths.append(d)
        # Fallback: look for plain 'path' if namespace not present
        if not paths:
            for p in root.findall('.//path'):
                d = p.get('d')
                if d:
                    paths.append(d)
        return paths
    except Exception:
        return []


def sample_path_by_arc_length(path_d: str, num_samples: int = 200) -> np.ndarray:
    """Sample a single SVG path string by arc length into an (N,2) numpy array.
    Returns empty array on failure."""
    if spt is None:
        raise ImportError("svgpathtools is required for path sampling")
    try:
        path = spt.parse_path(path_d)
        L = path.length()
        if L == 0:
            return np.empty((0, 2))
        samples = []
        for i in range(num_samples):
            arcl = (i / max(1, num_samples - 1)) * L
            t = path.ilength(arcl)
            pt = path.point(t)
            samples.append((pt.real, pt.imag))
        return np.array(samples)
    except Exception:
        return np.empty((0, 2))


def compute_edge_metrics_tolerant(predicted_edges: np.ndarray, ground_truth_edges: np.ndarray, tolerance_radius: int = 2) -> Dict:
    """Compute precision, recall, and F1 for binary edge maps with tolerance.
    predicted_edges and ground_truth_edges must be boolean arrays of same shape.
    Returns a dict with precision/recall/f1 and counts."""
    if dilation is None or disk is None or thin is None:
        raise ImportError("scikit-image is required for edge metrics")

    try:
        # Ensure boolean
        pred = predicted_edges.astype(bool)
        gt = ground_truth_edges.astype(bool)

        # Thin to single-pixel width
        pred_thin = thin(pred)
        gt_thin = thin(gt)

        # Dilate GT for tolerant precision calculation
        selem = disk(tolerance_radius)
        gt_dil = dilation(gt_thin, selem)
        tp_for_precision = np.logical_and(pred_thin, gt_dil).sum()
        pred_count = pred_thin.sum()

        precision = None
        if pred_count > 0:
            precision = float(tp_for_precision) / float(pred_count)

        # Dilate Pred for tolerant recall calculation
        pred_dil = dilation(pred_thin, selem)
        tp_for_recall = np.logical_and(gt_thin, pred_dil).sum()
        gt_count = gt_thin.sum()
        recall = None
        if gt_count > 0:
            recall = float(tp_for_recall) / float(gt_count)

        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)

        # False positives/negatives estimation
        tp = int(tp_for_precision)
        fp = int(pred_count - tp_for_precision) if pred_count is not None else None
        fn = int(gt_count - tp_for_recall) if gt_count is not None else None

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp_for_precision': int(tp_for_precision),
            'tp_for_recall': int(tp_for_recall),
            'pred_count': int(pred_count),
            'gt_count': int(gt_count),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    except Exception as e:
        return {'precision': None, 'recall': None, 'f1': None, 'error': str(e)}


def compute_svg_metrics(svg_string: str, optimized_svg_string: str) -> Dict:
    try:
        import xml.etree.ElementTree as ET
        original_size = len(svg_string.encode('utf-8'))
        optimized_size = len(optimized_svg_string.encode('utf-8'))
        compression_ratio = None
        if original_size > 0:
            compression_ratio = (original_size - optimized_size) / original_size * 100.0

        path_count = 0
        try:
            root = ET.fromstring(svg_string)
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            path_elems = root.findall('.//svg:path', ns)
            if not path_elems:
                path_elems = root.findall('.//path')
            path_count = len(path_elems)
        except Exception:
            path_count = 0

        return {
            'path_count': int(path_count),
            'original_size_bytes': int(original_size),
            'optimized_size_bytes': int(optimized_size),
            'compression_ratio_percent': float(compression_ratio) if compression_ratio is not None else None
        }
    except Exception as e:
        return {'error': str(e)}


def compute_hausdorff_distance(generated_svg_string: str, reference_svg_string: str, samples_per_path: int = 200) -> Dict:
    if spt is None or directed_hausdorff is None or sp_distance is None:
        raise ImportError("svgpathtools and scipy are required for Hausdorff distance computation")

    try:
        gen_paths = parse_svg_paths(generated_svg_string)
        ref_paths = parse_svg_paths(reference_svg_string)

        gen_points = []
        ref_points = []

        for d in gen_paths:
            pts = sample_path_by_arc_length(d, samples_per_path)
            if pts.size:
                gen_points.append(pts)
        for d in ref_paths:
            pts = sample_path_by_arc_length(d, samples_per_path)
            if pts.size:
                ref_points.append(pts)

        if not gen_points or not ref_points:
            return {'hausdorff_distance': None, 'modified_hausdorff_distance': None, 'num_generated_points': 0, 'num_reference_points': 0}

        gen_all = np.vstack(gen_points)
        ref_all = np.vstack(ref_points)

        # directed Hausdorff
        d1 = directed_hausdorff(gen_all, ref_all)[0]
        d2 = directed_hausdorff(ref_all, gen_all)[0]
        hausdorff = max(d1, d2)

        # modified Hausdorff (mean of nearest neighbor distances both ways)
        D = sp_distance.cdist(gen_all, ref_all, 'euclidean')
        mhd1 = D.min(axis=1).mean()
        mhd2 = D.min(axis=0).mean()
        modified = max(mhd1, mhd2)

        return {
            'hausdorff_distance': float(hausdorff),
            'modified_hausdorff_distance': float(modified),
            'num_generated_points': int(len(gen_all)),
            'num_reference_points': int(len(ref_all))
        }
    except Exception as e:
        return {'error': str(e)}


def rasterize_svg_to_edges(svg_string: str, target_width: int, target_height: int, use_canny: bool = True) -> np.ndarray:
    """Rasterize SVG to a binary edge map (boolean numpy array).
    Uses CairoSVG to rasterize then applies edge detection (Canny) or simple threshold.
    """
    if cairosvg is None or Image is None:
        raise ImportError("Pillow and cairosvg are required for rasterizing SVGs")
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=target_width, output_height=target_height)
        img = Image.open(io.BytesIO(png_bytes)).convert('L')
        arr = np.array(img)

        if use_canny and canny is not None:
            # Normalize to [0,1]
            edges = canny(arr / 255.0)
            return edges.astype(bool)
        else:
            # Fallback: simple edge by thresholding gradient
            gx = np.abs(np.diff(arr.astype(float), axis=1))
            gy = np.abs(np.diff(arr.astype(float), axis=0))
            g = np.zeros_like(arr)
            g[:, :-1] += gx
            g[:-1, :] += gy
            mask = g > g.mean() * 1.5
            return mask.astype(bool)
    except Exception as e:
        raise


def calculate_all_metrics(generated_svg: str, optimized_svg: str, reference_svg: Optional[str] = None, original_image_shape: Optional[Tuple[int, int]] = None, tolerance_radius: int = 2) -> MetricsResult:
    start = time.time()
    mr = MetricsResult()
    try:
        # SVG metrics
        svg_metrics = compute_svg_metrics(generated_svg, optimized_svg)
        mr.path_count = svg_metrics.get('path_count')
        mr.svg_file_size_bytes = svg_metrics.get('original_size_bytes')
        mr.svg_optimized_size_bytes = svg_metrics.get('optimized_size_bytes')
        mr.compression_ratio_percent = svg_metrics.get('compression_ratio_percent')

        # Hausdorff and modified Hausdorff
        if reference_svg:
            try:
                hd = compute_hausdorff_distance(generated_svg, reference_svg)
                mr.hausdorff_distance = hd.get('hausdorff_distance')
                mr.modified_hausdorff_distance = hd.get('modified_hausdorff_distance')
            except Exception:
                mr.hausdorff_distance = None
                mr.modified_hausdorff_distance = None

        # Edge metrics: rasterize both to the same shape
        if reference_svg and original_image_shape:
            try:
                h, w = original_image_shape[0], original_image_shape[1]
                pred_edges = rasterize_svg_to_edges(optimized_svg, w, h)
                gt_edges = rasterize_svg_to_edges(reference_svg, w, h)
                em = compute_edge_metrics_tolerant(pred_edges, gt_edges, tolerance_radius=tolerance_radius)
                mr.edge_precision = em.get('precision')
                mr.edge_recall = em.get('recall')
                mr.edge_f1 = em.get('f1')
                mr.tp = em.get('tp')
                mr.fp = em.get('fp')
                mr.fn = em.get('fn')
                mr.pred_edge_count = em.get('pred_count')
                mr.gt_edge_count = em.get('gt_count')
            except Exception:
                # Leave edge metrics as None
                pass

    finally:
        mr.processing_time_seconds = time.time() - start

    return mr
