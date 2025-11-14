"""Benchmarking utilities for the Image-to-SVG-Line-Drawing project.

This module implements timing, memory profiling, disk-based vs in-memory
comparisons, report generation, and a small CLI for running benchmarks.
It is defensive about optional dependencies and performs lazy imports to
remain importable in minimal test environments.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import time
import threading
import tempfile
import os
import json
import csv
import platform
import argparse
import math


@dataclass
class BenchmarkResult:
    image_path: str
    image_size_pixels: int
    image_dimensions: Tuple[int, int]
    category: Optional[str]
    complexity: Optional[str]
    processing_time_seconds: Optional[float]
    peak_memory_mb: Optional[float]
    average_memory_mb: Optional[float]
    svg_original_size_bytes: Optional[int]
    svg_optimized_size_bytes: Optional[int]
    compression_ratio_percent: Optional[float]
    path_count: Optional[int]
    throughput_pixels_per_second: Optional[float]
    method: str
    success: bool
    error_message: Optional[str]


@dataclass
class BenchmarkSuite:
    suite_name: str
    total_images: int
    successful_runs: int
    failed_runs: int
    total_time_seconds: float
    average_time_seconds: float
    median_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    average_memory_mb: Optional[float]
    peak_memory_mb: Optional[float]
    total_compression_percent: Optional[float]
    results: List[BenchmarkResult]


@dataclass
class ComparisonResult:
    image_path: str
    in_memory_time: Optional[float]
    disk_based_time: Optional[float]
    speedup_factor: Optional[float]
    in_memory_memory_mb: Optional[float]
    disk_based_memory_mb: Optional[float]
    memory_difference_mb: Optional[float]


def _lazy_import_streamlet_functions():
    try:
        from streamlit_app import process_image, generate_svg_from_paths, optimize_svg_string
        return process_image, generate_svg_from_paths, optimize_svg_string
    except Exception as e:
        raise ImportError(f"Could not import processing pipeline from streamlit_app: {e}")


class MemorySampler:
    """Simple memory sampler using psutil to record during a workload."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._stop = threading.Event()
        self._samples: List[int] = []
        self._thread = None
        self._psutil = None

    def __enter__(self):
        try:
            import psutil
            self._psutil = psutil
        except Exception:
            self._psutil = None
            return (None, None)

        self._proc = self._psutil.Process()

        def _run():
            while not self._stop.is_set():
                try:
                    self._samples.append(self._proc.memory_info().rss)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._psutil is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_stats(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._samples:
            return None, None
        peak = max(self._samples) / (1024.0 * 1024.0)
        avg = sum(self._samples) / len(self._samples) / (1024.0 * 1024.0)
        return peak, avg


def measure_processing_time(image_array, canny_sigma: float, min_contour_frac: float) -> Tuple[float, Optional[str], Optional[str], Any]:
    """Measure processing time by invoking the core pipeline functions.

    Returns (processing_time_seconds, svg_string, optimized_svg_string, path_data)
    """
    process_image, generate_svg_from_paths, optimize_svg_string = _lazy_import_streamlet_functions()
    t0 = time.perf_counter()
    path_data = process_image(image_array, canny_sigma, min_contour_frac)
    if not path_data:
        dt = time.perf_counter() - t0
        return dt, None, None, None
    svg = generate_svg_from_paths(path_data, image_array.shape[1], image_array.shape[0])
    opt_svg = optimize_svg_string(svg)
    dt = time.perf_counter() - t0
    return dt, svg, opt_svg, path_data


def measure_memory_usage(func, *args, **kwargs) -> Tuple[Optional[float], Optional[float], Any]:
    """Run func under a MemorySampler and return (peak_mb, avg_mb, result).

    If psutil is unavailable, returns (None, None, result).
    """
    with MemorySampler() as sampler:
        result = func(*args, **kwargs)
    if sampler is None or sampler._psutil is None:
        return None, None, result
    peak, avg = sampler.get_stats()
    return peak, avg, result


def measure_svg_compression(svg_string: Optional[str], optimized_svg_string: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    if svg_string is None or optimized_svg_string is None:
        return None, None, None
    orig = len(svg_string.encode('utf-8'))
    opt = len(optimized_svg_string.encode('utf-8'))
    try:
        ratio = (orig - opt) / float(orig) * 100.0
    except Exception:
        ratio = None
    return orig, opt, ratio


def calculate_throughput(image_dimensions: Tuple[int, int], processing_time: Optional[float]) -> Optional[float]:
    if processing_time is None or processing_time <= 0:
        return None
    w, h = image_dimensions
    return (w * h) / processing_time


def process_image_disk_based(image_array, canny_sigma: float, min_contour_frac: float, temp_dir: str) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Run a disk-based variant: write intermediate images and then run vectorization.

    Returns (path_data, svg_string, optimized_svg_string)
    """
    try:
        import cv2
    except Exception:
        raise ImportError('cv2 is required for disk-based processing')

    # 1. prepare gray & blurred
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v = cv2.medianBlur(blurred, 1)
        lower = int(max(0, (1.0 - canny_sigma) * float(v.mean())))
        upper = int(min(255, (1.0 + canny_sigma) * float(v.mean())))
        edges = cv2.Canny(blurred, lower, upper)
    except Exception as e:
        raise RuntimeError(f"Disk-based preprocessing failed: {e}")

    # write edges and mask to disk
    edges_path = os.path.join(temp_dir, 'edges.png')
    mask_path = os.path.join(temp_dir, 'mask.png')
    try:
        cv2.imwrite(edges_path, edges)
    except Exception:
        pass

    # create mask by finding contours and drawing significant ones
    try:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_area = gray.shape[0] * gray.shape[1]
        min_area = image_area * min_contour_frac
        significant = [c for c in contours if cv2.contourArea(c) > min_area]
        clean_mask = (edges * 0).astype('uint8')
        if significant:
            cv2.drawContours(clean_mask, significant, -1, (255), thickness=cv2.FILLED)
        cv2.imwrite(mask_path, clean_mask)
    except Exception:
        pass

    # read mask back and vectorize
    try:
        # lazy import potrace
        try:
            import potrace
        except Exception:
            raise ImportError('potrace is required for disk-based vectorization')

        import numpy as _np
        from io import BytesIO

        mask = _np.asarray(_np.fromfile(mask_path, dtype=_np.uint8)) if os.path.exists(mask_path) else clean_mask
        # ensure 2D
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        bmp = potrace.Bitmap((255 - mask).astype(_np.bool_))
        path_data = bmp.trace()
        # try to reuse streamlit_app svg generator if available
        try:
            from streamlit_app import generate_svg_from_paths, optimize_svg_string
        except Exception:
            generate_svg_from_paths = None
            optimize_svg_string = None

        if generate_svg_from_paths is not None:
            svg = generate_svg_from_paths(path_data, image_array.shape[1], image_array.shape[0])
            opt = optimize_svg_string(svg) if optimize_svg_string is not None else None
            return path_data, svg, opt
        else:
            return path_data, None, None
    except Exception as e:
        raise RuntimeError(f"Disk-based vectorization failed: {e}")


def benchmark_in_memory_vs_disk(image_array, canny_sigma: float, min_contour_frac: float, temp_dir: Optional[str] = None, iterations: int = 3) -> ComparisonResult:
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='benchmark_')
    in_times = []
    disk_times = []
    in_mem = []
    disk_mem = []

    # in-memory runs
    for _ in range(iterations):
        peak, avg, res = measure_memory_usage(measure_processing_time, image_array, canny_sigma, min_contour_frac)
        try:
            dt, svg, opt_svg, path_data = measure_processing_time(image_array, canny_sigma, min_contour_frac)
        except Exception:
            dt = None
        in_times.append(dt or float('nan'))
        in_mem.append(peak)

    # disk-based runs
    for _ in range(iterations):
        try:
            start = time.perf_counter()
            path_data, svg2, opt2 = process_image_disk_based(image_array, canny_sigma, min_contour_frac, temp_dir)
            dt = time.perf_counter() - start
        except Exception:
            dt = None
        disk_times.append(dt or float('nan'))
        peak, avg, _ = measure_memory_usage(process_image_disk_based, image_array, canny_sigma, min_contour_frac, temp_dir)
        disk_mem.append(peak)

    # compute medians
    def _median(xs):
        xs2 = [x for x in xs if x is not None and (not (isinstance(x, float) and math.isnan(x)))]
        if not xs2:
            return None
        xs2.sort()
        mid = len(xs2) // 2
        if len(xs2) % 2 == 1:
            return xs2[mid]
        return 0.5 * (xs2[mid - 1] + xs2[mid])

    in_med = _median(in_times)
    disk_med = _median(disk_times)
    in_peak = _median(in_mem)
    disk_peak = _median(disk_mem)

    speedup = None
    if in_med and disk_med and disk_med > 0:
        speedup = disk_med / in_med

    mem_diff = None
    if in_peak is not None and disk_peak is not None:
        mem_diff = disk_peak - in_peak

    return ComparisonResult(image_path=str(getattr(getattr(image_array, 'filename', None), '__str__', lambda: 'in-memory')()), in_memory_time=in_med, disk_based_time=disk_med, speedup_factor=speedup, in_memory_memory_mb=in_peak, disk_based_memory_mb=disk_peak, memory_difference_mb=mem_diff)


def benchmark_single_image(image_path: str, metadata: Optional[Dict[str, Any]] = None, method: str = 'in-memory', temp_dir: Optional[str] = None, iterations: int = 1) -> BenchmarkResult:
    # Load image
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        return BenchmarkResult(image_path=image_path, image_size_pixels=0, image_dimensions=(0,0), category=None, complexity=None, processing_time_seconds=None, peak_memory_mb=None, average_memory_mb=None, svg_original_size_bytes=None, svg_optimized_size_bytes=None, compression_ratio_percent=None, path_count=None, throughput_pixels_per_second=None, method=method, success=False, error_message=str(e))

    try:
        im = Image.open(image_path).convert('RGB')
        arr = np.array(im)
    except Exception as e:
        return BenchmarkResult(image_path=image_path, image_size_pixels=0, image_dimensions=(0,0), category=None, complexity=None, processing_time_seconds=None, peak_memory_mb=None, average_memory_mb=None, svg_original_size_bytes=None, svg_optimized_size_bytes=None, compression_ratio_percent=None, path_count=None, throughput_pixels_per_second=None, method=method, success=False, error_message=str(e))

    h, w = arr.shape[:2]
    size_pixels = w * h
    cat = None
    comp = None
    if metadata:
        cat = metadata.get('category')
        comp = metadata.get('geometry_complexity') or metadata.get('complexity')

    # perform measurement
    peak_mem = None
    avg_mem = None
    proc_time = None
    svg = None
    opt_svg = None
    path_count = None
    try:
        if method == 'in-memory':
            peak, avg, _ = measure_memory_usage(measure_processing_time, arr, metadata.get('canny_sigma', 0.33) if metadata else 0.33, metadata.get('min_contour_frac', 0.0001) if metadata else 0.0001)
            proc_time, svg, opt_svg, path_data = measure_processing_time(arr, metadata.get('canny_sigma', 0.33) if metadata else 0.33, metadata.get('min_contour_frac', 0.0001) if metadata else 0.0001)
            peak_mem = peak
            avg_mem = avg
            if path_data is not None:
                try:
                    path_count = len(list(path_data))
                except Exception:
                    path_count = None
        else:
            # disk-based
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix='benchmark_')
            start = time.perf_counter()
            path_data, svg, opt_svg = process_image_disk_based(arr, metadata.get('canny_sigma', 0.33) if metadata else 0.33, metadata.get('min_contour_frac', 0.0001) if metadata else 0.0001, temp_dir)
            proc_time = time.perf_counter() - start
            peak, avg, _ = measure_memory_usage(process_image_disk_based, arr, metadata.get('canny_sigma', 0.33) if metadata else 0.33, metadata.get('min_contour_frac', 0.0001) if metadata else 0.0001, temp_dir)
            peak_mem = peak
            avg_mem = avg
            if path_data is not None:
                try:
                    path_count = len(list(path_data))
                except Exception:
                    path_count = None

        orig_sz, opt_sz, ratio = measure_svg_compression(svg, opt_svg)
        throughput = calculate_throughput((w, h), proc_time)

        return BenchmarkResult(image_path=image_path, image_size_pixels=size_pixels, image_dimensions=(w, h), category=cat, complexity=comp, processing_time_seconds=proc_time, peak_memory_mb=peak_mem, average_memory_mb=avg_mem, svg_original_size_bytes=orig_sz, svg_optimized_size_bytes=opt_sz, compression_ratio_percent=ratio, path_count=path_count, throughput_pixels_per_second=throughput, method=method, success=True, error_message=None)

    except Exception as e:
        return BenchmarkResult(image_path=image_path, image_size_pixels=size_pixels, image_dimensions=(w, h), category=cat, complexity=comp, processing_time_seconds=None, peak_memory_mb=None, average_memory_mb=None, svg_original_size_bytes=None, svg_optimized_size_bytes=None, compression_ratio_percent=None, path_count=None, throughput_pixels_per_second=None, method=method, success=False, error_message=str(e))


def benchmark_test_suite(metadata_path: str = 'test_images/test_metadata.json', method: str = 'in-memory', filter_category: Optional[str] = None, filter_complexity: Optional[str] = None, temp_dir: Optional[str] = None) -> BenchmarkSuite:
    # lazy load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    keys = list(meta.keys())
    results: List[BenchmarkResult] = []
    start = time.perf_counter()
    for k in keys:
        md = meta.get(k, {})
        if filter_category and md.get('category') != filter_category:
            continue
        if filter_complexity and md.get('geometry_complexity') != filter_complexity:
            continue
        img_path = os.path.join('test_images', k)
        res = benchmark_single_image(img_path, metadata=md, method=method, temp_dir=temp_dir)
        results.append(res)

    total_time = time.perf_counter() - start
    times = [r.processing_time_seconds for r in results if r.processing_time_seconds is not None]
    peak_mem = [r.peak_memory_mb for r in results if r.peak_memory_mb is not None]
    avg_mem = [r.average_memory_mb for r in results if r.average_memory_mb is not None]
    compress = [r.compression_ratio_percent for r in results if r.compression_ratio_percent is not None]

    successful = len([r for r in results if r.success])
    failed = len(results) - successful
    avg_time = float(sum(times) / len(times)) if times else 0.0
    median_time = sorted(times)[len(times)//2] if times else 0.0
    min_time = min(times) if times else 0.0
    max_time = max(times) if times else 0.0

    return BenchmarkSuite(suite_name=os.path.basename(metadata_path), total_images=len(results), successful_runs=successful, failed_runs=failed, total_time_seconds=total_time, average_time_seconds=avg_time, median_time_seconds=median_time, min_time_seconds=min_time, max_time_seconds=max_time, average_memory_mb=(sum(avg_mem)/len(avg_mem) if avg_mem else None), peak_memory_mb=(max(peak_mem) if peak_mem else None), total_compression_percent=(sum(compress)/len(compress) if compress else None), results=results)


def generate_benchmark_report(benchmark_suite: BenchmarkSuite, output_dir: str = 'benchmark_reports', report_name: str = 'benchmark') -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time())
    base = f"{report_name}_{ts}"
    json_path = os.path.join(output_dir, base + '.json')
    csv_path = os.path.join(output_dir, base + '.csv')
    md_path = os.path.join(output_dir, base + '.md')

    # JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        payload = {
            'suite': asdict(benchmark_suite),
            'system_info': get_system_info(),
            'timestamp': ts
        }
        json.dump(payload, f, indent=2)

    # CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'category', 'complexity', 'width', 'height', 'processing_time_seconds', 'peak_memory_mb', 'average_memory_mb', 'compression_ratio_percent', 'throughput_pixels_per_second', 'success', 'error_message'])
        for r in benchmark_suite.results:
            w, h = r.image_dimensions if r.image_dimensions else (None, None)
            writer.writerow([r.image_path, r.category, r.complexity, w, h, r.processing_time_seconds, r.peak_memory_mb, r.average_memory_mb, r.compression_ratio_percent, r.throughput_pixels_per_second, r.success, r.error_message])

    # Markdown summary
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report: {benchmark_suite.suite_name}\n\n")
        f.write(f"Total images: {benchmark_suite.total_images}\n\n")
        f.write(f"Successful runs: {benchmark_suite.successful_runs}\n\n")
        f.write(f"Failed runs: {benchmark_suite.failed_runs}\n\n")
        f.write(f"Total time (s): {benchmark_suite.total_time_seconds:.3f}\n\n")
        f.write(f"Average time (s): {benchmark_suite.average_time_seconds:.3f}\n\n")

    return {'json': json_path, 'csv': csv_path, 'md': md_path}


def create_performance_visualizations(benchmark_suite: BenchmarkSuite, output_dir: str = 'benchmark_reports') -> List[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    times = [r.processing_time_seconds for r in benchmark_suite.results if r.processing_time_seconds is not None]
    sizes = [r.image_size_pixels for r in benchmark_suite.results if r.processing_time_seconds is not None]
    # Time distribution
    if times:
        fig = plt.figure()
        plt.hist(times, bins=30)
        plt.xlabel('Processing time (s)')
        plt.ylabel('Count')
        p = os.path.join(output_dir, 'time_distribution.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # Time vs Size
    if times and sizes:
        fig = plt.figure()
        plt.scatter(sizes, times)
        # trend line
        try:
            z = np.polyfit(sizes, times, 1)
            pfit = np.poly1d(z)
            xs = np.linspace(min(sizes), max(sizes), 100)
            plt.plot(xs, pfit(xs), color='red')
        except Exception:
            pass
        plt.xlabel('Image pixels')
        plt.ylabel('Processing time (s)')
        p = os.path.join(output_dir, 'time_vs_size.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def get_system_info() -> Dict[str, Any]:
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['total_ram_bytes'] = psutil.virtual_memory().total
    except Exception:
        info['cpu_count'] = None
        info['total_ram_bytes'] = None
    try:
        import numpy as _np
        info['numpy_version'] = _np.__version__
    except Exception:
        info['numpy_version'] = None
    try:
        import cv2
        info['opencv_version'] = cv2.__version__
    except Exception:
        info['opencv_version'] = None
    return info


def create_synthetic_test_image(width: int, height: int, complexity: str = 'moderate'):
    try:
        import numpy as np
        import cv2
    except Exception:
        raise ImportError('numpy and cv2 are required for synthetic image generation')
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    if complexity == 'simple':
        cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
    elif complexity == 'moderate':
        cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
        cv2.circle(img, (width // 2, height // 2), min(width, height)//6, (0, 0, 0), -1)
    else:
        # complex: many random shapes
        import random
        for _ in range(200):
            x1 = random.randint(0, width-1)
            y1 = random.randint(0, height-1)
            x2 = random.randint(0, width-1)
            y2 = random.randint(0, height-1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    return img


def safe_load_image(image_path: str):
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(image_path).convert('RGB')
        arr = np.array(im)
        # validate if possible
        try:
            from utils import validate_image_array
            validate_image_array(arr)
        except Exception:
            pass
        return arr
    except Exception:
        return None


def format_bytes(b: Optional[int]) -> str:
    if b is None:
        return 'N/A'
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(b) < 1024.0:
            return f"{b:3.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} TB"


def format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return 'N/A'
    if seconds < 1.0:
        return f"{seconds*1000.0:.1f} ms"
    return f"{seconds:.3f} s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', action='store_true')
    p.add_argument('--single', help='Single image path')
    p.add_argument('--category', help='Filter by category')
    p.add_argument('--complexity', help='Filter by complexity')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--size-scaling', action='store_true')
    p.add_argument('--method', choices=['in-memory', 'disk-based'], default='in-memory')
    p.add_argument('--output-dir', default='benchmark_reports')
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--baseline', help='Path to baseline JSON report')
    p.add_argument('--validate-claims', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.single:
        res = benchmark_single_image(args.single, method=args.method)
        suite = BenchmarkSuite(suite_name='single', total_images=1, successful_runs=1 if res.success else 0, failed_runs=0 if res.success else 1, total_time_seconds=res.processing_time_seconds or 0.0, average_time_seconds=res.processing_time_seconds or 0.0, median_time_seconds=res.processing_time_seconds or 0.0, min_time_seconds=res.processing_time_seconds or 0.0, max_time_seconds=res.processing_time_seconds or 0.0, average_memory_mb=res.average_memory_mb, peak_memory_mb=res.peak_memory_mb, total_compression_percent=res.compression_ratio_percent, results=[res])
        generate_benchmark_report(suite, output_dir=args.output_dir)
        create_performance_visualizations(suite, output_dir=args.output_dir)
        return

    if args.size_scaling:
        base = create_synthetic_test_image(800, 600)
        sizes = [(320,240),(640,480),(1280,960),(1920,1440),(2560,1920),(3840,2880)]
        results = []
        for w,h in sizes:
            img = create_synthetic_test_image(w, h, complexity='moderate')
            r = benchmark_single_image('__synthetic__', metadata=None, method=args.method)
            results.append(r)
        suite = BenchmarkSuite(suite_name='size_scaling', total_images=len(results), successful_runs=len([r for r in results if r.success]), failed_runs=len([r for r in results if not r.success]), total_time_seconds=sum((r.processing_time_seconds or 0.0) for r in results), average_time_seconds=(sum((r.processing_time_seconds or 0.0) for r in results)/len(results) if results else 0.0), median_time_seconds=0.0, min_time_seconds=0.0, max_time_seconds=0.0, average_memory_mb=None, peak_memory_mb=None, total_compression_percent=None, results=results)
        generate_benchmark_report(suite, output_dir=args.output_dir)
        create_performance_visualizations(suite, output_dir=args.output_dir)
        return

    if args.suite:
        suite = benchmark_test_suite(method=args.method, filter_category=args.category, filter_complexity=args.complexity)
        generate_benchmark_report(suite, output_dir=args.output_dir)
        create_performance_visualizations(suite, output_dir=args.output_dir)
        return

    if args.compare:
        # runs comparison for all images in test metadata
        if not os.path.exists('test_images/test_metadata.json'):
            print('metadata not found')
            return
        with open('test_images/test_metadata.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
        comparisons = []
        for k, md in meta.items():
            img_path = os.path.join('test_images', k)
            arr = safe_load_image(img_path)
            if arr is None:
                continue
            comp = benchmark_in_memory_vs_disk(arr, md.get('canny_sigma', 0.33), md.get('min_contour_frac', 0.0001), iterations=args.iterations)
            comparisons.append(asdict(comp))
        out = os.path.join(args.output_dir, f'comparison_{int(time.time())}.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': comparisons, 'system_info': get_system_info()}, f, indent=2)
        print('Wrote', out)
        return

    p.print_help()


if __name__ == '__main__':
    main()
