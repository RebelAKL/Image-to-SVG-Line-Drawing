import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Callable, Any

# Heavy numeric/plotting libraries are imported lazily inside functions to avoid import-time failures

# Optional imports from project modules; avoid top-level circular imports by importing inside functions
try:
    from metrics import MetricsResult, calculate_all_metrics
except Exception:
    MetricsResult = None
    calculate_all_metrics = None


@dataclass
class GridSearchConfig:
    sigma_range: Tuple[float, float, int] = (0.0, 1.0, 5)  # min, max, steps
    contour_frac_range: Tuple[float, float, int] = (1e-6, 1e-3, 5)  # min, max, steps
    search_type: str = 'quick'  # 'quick'|'standard'|'fine'
    center_point: Optional[Tuple[float, float]] = None
    timeout_per_combination: Optional[float] = None


@dataclass
class ParameterCombination:
    canny_sigma: float
    min_contour_frac: float
    score: Optional[float] = None
    metrics: Optional[Any] = None
    processing_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class OptimizationResult:
    best_params: Optional[ParameterCombination]
    all_results: List[ParameterCombination]
    total_combinations: int
    successful_combinations: int
    total_time: float
    image_info: dict


def generate_parameter_grid(config: GridSearchConfig) -> List[Tuple[float, float]]:
    stype = config.search_type
    # import numpy locally to avoid top-level dependency
    try:
        import numpy as np
    except Exception:
        np = None

    if stype == 'quick':
        sigmas = [0.15, 0.25, 0.33, 0.45, 0.60]
        fracs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    elif stype == 'standard':
        if np is None:
            # fallback to quick if numpy not available
            sigmas = [0.15, 0.25, 0.33, 0.45, 0.60]
            fracs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        else:
            sigmas = list(np.linspace(config.sigma_range[0], config.sigma_range[1], 10))
            fracs = list(np.linspace(config.contour_frac_range[0], config.contour_frac_range[1], 10))
    elif stype == 'fine' and config.center_point is not None:
        c_sigma, c_frac = config.center_point
        if np is None:
            sigmas = [c_sigma]
            fracs = [c_frac]
        else:
            sigmas = list(np.linspace(c_sigma * 0.8, c_sigma * 1.2, 7))
            fracs = list(np.linspace(max(1e-8, c_frac * 0.8), c_frac * 1.2, 7))
    else:
        # default to quick
        sigmas = [0.15, 0.25, 0.33, 0.45, 0.60]
        fracs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    grid = []
    for s in sigmas:
        for f in fracs:
            grid.append((float(s), float(f)))
    return grid


def compute_composite_score(metrics_result: Optional[Any], has_reference: bool = True) -> float:
    """Return a score scaled 0-100. Handle missing metrics gracefully."""
    if metrics_result is None:
        return 0.0

    # metrics_result may be a dataclass (MetricsResult) or dict
    def _get(m, k):
        try:
            return m.get(k) if isinstance(m, dict) else getattr(m, k, None)
        except Exception:
            return None

    if has_reference:
        f1 = _get(metrics_result, 'edge_f1') or 0.0
        hd = _get(metrics_result, 'hausdorff_distance')
        pc = _get(metrics_result, 'path_count') or 1.0

        # normalize components
        f1_norm = float(f1)  # already 0..1

        # Inverse Hausdorff: smaller is better. We normalize by a heuristic cap.
        if hd is None:
            hd_score = 0.0
        else:
            hd_cap = max(1.0, float(hd))
            hd_score = 1.0 / (1.0 + (hd_cap - 1.0))

        # Path count: prefer fewer paths (normalized by a heuristic)
        pc_norm = 1.0 / (1.0 + (float(pc) / 50.0))

        # weights: F1 50%, invHD 30%, invPathCount 20%
        composite = 0.5 * f1_norm + 0.3 * hd_score + 0.2 * pc_norm
        return float(composite * 100.0)
    else:
        # heuristics when no reference is available
        # Use available edge proxy (edge_f1 or edge_precision) as a measure of edge consistency
        edge_proxy = _get(metrics_result, 'edge_f1')
        if edge_proxy is None:
            edge_proxy = _get(metrics_result, 'edge_precision') or 0.0

        pc = _get(metrics_result, 'path_count') or 1.0
        # Prefer optimized svg size when present
        fs = _get(metrics_result, 'svg_optimized_size_bytes') or _get(metrics_result, 'svg_file_size_bytes') or 1.0

        try:
            ec_norm = float(edge_proxy)
        except Exception:
            ec_norm = 0.0

        pc_norm = 1.0 / (1.0 + (float(pc) / 100.0))
        fs_norm = 1.0 / (1.0 + (float(fs) / 100000.0))

        # weights: edge proxy 50%, path count 30%, file size 20%
        composite = 0.5 * ec_norm + 0.3 * pc_norm + 0.2 * fs_norm
        return float(composite * 100.0)


def create_optimization_heatmap(opt_result: OptimizationResult):
    # Build grid arrays
    all_rs = opt_result.all_results
    if not all_rs:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None
        if plt is None:
            return None
        fig = plt.figure()
        plt.text(0.5, 0.5, 'No results', ha='center')
        return fig

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        np = None
        plt = None

    sigmas = sorted(list(set([round(r.canny_sigma, 6) for r in all_rs])))
    fracs = sorted(list(set([round(r.min_contour_frac, 12) for r in all_rs])))

    sigma_idx = {s: i for i, s in enumerate(sigmas)}
    frac_idx = {f: i for i, f in enumerate(fracs)}

    if np is None:
        return None
    grid = np.full((len(fracs), len(sigmas)), np.nan)
    for r in all_rs:
        s = round(r.canny_sigma, 6)
        f = round(r.min_contour_frac, 12)
        i = frac_idx.get(f)
        j = sigma_idx.get(s)
        if i is not None and j is not None and r.score is not None:
            grid[i, j] = r.score

    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{s:.3f}" for s in sigmas], rotation=45)
    ax.set_yticks(np.arange(len(fracs)))
    ax.set_yticklabels([f"{f:.6g}" for f in fracs])
    ax.set_xlabel('canny_sigma')
    ax.set_ylabel('min_contour_frac')
    ax.set_title('Optimization Score Heatmap')
    cbar = fig.colorbar(im, ax=ax)

    # mark best
    if opt_result.best_params:
        try:
            best_s = round(opt_result.best_params.canny_sigma, 6)
            best_f = round(opt_result.best_params.min_contour_frac, 12)
            bx = sigma_idx.get(best_s)
            by = frac_idx.get(best_f)
            if bx is not None and by is not None:
                ax.plot(bx, by, marker='*', color='red', markersize=12)
        except Exception:
            pass

    fig.tight_layout()
    return fig


def run_grid_search(image_array, reference_svg: Optional[str] = None, config: Optional[GridSearchConfig] = None, progress_callback: Optional[Callable] = None) -> OptimizationResult:
    # Lazy import processing functions to avoid circular imports
    try:
        from streamlit_app import process_image, generate_svg_from_paths, optimize_svg_string
    except Exception as e:
        raise ImportError(f"Could not import processing pipeline from streamlit_app: {e}")

    if config is None:
        config = GridSearchConfig()

    grid = generate_parameter_grid(config)
    results: List[ParameterCombination] = []
    start = time.time()
    success_count = 0

    total = len(grid)
    for idx, (s, f) in enumerate(grid, start=1):
        pc = ParameterCombination(canny_sigma=s, min_contour_frac=f)
        t0 = time.time()
        try:
            path_data = process_image(image_array, s, f)
            if not path_data:
                pc.success = False
                pc.error_message = 'no_significant_edges'
                pc.processing_time = time.time() - t0
                results.append(pc)
                if progress_callback:
                    progress_callback(idx / total, pc)
                continue

            svg = generate_svg_from_paths(path_data, image_array.shape[1], image_array.shape[0])
            opt_svg = optimize_svg_string(svg)

            # compute metrics if available
            metrics_res = None
            if calculate_all_metrics is not None:
                try:
                    metrics_res = calculate_all_metrics(svg, opt_svg, reference_svg, original_image_shape=image_array.shape)
                except Exception as e:
                    metrics_res = None

            pc.metrics = metrics_res
            pc.processing_time = time.time() - t0
            pc.success = True

            # scoring
            pc.score = compute_composite_score(metrics_res, has_reference=(reference_svg is not None))
            success_count += 1
        except Exception as e:
            pc.success = False
            pc.error_message = str(e)
            pc.processing_time = time.time() - t0
        results.append(pc)
        if progress_callback:
            try:
                progress_callback(idx / total, pc)
            except Exception:
                pass

    total_time = time.time() - start
    # find best (highest score)
    best = None
    scored = [r for r in results if r.score is not None]
    if scored:
        best = max(scored, key=lambda x: x.score)

    img_info = {'height': int(image_array.shape[0]), 'width': int(image_array.shape[1])}
    opt_res = OptimizationResult(best_params=best, all_results=results, total_combinations=total, successful_combinations=success_count, total_time=total_time, image_info=img_info)

    return opt_res


def generate_optimization_report(opt_res: OptimizationResult, output_dir: str = 'optimization_reports', image_name: str = 'optimization') -> dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time())
    base = f"{image_name}_{ts}"
    json_path = os.path.join(output_dir, base + '.json')
    csv_path = os.path.join(output_dir, base + '.csv')
    md_path = os.path.join(output_dir, base + '.md')
    png_path = os.path.join(output_dir, base + '_heatmap.png')

    # JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'optimization': asdict(opt_res)}, f, indent=2, default=str)

    # CSV
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['canny_sigma', 'min_contour_frac', 'score', 'success', 'processing_time', 'error_message'])
        for r in opt_res.all_results:
            writer.writerow([r.canny_sigma, r.min_contour_frac, r.score, r.success, r.processing_time, r.error_message])

    # Heatmap
    try:
        fig = create_optimization_heatmap(opt_res)
        if fig is not None:
            fig.savefig(png_path, dpi=150)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        png_path = None

    # Markdown summary
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Optimization Report for {image_name}\n\n")
        f.write(f"Total combinations: {opt_res.total_combinations}\n\n")
        f.write(f"Successful: {opt_res.successful_combinations}\n\n")
        if opt_res.best_params:
            bp = opt_res.best_params
            f.write(f"**Best parameters**: sigma={bp.canny_sigma:.4f}, contour_frac={bp.min_contour_frac:.6g}, score={bp.score:.2f}\n\n")
        if png_path:
            f.write(f"![heatmap]({os.path.basename(png_path)})\n\n")

    return {'json': json_path, 'csv': csv_path, 'md': md_path, 'png': png_path}


def optimize_test_dataset(metadata_path: str = 'test_images/test_metadata.json', config: Optional[GridSearchConfig] = None, output_dir: str = 'optimization_reports', update_metadata: bool = False):
    # Batch optimize all test cases listed in metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError('metadata not found')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    all_results = {}
    for image_rel, md in meta.items():
        try:
            img_path = os.path.join('test_images', image_rel)
            from PIL import Image
            im = Image.open(img_path).convert('RGB')
            try:
                import numpy as np
                arr = np.array(im)
            except Exception:
                # fallback: use list conversion
                arr = Image.open(img_path).convert('RGB')
        except Exception as e:
            all_results[image_rel] = {'error': str(e)}
            continue

        ref = None
        expected_rel = md.get('expected_output')
        if expected_rel and os.path.exists(expected_rel):
            with open(expected_rel, 'r', encoding='utf-8') as f:
                ref = f.read()

        opt_res = run_grid_search(arr, reference_svg=ref, config=config)
        rep = generate_optimization_report(opt_res, output_dir=output_dir, image_name=os.path.splitext(os.path.basename(image_rel))[0])
        all_results[image_rel] = {'report_files': rep, 'best': asdict(opt_res.best_params) if opt_res.best_params else None}

        if update_metadata and opt_res.best_params:
            md.setdefault('optimal_parameters', {})
            md['optimal_parameters']['canny_sigma'] = opt_res.best_params.canny_sigma
            md['optimal_parameters']['min_contour_frac'] = opt_res.best_params.min_contour_frac

    if update_metadata:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    # Aggregate best parameters by category and write a summary report
    try:
        from statistics import mean, median
        category_map = {}
        for image_rel, md in meta.items():
            cat = md.get('category', 'unknown') if isinstance(md, dict) else 'unknown'
            entry = all_results.get(image_rel, {})
            best = entry.get('best') if isinstance(entry, dict) else None
            if best:
                category_map.setdefault(cat, []).append(best)

        category_summary = {}
        for cat, bests in category_map.items():
            sigmas = [b.get('canny_sigma') for b in bests if b and b.get('canny_sigma') is not None]
            fracs = [b.get('min_contour_frac') for b in bests if b and b.get('min_contour_frac') is not None]
            category_summary[cat] = {
                'count': len(bests),
                'sigma_mean': mean(sigmas) if sigmas else None,
                'sigma_median': median(sigmas) if sigmas else None,
                'contour_frac_mean': mean(fracs) if fracs else None,
                'contour_frac_median': median(fracs) if fracs else None,
            }

        # Write summary files
        os.makedirs(output_dir, exist_ok=True)
        ts = int(time.time())
        summary_json = os.path.join(output_dir, f'category_summary_{ts}.json')
        summary_md = os.path.join(output_dir, f'category_summary_{ts}.md')
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump({'category_summary': category_summary}, f, indent=2)

        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write('# Category Optimization Summary\n\n')
            for cat, stats in category_summary.items():
                f.write(f'## {cat}\n')
                f.write(f"Count: {stats['count']}\n\n")
                f.write(f"Sigma mean: {stats['sigma_mean']}\n\n")
                f.write(f"Sigma median: {stats['sigma_median']}\n\n")
                f.write(f"Contour fraction mean: {stats['contour_frac_mean']}\n\n")
                f.write(f"Contour fraction median: {stats['contour_frac_median']}\n\n")

    except Exception:
        # Non-fatal: aggregation is best-effort
        pass

    return all_results
