import os
import json
import time
import csv
from typing import Dict, List

from pathlib import Path

try:
    from metrics import calculate_all_metrics, MetricsResult
except Exception:
    calculate_all_metrics = None
    MetricsResult = None

try:
    from PIL import Image
except Exception:
    Image = None


def load_test_metadata(metadata_path: str = "test_images/test_metadata.json") -> Dict:
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def validate_single_image(image_path: str, metadata: Dict, base_dir: str = "test_images") -> Dict:
    result = {
        'image_path': image_path,
        'status': 'failed',
        'metrics': None,
        'processing_time': None,
        'error_message': None
    }
    try:
        full_path = os.path.join(base_dir, image_path)
        if not os.path.exists(full_path):
            result['status'] = 'missing_image'
            return result

        # Load image
        if Image is None:
            raise ImportError('Pillow is required for validation')
        img = Image.open(full_path).convert('RGB')
        image_array = None
        try:
            import numpy as np
            image_array = np.array(img)
        except Exception:
            image_array = None

        params = metadata.get('optimal_parameters', {})
        canny_sigma = params.get('canny_sigma', 0.33)
        min_contour_frac = params.get('min_contour_frac', 0.0001)

        # Import processing functions from streamlit_app
        try:
            from streamlit_app import process_image, generate_svg_from_paths, optimize_svg_string
        except Exception as e:
            raise ImportError(f'Could not import processing functions: {e}')

        t0 = time.time()
        path_data = process_image(image_array, canny_sigma, min_contour_frac)
        if not path_data:
            result['status'] = 'no_significant_edges'
            result['processing_time'] = time.time() - t0
            return result

        svg_string = generate_svg_from_paths(path_data, image_array.shape[1], image_array.shape[0])
        optimized_svg = optimize_svg_string(svg_string)

        reference_svg = None
        expected_rel = metadata.get('expected_output')
        if expected_rel:
            ref_path = expected_rel
            if os.path.exists(ref_path):
                with open(ref_path, 'r', encoding='utf-8') as f:
                    reference_svg = f.read()

        if calculate_all_metrics is not None:
            mr = calculate_all_metrics(svg_string, optimized_svg, reference_svg=reference_svg, original_image_shape=image_array.shape)
            result['metrics'] = mr.__dict__ if mr is not None else None

        result['status'] = 'success'
        result['processing_time'] = time.time() - t0
        return result

    except Exception as e:
        result['error_message'] = str(e)
        result['status'] = 'failed'
        return result


def validate_all_test_cases(metadata_path: str = "test_images/test_metadata.json", base_dir: str = "test_images") -> List[Dict]:
    meta = load_test_metadata(metadata_path)
    results = []
    total = len(meta)
    idx = 0
    for image_rel, md in meta.items():
        idx += 1
        print(f"Processing {idx}/{total}: {image_rel}")
        r = validate_single_image(image_rel, md, base_dir=base_dir)
        results.append(r)
    return results


def generate_validation_report(results: List[Dict], output_dir: str = "validation_reports") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    json_path = os.path.join(output_dir, 'validation_report.json')
    csv_path = os.path.join(output_dir, 'validation_report.csv')
    md_path = os.path.join(output_dir, 'validation_report.md')

    # JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'generated_at': timestamp}, f, indent=2)

    # CSV
    fieldnames = ['image_path', 'status', 'edge_precision', 'edge_recall', 'edge_f1', 'path_count', 'hausdorff_distance', 'processing_time']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            metrics = r.get('metrics') or {}
            writer.writerow({
                'image_path': r.get('image_path'),
                'status': r.get('status'),
                'edge_precision': metrics.get('edge_precision') if isinstance(metrics, dict) else None,
                'edge_recall': metrics.get('edge_recall') if isinstance(metrics, dict) else None,
                'edge_f1': metrics.get('edge_f1') if isinstance(metrics, dict) else None,
                'path_count': metrics.get('path_count') if isinstance(metrics, dict) else None,
                'hausdorff_distance': metrics.get('hausdorff_distance') if isinstance(metrics, dict) else None,
                'processing_time': r.get('processing_time')
            })

    # Markdown summary
    total = len(results)
    successes = sum(1 for r in results if r.get('status') == 'success')
    fails = total - successes
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Validation Report\n\n")
        f.write(f"Total cases: {total}\n\n")
        f.write(f"Successes: {successes}\n\n")
        f.write(f"Failures: {fails}\n\n")
        f.write("## Detailed Results\n\n")
        f.write("| image_path | status | edge_f1 | path_count | hausdorff_distance | processing_time |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            m = r.get('metrics') or {}
            f.write(f"| {r.get('image_path')} | {r.get('status')} | {m.get('edge_f1')} | {m.get('path_count')} | {m.get('hausdorff_distance')} | {r.get('processing_time')} |\n")

    return {'json': json_path, 'csv': csv_path, 'md': md_path}


def compare_validation_runs(report1_path: str, report2_path: str, output_dir: str = "validation_reports") -> Dict:
    """
    Compare two validation JSON reports (produced by generate_validation_report).
    Produces a comparison JSON and Markdown summary containing per-case deltas.
    """
    if not os.path.exists(report1_path) or not os.path.exists(report2_path):
        raise FileNotFoundError("One of the comparison report paths does not exist")

    with open(report1_path, 'r', encoding='utf-8') as f:
        j1 = json.load(f)
    with open(report2_path, 'r', encoding='utf-8') as f:
        j2 = json.load(f)

    r1 = {r.get('image_path'): r for r in j1.get('results', [])}
    r2 = {r.get('image_path'): r for r in j2.get('results', [])}

    images = sorted(set(r1.keys()) | set(r2.keys()))
    comparisons = []
    for img in images:
        a = r1.get(img)
        b = r2.get(img)
        comp = {'image_path': img, 'status_a': a.get('status') if a else None, 'status_b': b.get('status') if b else None}

        ma = (a.get('metrics') or {}) if a else {}
        mb = (b.get('metrics') or {}) if b else {}

        def _val(d, k):
            try:
                return None if d is None else (d.get(k) if isinstance(d, dict) else getattr(d, k, None))
            except Exception:
                return None

        for key in ('edge_f1', 'edge_precision', 'edge_recall', 'path_count', 'hausdorff_distance'):
            va = _val(ma, key)
            vb = _val(mb, key)
            delta = None
            if va is not None and vb is not None:
                try:
                    delta = vb - va
                except Exception:
                    delta = None
            comp[f'{key}_a'] = va
            comp[f'{key}_b'] = vb
            comp[f'{key}_delta'] = delta

        comparisons.append(comp)

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, 'comparison_report.json')
    out_md = os.path.join(output_dir, 'comparison_report.md')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'comparison': comparisons, 'report_a': report1_path, 'report_b': report2_path}, f, indent=2)

    # Write a short markdown summary
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f"# Comparison Report\n\n")
        f.write(f"Report A: {report1_path}\n\n")
        f.write(f"Report B: {report2_path}\n\n")
        f.write("| image_path | status_a | status_b | edge_f1_a | edge_f1_b | edge_f1_delta |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for c in comparisons:
            f.write(f"| {c['image_path']} | {c.get('status_a')} | {c.get('status_b')} | {c.get('edge_f1_a')} | {c.get('edge_f1_b')} | {c.get('edge_f1_delta')} |\n")

    return {'json': out_json, 'md': out_md}


def calculate_category_statistics(results: List[Dict], metadata_path: str = "test_images/test_metadata.json") -> Dict:
    """Aggregate metrics by category (requires metadata with category per test case)."""
    meta = load_test_metadata(metadata_path) if metadata_path and os.path.exists(metadata_path) else {}
    # Map image -> category
    img_to_cat = {k: v.get('category', 'unknown') for k, v in meta.items()} if meta else {}

    stats = {}
    for r in results:
        img = r.get('image_path')
        cat = img_to_cat.get(img, 'unknown')
        m = (r.get('metrics') or {})
        if cat not in stats:
            stats[cat] = {'count': 0, 'edge_f1_sum': 0.0, 'edge_f1_count': 0, 'hausdorff_sum': 0.0, 'hausdorff_count': 0}
        stats[cat]['count'] += 1
        ef = m.get('edge_f1') if isinstance(m, dict) else None
        if ef is not None:
            stats[cat]['edge_f1_sum'] += ef
            stats[cat]['edge_f1_count'] += 1
        hd = m.get('hausdorff_distance') if isinstance(m, dict) else None
        if hd is not None:
            stats[cat]['hausdorff_sum'] += hd
            stats[cat]['hausdorff_count'] += 1

    # finalize averages
    for cat, v in stats.items():
        v['edge_f1_mean'] = (v['edge_f1_sum'] / v['edge_f1_count']) if v['edge_f1_count'] else None
        v['hausdorff_mean'] = (v['hausdorff_sum'] / v['hausdorff_count']) if v['hausdorff_count'] else None

    return stats


def identify_outliers(results: List[Dict], metric: str = 'edge_f1', threshold: float = None) -> List[Dict]:
    """Identify outliers for a given scalar metric. If threshold is None, use mean-2*std rule."""
    vals = []
    for r in results:
        m = (r.get('metrics') or {})
        val = None
        if isinstance(m, dict):
            val = m.get(metric)
        else:
            try:
                val = getattr(m, metric, None)
            except Exception:
                val = None
        if val is not None:
            vals.append(float(val))

    import statistics
    if not vals:
        return []
    mean = statistics.mean(vals)
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if threshold is None:
        threshold = mean - 2 * stdev

    outliers = []
    for r in results:
        m = (r.get('metrics') or {})
        val = None
        if isinstance(m, dict):
            val = m.get(metric)
        else:
            try:
                val = getattr(m, metric, None)
            except Exception:
                val = None
        try:
            if val is not None and float(val) < threshold:
                outliers.append({'image_path': r.get('image_path'), metric: val})
        except Exception:
            continue

    return outliers


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run batch validation against test_images and expected_outputs')
    parser.add_argument('--metadata', default='test_images/test_metadata.json')
    parser.add_argument('--base-dir', default='test_images')
    parser.add_argument('--output-dir', default='validation_reports')
    parser.add_argument('--single', help='Validate a single test case (relative path)')
    parser.add_argument('--compare', nargs=2, help='Compare two validation JSON reports: provide two file paths')
    args = parser.parse_args()

    if args.compare:
        # Compare two existing JSON reports
        a, b = args.compare
        paths = compare_validation_runs(a, b, output_dir=args.output_dir)
        print('Comparison reports generated:', paths)
        return

    if args.single:
        meta = load_test_metadata(args.metadata)
        if args.single not in meta:
            print('Test case not found in metadata')
            return
        r = validate_single_image(args.single, meta[args.single], base_dir=args.base_dir)
        print(r)
    else:
        results = validate_all_test_cases(args.metadata, base_dir=args.base_dir)
        paths = generate_validation_report(results, output_dir=args.output_dir)
        print('Reports generated:', paths)


if __name__ == '__main__':
    main()
