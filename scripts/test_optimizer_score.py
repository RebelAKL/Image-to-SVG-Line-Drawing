import importlib.util
import os

spec = importlib.util.spec_from_file_location('parameter_optimizer', os.path.join(os.path.dirname(__file__), '..', 'parameter_optimizer.py'))
po = importlib.util.module_from_spec(spec)
spec.loader.exec_module(po)
compute_composite_score = po.compute_composite_score

# Create a set of mocked metrics (no reference case)
for pc in [10, 50, 200]:
    for size in [2000, 10000, 50000]:
        m = {'path_count': pc, 'svg_optimized_size_bytes': size}
        score = compute_composite_score(m, has_reference=False)
        print(f'path_count={pc:4d}, size={size:6d} -> score={score:.2f}')

print('\nWith edge_f1 as proxy:')
for f1 in [0.2, 0.5, 0.9]:
    m = {'edge_f1': f1, 'path_count': 50, 'svg_optimized_size_bytes': 8000}
    print(f'edge_f1={f1:.2f} -> score={compute_composite_score(m, has_reference=False):.2f}')
