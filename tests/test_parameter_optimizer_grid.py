from parameter_optimizer import generate_parameter_grid, GridSearchConfig


def test_generate_parameter_grid_simple():
    cfg = GridSearchConfig()
    cfg.search_type = 'quick'
    combos = list(generate_parameter_grid(cfg))
    # quick search exposes a 5x5 grid
    assert len(combos) == 25
    # values are (sigma, frac) tuples
    assert all(isinstance(c, tuple) and len(c) == 2 for c in combos)
