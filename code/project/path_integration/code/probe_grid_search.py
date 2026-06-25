"""
probe_grid_search.py — reduced grid search across all PI design-choice configs.

Runs the same configs as train_model.py but with a trimmed parameter space and
results written to /out/probe/... so they don't overwrite production outputs.

Overrides applied to every config:
  out_path          → unchanged (uses the config's own out_path)
  R_k_avg           → [3, 4]   (reduced sweep instead of [1..6])
  R_self_loops      → 0        (fixed — no self-loops sweep)
  n_nodes           → max of list (e.g. 1024 for 1D, 2048 for 2D)
  training.optim    → ridge, alpha=1e-3  (fast closed-form; overrides any adam config)
  grid_search.run   → [train]  (no kqgr)
  multiverse_overrides → null  (removes universe expansion and combination duplicates)
"""
import os
import sys
import tempfile

import yaml

sys.path.insert(0, '.')
from project.boolean_reservoir.code.parameter import deep_merge
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes

CONFIGS = [
    'config/path_integration/1D/grid_search/design_choices/continuous.yaml',
    'config/path_integration/1D/grid_search/design_choices/discrete.yaml',
    'config/path_integration/2D/grid_search/design_choices/continuous.yaml',
    'config/path_integration/2D/grid_search/design_choices/discrete.yaml',
]

_OVERRIDES = {
    'logging': {
        'grid_search': {
            'n_samples': 1,
            'run': ['train'],
        },
    },
    'model': {
        'variables': {
            'R_k_avg': [3, 4],
            'R_self_loops': 0,
        },
        'training': {
            'epochs': 1,
            'optim': {
                'name': 'ridge',
                'params': {'alpha': 1.0e-3},
            },
        },
    },
    'multiverse_overrides': None,
}


def build_probe_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    merged = deep_merge(raw, _OVERRIDES)
    n_nodes = raw.get('model', {}).get('reservoir_layer', {}).get('n_nodes')
    if isinstance(n_nodes, list):
        merged['model']['reservoir_layer']['n_nodes'] = max(n_nodes)
    return merged


def run_probe(config_path: str):
    cfg = build_probe_config(config_path)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    try:
        boolean_reservoir_grid_search(tmp.name, cpu_max_workers=18)
    finally:
        os.unlink(tmp.name)


if __name__ == '__main__':
    run_on_nodes(CONFIGS, run_fn=run_probe)
