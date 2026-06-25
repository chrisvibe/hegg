"""
Extraction cache warmup. Run via: docker compose run --rm warmup

Calls custom_load_grid_search_data with warmup=True for each dataset.
On a warm cache this is a no-op (existence checks only, no data loaded).
On a cold or stale cache it runs the extractions and writes the parquet files.
"""
import sys

from wsgi.base import warm_cache
from wsgi.apps.path_integration import paths as pi_paths, extractions as pi_extractions
from wsgi.apps.temporal import paths as t_paths, extractions as t_extractions
from wsgi.apps.figure1_snyder_2012 import paths as f1_paths, extractions as f1_extractions

print('Starting extraction cache warmup...', file=sys.stderr, flush=True)

for label, paths, extractions in [
    ('path_integration', pi_paths, pi_extractions),
    ('temporal',         t_paths,  t_extractions),
    ('figure1',          f1_paths, f1_extractions),
    # polar reuses the above paths/extractions — already covered
]:
    print(f'  {label}...', file=sys.stderr, flush=True)
    warm_cache(paths, extractions)

print('Warmup complete.', file=sys.stderr, flush=True)
