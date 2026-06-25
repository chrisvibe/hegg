"""
Shared loader for scatter dashboard wsgi apps.

Filters config paths to those with existing data files, so apps start
with whatever data is available rather than crashing when runs are missing.
"""
import os
import sys
from pathlib import Path

_DEFAULT_CACHE_DIR = Path(os.environ.get('BOOLEAN_RESERVOIR_CACHE_DIR', '/tmp/boolean_reservoir_cache'))

from dashboards.scatter import create_scatter_dashboard
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data, get_data_path, is_cache_warm
from project.boolean_reservoir.code.utils.metrics import get_reservoir_metrics, reservoir_key

RESERVOIR_METRICS_EXTRACTION = ('M', lambda p: get_reservoir_metrics(p), {'lut_p'}, reservoir_key)

# Extractions shared across all grid-search scatter apps.
# Each app prepends its task-specific T and D entries, then appends this list.
GRID_SEARCH_EXTRACTIONS_COMMON = [
    ('kqgr',       lambda p: p.L.kqgr,   {'kq', 'gr', 'delta'}),
    RESERVOIR_METRICS_EXTRACTION,
    ('L',          lambda p: p.L,         {'universe', 'out_path'}),
    ('L_out_name', lambda p: Path(p.L.out_path).name if p.L.out_path else None, None),
    ('kqgr',       lambda p: p.U.kqgr.D, {'tau', 'tau_mode', 'tau_axis', 'evaluation'}),
    ('O',          lambda p: p.M.O,       {'mode'}),
    ('I',          lambda p: p.M.I,       {'perturbation', 'encoding', 'redundancy', 'ticks'}),
    ('R',          lambda p: {
        **dict(p.M.R or {}),
        **{k[2:]: v for k, v in (p.M.variables or {}).items() if k.startswith('R_')},
    }, {'mode', 'k_avg', 'self_loops', 'n_nodes', 'init'}),
]


def make_lazy_loader(module_name: str, paths: list, extractions: list,
                     dashboard_kwargs: dict = None):
    """Return a module-level __getattr__ that defers load_scatter_dashboard until app/server is accessed."""
    def __getattr__(name):
        if name in ('app', 'server'):
            import sys
            mod = sys.modules[module_name]
            mod.app, mod.server = load_scatter_dashboard(paths, extractions, dashboard_kwargs=dashboard_kwargs)
            return getattr(mod, name)
        raise AttributeError(name)
    return __getattr__


_DEFAULT_VIEWS_DIR = Path('/out/dashboard')


def views_dir(path_str, default: Path = _DEFAULT_VIEWS_DIR) -> Path:
    """Resolve a user-supplied views directory string, falling back to default."""
    return Path(path_str.strip()) if path_str and path_str.strip() else default


def list_view_options(path_str=None, default: Path = _DEFAULT_VIEWS_DIR) -> list:
    """Return [{label, value}] for all saved views in the given directory."""
    vdir = views_dir(path_str, default)
    if not vdir.exists():
        return []
    return [{'label': p.stem, 'value': p.stem} for p in sorted(vdir.glob('*.json'))]


def is_light_theme(theme) -> bool:
    return bool(theme and 'light' in theme)


def register_config_file_route(app, url_prefix, allowed_root=Path('/code/config')):
    """Register a GET route at {url_prefix}config-file that serves YAML files.

    Accepts ?path=<path-relative-to-/code> and returns the file as plain text.
    Requests outside allowed_root are rejected with 403.
    """
    from flask import request as _req, Response as _Resp

    @app.server.route(f'{url_prefix}config-file')
    def _serve_config_file():
        rel  = _req.args.get('path', '')
        full = Path('/code') / rel
        try:
            if not full.resolve().is_relative_to(allowed_root.resolve()):
                return _Resp('Forbidden', status=403, content_type='text/plain')
        except Exception:
            return _Resp('Forbidden', status=403, content_type='text/plain')
        if not full.exists():
            return _Resp('Not found', status=404, content_type='text/plain')
        return _Resp(full.read_text(), content_type='text/plain; charset=utf-8')


def register_view_refresh(app, open_trigger_id, list_options_fn, body_state_id=None,
                          view_picker_id='view-picker', views_dir_id='views-dir'):
    """Register callbacks to keep a view-picker dropdown in sync with the filesystem.

    Registers two callbacks:
    - Refreshes on views-dir input change
    - Refreshes when the panel opens (open_trigger_id click)
      body_state_id: if given, only refresh when that element's style.display == 'none'
                     (panel is currently closed and about to open)

    view_picker_id / views_dir_id: override for dashboards with prefixed component IDs
    (e.g. trace uses 'tp-view-select' and 'tp-views-dir').
    """
    from dash import Output, Input, State, no_update

    @app.callback(
        Output(view_picker_id, 'options', allow_duplicate=True),
        Input(views_dir_id, 'value'),
        prevent_initial_call=True,
    )
    def _on_dir_change(views_dir_str):
        return list_options_fn(views_dir_str)

    if body_state_id:
        @app.callback(
            Output(view_picker_id, 'options', allow_duplicate=True),
            Input(open_trigger_id, 'n_clicks'),
            State(body_state_id, 'style'),
            State(views_dir_id, 'value'),
            prevent_initial_call=True,
        )
        def _on_panel_open(_, body_style, views_dir_str):
            if (body_style or {}).get('display') == 'none':
                return list_options_fn(views_dir_str)
            return no_update
    else:
        @app.callback(
            Output(view_picker_id, 'options', allow_duplicate=True),
            Input(open_trigger_id, 'n_clicks'),
            State(views_dir_id, 'value'),
            prevent_initial_call=True,
        )
        def _on_panel_open(_, views_dir_str):
            return list_options_fn(views_dir_str)


def _filter_existing(config_paths: list) -> list:
    existing = []
    for p in config_paths:
        try:
            data_path = get_data_path(p)
            if data_path.exists():
                existing.append(p)
            else:
                print(f'INFO: no data yet for {p}, skipping', file=sys.stderr, flush=True)
        except Exception as e:
            print(f'WARNING: could not resolve {p}: {e}', file=sys.stderr, flush=True)
    return existing


def warm_cache(config_paths: list, extractions: list, **loader_kwargs):
    """Populate the extraction cache for config_paths without loading data into memory."""
    existing = _filter_existing(config_paths)
    if not existing:
        return
    loader_kwargs.setdefault('cache_dir', _DEFAULT_CACHE_DIR)
    custom_load_grid_search_data(config_paths=existing, extractions=extractions,
                                 warmup=True, **loader_kwargs)


def load_scatter_dashboard(config_paths: list, extractions: list,
                   dashboard_kwargs: dict = None, **loader_kwargs):
    """Load data and create a scatter dashboard, skipping missing data files.

    Returns (app, server). If no data exists yet, returns a placeholder app.
    """
    existing = _filter_existing(config_paths)
    url_prefix = os.environ.get('URL_PREFIX', '/')

    if not existing:
        print('INFO: no data available, starting placeholder app', file=sys.stderr, flush=True)
        from dash import Dash, html
        app = Dash(__name__, url_base_pathname=url_prefix)
        app.layout = html.Div('Data not yet available.', style={'padding': '40px', 'fontFamily': 'monospace'})
        return app, app.server

    loader_kwargs.setdefault('cache_dir', _DEFAULT_CACHE_DIR)
    df, factors = custom_load_grid_search_data(config_paths=existing, extractions=extractions, **loader_kwargs)
    app = create_scatter_dashboard(df, factors, url_prefix=url_prefix, **(dashboard_kwargs or {}))
    return app, app.server
