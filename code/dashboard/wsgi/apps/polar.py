import os
import sys

from dash import Dash, html

from wsgi.base import _filter_existing
from wsgi.apps.path_integration    import paths as _pi_paths,  extractions as _pi_extractions
from wsgi.apps.temporal            import paths as _t_paths,   extractions as _t_extractions
from wsgi.apps.figure1_snyder_2012 import paths as _f1_paths,  extractions as _f1_extractions
from dashboards.polar import create_polar_dashboard, _config_label


def _build_registry():
    entries = []
    for app_paths, app_extractions in [
        (_pi_paths, _pi_extractions),
        (_t_paths,  _t_extractions),
        (_f1_paths, _f1_extractions),
    ]:
        for p in app_paths:
            entries.append({'path': p, 'extractions': app_extractions,
                            'label': _config_label(p)})
    return entries


_REGISTRY = _build_registry()


def __getattr__(name):
    if name in ('app', 'server'):
        url_prefix = os.environ.get('URL_PREFIX', '/')
        available  = [e for e in _REGISTRY if _filter_existing([e['path']])]
        if not available:
            _app = Dash(__name__, url_base_pathname=url_prefix)
            _app.layout = html.Div('Data not yet available.',
                                   style={'padding': '40px', 'fontFamily': 'monospace'})
        else:
            _app = create_polar_dashboard(available, url_prefix=url_prefix)
        mod = sys.modules[__name__]
        mod.app, mod.server = _app, _app.server
        return getattr(mod, name)
    raise AttributeError(name)
