from wsgi.base import load_scatter_dashboard, make_lazy_loader, RESERVOIR_METRICS_EXTRACTION

paths = [
    'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml',
]

_UNIVERSE_LABELS = {
    'kqgr_heterogeneous':         'replication_decoupled',
    'kqgr_homogeneous':           'replication_decoupled',
    'kqgr_heterogeneous_identity': 'replication_decoupled_guarantee',
    'kqgr_homogeneous_identity':   'replication_decoupled_guarantee',
    'kqgr_heterogeneous_real':    'replication',
    'kqgr_homogeneous_real':      'replication',
}

extractions = [
    ('kqgr', lambda p: p.L.kqgr, {'kq', 'gr', 'delta'}),
    RESERVOIR_METRICS_EXTRACTION,
    ('L', lambda p: p.L, {'universe'}),
    ('L_universe_label', lambda p: _UNIVERSE_LABELS.get(p.L.universe, p.L.universe), None),
    ('D', lambda p: p.D, {'tau', 'evaluation'}),
    ('R', lambda p: p.M.R, {'n_nodes', 'init'}),
    ('R_k_avg', lambda p: p.M.variables.R_k_avg, None),
    ('R_mode', lambda p: p.M.variables.get('R_mode') or (
        'heterogeneous' if 'heterogeneous' in (p.L.universe or '') else 'homogeneous'
    ), None),
]

__getattr__ = make_lazy_loader(__name__, paths, extractions,
                               dashboard_kwargs={'initial_views_dir': '/out/dashboard/scatter/snyder_2012'})

if __name__ == '__main__':
    app.run(port=8053, debug=False, dev_tools_hot_reload=False)
