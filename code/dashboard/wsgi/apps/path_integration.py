from wsgi.base import load_scatter_dashboard, make_lazy_loader, GRID_SEARCH_EXTRACTIONS_COMMON

paths = [
    'config/path_integration/1D/grid_search/design_choices/continuous_redundancy.yaml',
    'config/path_integration/2D/grid_search/design_choices/continuous_redundancy.yaml',
    'config/path_integration/1D/grid_search/design_choices/continuous.yaml',
    'config/path_integration/2D/grid_search/design_choices/continuous.yaml',
]

extractions = [
    ('T', lambda p: p.L.T,  {'accuracy', 'loss'}),
    ('D', lambda p: p.D,    {'dimensions', 'mode'}),
] + GRID_SEARCH_EXTRACTIONS_COMMON

__getattr__ = make_lazy_loader(__name__, paths, extractions,
                               dashboard_kwargs={'initial_views_dir': '/out/dashboard/scatter/path_integration'})
