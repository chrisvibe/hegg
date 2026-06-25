from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.load_save import custom_load_grid_search_data
from project.temporal.code.visualization import plot_kq_and_gr_many_config, plot_kq_and_gr_per_config

def load_data(config_path, df_filter_mask=None):
    P = load_yaml_config(config_path)
    extractions = [
        ('params', lambda p: p, None),
        ('config', lambda p: p.L.kqgr.config if p.L.kqgr else None, None),
        ('M', lambda p: p.L.kqgr, {'kq', 'gr', 'delta', 'spectral_radius'}),
        ('R', lambda p: p.M.R, {'k_avg', 'n_nodes', 'mode', 'init'}),
        ('I', lambda p: p.M.I, {'n_nodes'}),
    ]
    df, _ = custom_load_grid_search_data(config_paths=[config_path], extractions=extractions, df_filter_mask=df_filter_mask, cache_dir=None)
    return P, df

if __name__ == '__main__':
    # path = 'config/temporal/density/grid_search/design_choices/all.yaml'
    # P, df = load_data(path, df_filter_mask=lambda df: (df['R_n_nodes'] == 512) & (df['I_ticks'] == 1))
    path = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'
    # P, df = load_data(path, df_filter_mask=lambda df: (df['R_mode'] == 'heterogeneous') & (df['R_n_nodes'] == 25) & (df['R_init'] == 'random'))
    P, df = load_data(path, df_filter_mask=lambda df: (df['R_mode'] == 'heterogeneous') & (df['R_n_nodes'] == 25) & (df['I_n_nodes'] == 1) & (df['R_init'] == 'random'))
    df['M_delta'] = df['M_delta'].abs()

    plot_kq_and_gr_many_config(df, P, 'many_config.svg')
    plot_kq_and_gr_many_config(df, P, 'many_config_kq_only.svg', metrics=['M_kq'])
    plot_kq_and_gr_many_config(df, P, 'many_config_gr_only.svg', metrics=['M_gr'])
    plot_kq_and_gr_many_config(df, P, 'many_config_delta_only.svg', metrics=['M_delta'])
    plot_kq_and_gr_per_config(df, P)
