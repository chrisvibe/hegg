from project.boolean_reservoir.code.utils.load_save import load_grid_search_data
from project.temporal.code.stat import process_grid_search_data, polar_design_plot, graph_accuracy_vs_k_avg
import pandas as pd
from pathlib import Path


def load_custom_data(variable, dimension, selector:str):
    training_paths = list()

    if selector == '5_steps':
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/heterogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/homogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/heterogeneous_stochastic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/homogeneous_stochastic.yaml')

    if selector == 'no_self_loops':
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/no_self_loops/heterogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/no_self_loops/homogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/no_self_loops/heterogeneous_stochastic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/no_self_loops/homogeneous_stochastic.yaml')

    if selector == '3_steps':
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/3_steps/heterogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/3_steps/homogeneous_deterministic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/3_steps/heterogeneous_stochastic.yaml')
        training_paths.append(f'config/path_integration/{dimension}D/grid_search/3_steps/homogeneous_stochastic.yaml')

    d_set = {}
    i_set = {'connection', 'perturbation'}
    r_set = {'mode', 'k_avg', 'init'}
    factors = sorted([f'D_{x}' for x in d_set] + [f'I_{x}' for x in i_set] + [f'R_{x}' for x in r_set])
    data = list()

    data = list()
    for path in training_paths: # concat data
        df_i, _ = load_grid_search_data(config_paths=path)
        df_i = process_grid_search_data(df_i, d_set, i_set, r_set)
        data.append(df_i)
    df_train = pd.concat(data, ignore_index=True)

    # Note: make sure all factors represent main variations s.t. we get normal distributions within the groups
    df_train = df_train[df_train['I_connection'] != 'out-0:b:1/b'] # terrible performance, no point in including 
    factors = list(df_train[factors].nunique()[df_train[factors].nunique() > 1].index)

    df_metric = None

    df = aggregate_and_merge_data(df_metric, df_train, factors)
    df, factors = fix_combo(df, factors)
    groups_dict = {k: v[variable].values for k, v in df.groupby('combo')}
    return df, factors, groups_dict

def fix_combo(df, factors):
    scores = [8, 2, 1, 0, 7] # manually set new order of factors to make most important factor first (put k_avg last)
    factors = sorted(factors, key=lambda item: scores[factors.index(item)], reverse=True)
    df['combo'] = df['combo'].apply(lambda x: tuple(sorted(x, key=lambda item: scores[x.index(item)], reverse=True)))
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    df['combo_no_k_avg'] = df['combo'].apply(lambda row: row[:-1])
    df['combo_no_k_avg_str'] = df['combo_no_k_avg'].apply(lambda t: "_".join(map(str, t)))
    return df, factors

def aggregate_and_merge_data(df1, df2, factors):
    # max_values = df1.groupby(['D_delay', 'sample'])['delta'].idxmax() # max delta per delay-sample (over many k_avg)
    # max_subset = df1.loc[max_values]
    # max_subset['k_avg*'] = max_subset['k_avg'].mean() # average delta* over the grid_search samples

    # Note that dataset for KQ and GR dataframe, aka df1, should not be merged with df2 with dataset-based design choices as the metric is "dataset-agnostic"
    # df1 = df1.convert_dtypes()
    df2 = df2.convert_dtypes()
    # df1['combo'] = df1.apply(lambda row: tuple(row[feature] for feature in factors if feature[0:] != 'D_'), axis=1)
    # df1 = df1[['combo', 'kq', 'gr', 'delta', 'spectral_radius']]
    # df1 = df1.groupby('combo', as_index=False).mean(numeric_only=True)
    # df2['combo'] = df2.apply(lambda row: tuple(row[feature] for feature in factors if feature[0:] != 'D_'), axis=1)
    # df = pd.merge(df1, df2, on=['combo'], how='inner')
    # df.columns = [col[:-2] if col.endswith('_x') else col for col in df.columns]
    # df.drop([col for col in df.columns if col.endswith('_y')], axis=1, inplace=True)
    
    df = df2

    df['combo'] = df.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)
    return df


if __name__ == '__main__':

    # out_path = Path('/out/path_integration/stats/design_evaluation')
    # success_thresh = 0.3
    # response = 'accuracy'

    # # statistical evauluation
    # ####################################
    # for i in [1, 2]:
    #     path = out_path / f'{i}D'
    #     print(path)
    #     print('#'*60)
    #     df, factors, groups_dict = load_custom_data(response, i, '5_steps')
    #     polar_design_plot(path, df, factors, success_thresh, f'task: {i}D path integration', ascending=[1, 1, 0, 1, 1])
    #     graph_accuracy_vs_k_avg(path, df, success_thresh)


    # out_path = Path('/out/path_integration/stats/design_evaluation/no_self_loops')
    # success_thresh = 0.3
    # response = 'accuracy'

    # # statistical evauluation
    # ####################################
    # for i in [1, 2]:
    #     path = out_path / f'{i}D'
    #     print(path)
    #     print('#'*60)
    #     df, factors, groups_dict = load_custom_data(response, i, 'no_self_loops')
    #     polar_design_plot(path, df, factors, success_thresh, f'task: {i}D path integration', ascending=[1, 1, 0, 1, 1])
    #     graph_accuracy_vs_k_avg(path, df, success_thresh)


    out_path = Path('/out/path_integration/stats/design_evaluation/3_steps')
    success_thresh = 0.3
    response = 'accuracy'

    # statistical evauluation
    ####################################
    for i in [1, 2]:
        path = out_path / f'{i}D'
        print(path)
        print('#'*60)
        df, factors, groups_dict = load_custom_data(response, i, '3_steps')
        polar_design_plot(path, df, factors, success_thresh, f'task: {i}D path integration', ascending=[1, 1, 0, 1, 1])
        graph_accuracy_vs_k_avg(path, df, success_thresh)