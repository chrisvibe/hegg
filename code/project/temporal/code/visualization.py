from project.boolean_reservoir.code.visualization import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from project.boolean_reservoir.code.parameter import Params
import orjson
import matplotlib.pyplot as plt
import seaborn as sns
from labellines import labelLines
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from pathlib import Path
            

def plot_many_things(model, dataset, history, evaluation='test'):
    y_test = dataset.data['y_' + f'{evaluation}'][:500]
    y_hat_test = model(dataset.data['x_' + f'{evaluation}'][:500])
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[0, 1])
    # plot_dynamics_history(model.save_path)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

def group_df_data_by_parameters(df):
    def canonical_key(params_json):
        d = orjson.loads(params_json)
        d['model']['reservoir_layer']['k_min'] = None
        d['model']['reservoir_layer']['k_avg'] = None
        d['model']['reservoir_layer']['k_max'] = None
        d['model']['reservoir_layer']['seed'] = None
        d['model']['input_layer']['seed'] = None
        d['model']['output_layer']['seed'] = None
        d['model']['training']['seed'] = None
        d['dataset']['seed'] = None
        d['dataset']['path'] = None
        d['logging'] = None
        return json.dumps(d, sort_keys=True, default=str)
    df = df.sort_values(by='config')
    df['group_params_str'] = df['params_json'].apply(canonical_key)
    return df.groupby(df['group_params_str'])

def plot_kq_and_gr(df, P: Params, filename: str, metrics: list[str] = ['M_kq', 'M_gr', 'M_delta']):
    D = P.U.kqgr.D
    subtitle = f"Mode: {P.M.R.mode}, Nodes: {P.M.R.n_nodes}, Bit Stream Length: {D.bits}, Samples per config: {D.samples}"
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create a color mapper for spectral radius
    norm = plt.Normalize(df['M_spectral_radius'].min(), df['M_spectral_radius'].max())
    
    # Predefined marker styles
    markers = ['<', '>', '^']
    
    # Store handles and labels for manual legend creation
    scatter_handles = []
    trend_lines = []
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        
        color = plt.cm.tab10(i % 10)
        
        # Scatter plot
        scatter = ax.scatter(
            df['R_k_avg'],
            df[metric],
            label=metric,
            marker=markers[i % len(markers)],
            c=df['M_spectral_radius'],
            cmap='viridis',
            norm=norm,
            edgecolors='black',
            linewidth=1,
            alpha=0.7,
            s=30,
        )
        scatter_handles.append(scatter)
        
        # Regression line
        trend = sns.regplot(
            x='R_k_avg',
            y=metric,
            data=df,
            scatter=False,
            lowess=True,
            ax=ax,
            color=color,
            line_kws={'linestyle': '--', 'linewidth': 2}
        )
        trend_lines.append(trend.lines[-1])  # use -1 to get the most recent line
    
    ax.set_ylabel('Rank')
    ax.set_xlabel('Average K')
    
    # Add colorbar for spectral radius
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02, label='Spectral Radius')
    
    plt.subplots_adjust(right=0.8)
    
    # Custom legend with scatter points
    first_legend = ax.legend(
        scatter_handles,
        [h.get_label() for h in scatter_handles],
        title='Points',
        loc='upper left',
        bbox_to_anchor=(0.0, 1.0)
    )
    ax.add_artist(first_legend)
    
    # Trend lines legend
    trend_legend = ax.legend(
        [plt.Line2D([0], [0], color=plt.cm.tab10(i % 10), linestyle='--', linewidth=2) for i in range(len(metrics))],
        metrics,
        title='Lines (lowess)',
        loc='upper left',
        bbox_to_anchor=(0.0, 0.8)
    )
    
    plt.title('Reservoir Metrics: Kernel Quality, Generalization Rank, Delta', fontsize=16)
    fig.text(0.5, 0.01, subtitle, ha='right', fontsize=12)
    
    save_path = P.L.out_path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close(fig)

def plot_kq_and_gr_per_config(df, P: Params, metrics: list[str] = ['M_kq', 'M_gr', 'M_delta']):
    for i, (_, subset) in enumerate(group_df_data_by_parameters(df)):
        plot_kq_and_gr(subset, P, f'config_{i}_kq_and_gr.svg', metrics=metrics)

def plot_kq_and_gr_many_config(df, P: Params, filename: str, metrics: list[str] = ['M_kq', 'M_gr', 'M_delta']):
    grouped_df = group_df_data_by_parameters(df)
    fig, ax = plt.subplots(figsize=(18, 8))
    color_idx = 0
    xvals = list()
    shift = 5
    g = len(grouped_df)
    val_range = grouped_df['R_k_avg'].max().max() - shift

    for name, subset in grouped_df:
        n_metrics = len(metrics)
        for i, metric in enumerate(metrics):
            if metric not in subset.columns:
                continue
                
            color = plt.cm.tab10(color_idx % 10)
            
            # Aggregate by R_k_avg
            data = subset.groupby('R_k_avg').agg({metric: 'mean'}).reset_index()
            data.sort_values(by='R_k_avg', inplace=True)
            
            x_sorted = data['R_k_avg'].values
            y_sorted = data[metric].values
            
            # Cubic spline interpolation
            # cubic_spline = UnivariateSpline(x_sorted, y_sorted, k=3, s=0) # go thru each datapoint
            cubic_spline = UnivariateSpline(x_sorted, y_sorted, k=3, s=len(x_sorted))
            x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            y_smooth = cubic_spline(x_fine)
            
            ax.plot(x_fine, y_smooth, linestyle='--', linewidth=2, color=color, label=f'{color_idx}-{metric}')
            
            xvals.append(color_idx * val_range / g + ((i + 1) / n_metrics) - 1 + shift)
        color_idx += 1
    
    lines = plt.gca().get_lines()
    labelLines(lines, align=False, xvals=xvals, fontsize=10)
    
    ax.set_ylabel('Rank')
    ax.set_xlabel('Average K')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.subplots_adjust(right=0.8)
    plt.title('Reservoir Metrics: Kernel Quality, Generalization Rank, Delta', fontsize=16)
    
    save_path = P.L.out_path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close(fig)

def plot_optimal_k_avg_vs_configuration(out_path, df):
    filtered_df = df[df['metric'] == 'delta']
    grouped_df = group_df_data_by_parameters(filtered_df)
    data = dict()
    for i, (_, subset) in enumerate(grouped_df):
        # p = subset['params'].iloc[1]
        # group_id = f'{i}: {p.M.I.perturbation}-{p.M.I.connection}-{p.M.R.mode}-{p.M.R.init}'
        max_values = subset.groupby(['delay', 'sample'])['value'].idxmax() # max delta per delay-sample (over many k_avg)
        max_subset = subset.loc[max_values]
        max_subset['k_avg*'] = max_subset['k_avg'].mean() # average delta* over the grid_search samples
        max_subset['group_id'] = i
        data[i] = max_subset
    plot_data = pd.concat(data, ignore_index=True)
    plot_data.sort_values(by='group_id', inplace=True)
    flattened_params = lambda x: pd.concat([
        pd.Series({f"I.{k}": v for k, v in x.M.I.model_dump().items()}),
        pd.Series({f"R.{k}": v for k, v in x.M.R.model_dump().items()}),
        # pd.Series({f"O.{k}": v for k, v in x.M.O.model_dump().items()}),
        # pd.Series({f"T.{k}": v for k, v in x.M.T.model_dump().items()}),
        pd.Series({f"D.{k}": v for k, v in x.D.model_dump().items()}),
    ])
    df_flattened_params = plot_data['params'].apply(lambda p: flattened_params(p))
    plot_data = pd.concat([plot_data, df_flattened_params], axis=1)
    plot_data['hue'] = plot_data[['D.delay', 'R.mode']].apply(tuple, axis=1)
    legend = '(delay, mode)'

    save_path = out_path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    filename='optimal_k_avg_vs_configuration'
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.scatterplot(data=plot_data, x='group_id', y='k_avg', hue='hue', palette='viridis', ax=ax)
    plt.legend(title=legend)
    xticks = range(int(plot_data['group_id'].min()), int(plot_data['group_id'].max()) + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)
    plt.xlabel('Configuration vs K_avg*')
    plt.ylabel('k_avg*')
    plt.title('')
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()

    filename='optimal_k_avg_vs_configuration_lines'
    plot_data = plot_data.groupby(['group_id']).first().reset_index() # now k_avg* makes sense but not k_avg

    # make group_id invariant to delay 3 and 5 (assumes only delay is different in yaml for grid search)
    features = ['I.connection', 'I.perturbation', 'R.init']
    plot_data['combo'] = plot_data.apply(lambda row: tuple(row[feature] for feature in features), axis=1)
    plot_data['group_id'], _ = pd.factorize(plot_data['combo'])

    # # Assign a new order to 'group_id' to approximate monotonically decreasing k_avg* 
    # distance_df = plot_data.groupby(['group_id'], as_index=False)['k_avg*'].sum() # all hue combos are aggregated
    # distance_df.rename(columns={'k_avg*': 'distance'}, inplace=True)
    # plot_data = pd.merge(plot_data, distance_df, on=['group_id'], how='left')
    # plot_data.sort_values(by='distance', ascending=True, inplace=True)

    # Create a new DataFrame with each configuration value as a row and group IDs as columns
    unique_values = {feature: plot_data[feature].unique() for feature in features}
    config_map = pd.DataFrame(index=[str(value) for feature in features for value in unique_values[feature]],
                            columns=plot_data['group_id'].unique())

    # Fill the DataFrame with True/False depending on whether the configuration is present for each group_id
    plot_data['group_id_index'] = plot_data['group_id']
    plot_data.set_index('group_id_index', inplace=True)
    for idx, row in plot_data.iterrows():
        for feature in features:
            for value in unique_values[feature]:
                config_map.loc[value, idx] = value == row[feature]
    config_map.fillna(False, inplace=True)

    # custom_col_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # custom_col_order = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17]
    # custom_col_order = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
    # custom_col_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # custom_col_order = [16, 18, 17, 12, 19, 11, 14, 10, 8, 13, 9, 15, 4, 7, 6, 5, 1, 0, 2, 3]
    # custom_col_order = reduce_contention_index(plot_data)
    custom_col_order = optimize_ordering_with_scipy(plot_data)
    config_map = config_map.iloc[:, custom_col_order]

    custom_row_order = [
        'out-0:b:1/b',
        'out-1:1:1',
        'out-1:b:1/b',
        'out-2:2:1',
        'out-3:3:1',
        'override',
        'xor',
        'random',
        'zeros',
    ]
    # config_map = config_map.reindex(custom_row_order)
    scores = np.arange(len(config_map.columns))
    weighted_scores = config_map * scores
    row_sum_scores = np.sum(weighted_scores, axis=1)
    num_true_per_row = config_map.sum(axis=1)
    average_scores = row_sum_scores / num_true_per_row
    ranked_order = average_scores.rank(ascending=False).astype(int) - 1
    config_map = config_map.iloc[ranked_order.argsort()]

    fig, ax = plt.subplots(figsize=(12, 7))
    group_id_to_int = {gid: i for i, gid in enumerate(custom_col_order)}
    plot_data['x'] = plot_data['group_id'].map(group_id_to_int) + 0.5
    sns.lineplot(data=plot_data, x='x', y='k_avg*', hue='hue', palette='viridis', ax=ax, alpha=1, zorder=2)
    ax.set_xticklabels([])  # This hides the x-labels
    cmap = ListedColormap(['white', 'black'])
    config_map.columns = config_map.columns.map(str)
    ax2 = ax.twinx()
    sns.heatmap(config_map.astype(int), cmap=cmap, cbar=False, linecolor='black', linewidths=0.1, ax=ax2, alpha=0.1, zorder=1)

    # TODO labelines doesnt work1!!!
    # labelLines(plt.gca().get_lines(), fontsize=12, align=True)
    ax.legend(title=legend)
    ax.set_xlabel(f'combinations: {features}')
    ax2.set_xlabel(f'combinations: {features}')
    # ax2.set_xlabel(f'combinations: {features}')
    # ax2.set_ylabel('k_avg*')
    # ax2.set_ylabel('Design choice (True/False)')
    plt.title('Design choices vs K_avg*')
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close()

def reduce_contention_index(plot_data):
    """
    Function to compute a consolidated index that minimizes contention across sorting requirements based on different hues.
    
    Parameters:
    plot_data : pd.DataFrame
        The input DataFrame containing 'hue', 'k_avg*', and index columns.

    Returns:
    np.ndarray
        An array representing the consolidated index order for reduced contention.
    """
    # Retrieve unique hues
    unique_hues = plot_data['hue'].unique()

    # Create a dictionary to store ranks for each group_id_index
    rank_dict = {index: [] for index in plot_data.index}

    # Calculate ranks for each hue condition
    for hue in unique_hues:
        # Filter data for current hue
        hue_data = plot_data[plot_data['hue'] == hue]

        # Sort based on `k_avg*` and compute ranks
        sorted_data = hue_data.sort_values('k_avg*')
        ranks = sorted_data['k_avg*'].rank(method='dense')

        # Assign ranks to group_id_indices
        for index, rank in zip(sorted_data.index, ranks):
            rank_dict[index].append(rank)

    # Calculate average rank for each group_id_index
    average_ranks = {index: sum(ranks) / len(ranks) for index, ranks in rank_dict.items()}

    # Create a DataFrame to store and sort average ranks
    rank_df = pd.DataFrame(list(average_ranks.items()), columns=['group_id_index', 'average_rank'])
    rank_df.sort_values('average_rank', inplace=True)

    # Return the consolidated index order
    return rank_df['group_id_index'].values

def optimize_ordering_with_scipy(df, value_col='k_avg*', hue_col='hue', group_col='group_id'):
    """
    Find optimal ordering of group_ids using scipy's linear_sum_assignment.
    
    This approach transforms the problem into an assignment problem where we're 
    trying to assign positions to groups to minimize the total cost.
    
    Args:
        df: Input DataFrame with columns for group_id, hue, and value_col
        value_col: Column name containing the values to optimize (default: 'k_avg*')
        hue_col: Column name for the hue groups (default: 'hue')
        group_col: Column name for the group IDs (default: 'group_id')
    
    Returns:
        Tuple of (optimal_ordering, sorted_dataframe)
    """
    # Extract unique groups and hues
    unique_groups = df[group_col].unique()
    unique_hues = df[hue_col].unique()
    
    # Create a dictionary to hold values for each (hue, group) pair
    value_dict = {hue: {} for hue in unique_hues}
    
    # Populate the dictionary with values
    for hue in unique_hues:
        for group in unique_groups:
            subset = df[(df[hue_col] == hue) & (df[group_col] == group)]
            if not subset.empty:
                value_dict[hue][group] = subset[value_col].mean()
    
    # Create a cost matrix where cost[i,j] is the cost of placing group i at position j
    n = len(unique_groups)
    cost_matrix = np.zeros((n, n))
    
    # Convert groups to a list for indexing
    groups_list = list(unique_groups)
    
    # Fill the cost matrix
    for i, group in enumerate(groups_list):
        for j in range(n):
            # Cost of placing group at position j
            cost = 0
            
            for hue in unique_hues:
                if group not in value_dict[hue]:
                    continue
                    
                current_val = value_dict[hue][group]
                
                # Look at all pairs with groups that would be placed before this one
                for k, prev_group in enumerate(groups_list):
                    if prev_group == group or prev_group not in value_dict[hue]:
                        continue
                        
                    prev_val = value_dict[hue][prev_group]
                    
                    # If this group would be placed after prev_group (i.e., j > position of prev_group)
                    # but its value is less, add a penalty
                    for pos_k in range(n):
                        if pos_k < j and current_val < prev_val:
                            cost += prev_val - current_val
                        elif pos_k > j and current_val > prev_val:
                            cost += current_val - prev_val
            
            cost_matrix[i, j] = cost
    
    # Use scipy's linear_sum_assignment to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the optimal ordering based on the assignment
    ordering = [None] * n
    for i, j in zip(row_ind, col_ind):
        ordering[j] = groups_list[i]
    
    return ordering

if __name__ == '__main__':
    pass
    # from project.boolean_reservoir.code.visualizations import plot_grid_search
    # plot_grid_search('out/temporal/density/grid_search/initial_sweep/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/initial_sweep/log.h5')