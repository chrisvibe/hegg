from pathlib import Path
from project.boolean_reservoir.code.parameter import load_yaml_config, save_yaml_config
from project.boolean_reservoir.code.visualization import plot_grid_search 
from project.boolean_reservoir.code.utils.utils import override_symlink
from project.boolean_reservoir.code.utils.load_save import load_grid_search_data
import pandas as pd
from scipy.stats import f_oneway, levene, shapiro, kruskal, anderson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.genmod.families import Binomial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import fisher_exact, chi2_contingency
import seaborn as sns
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Any
import warnings
pio.templates.default = "plotly_white"
matplotlib.use('Agg')

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def load_custom_data(variable, one_hot_selector, delay, window_size):
    kq_and_gr_paths = list()
    training_paths = list()

    if one_hot_selector[0] == '1':
        kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_deterministic.yaml')
        kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/homogeneous_stochastic.yaml')
        # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_deterministic.yaml')
        # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_3/heterogeneous_stochastic.yaml')

    if one_hot_selector[1] == '1':
        kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_deterministic.yaml')
        kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/homogeneous_stochastic.yaml')
        # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_deterministic.yaml')
        # kq_and_gr_paths.append('config/temporal/kq_and_gr/fixed_delay/delay_5/heterogeneous_stochastic.yaml')

    if one_hot_selector[2] == '1':
        training_paths.append('config/temporal/density/grid_search/homogeneous_deterministic.yaml')
        training_paths.append('config/temporal/density/grid_search/homogeneous_stochastic.yaml')
        # training_paths.append('config/temporal/density/grid_search/heterogeneous_deterministic.yaml')
        # training_paths.append('config/temporal/density/grid_search/heterogeneous_stochastic.yaml')

    if one_hot_selector[3] == '1':
        training_paths.append('config/temporal/parity/grid_search/homogeneous_deterministic.yaml')
        training_paths.append('config/temporal/parity/grid_search/homogeneous_stochastic.yaml')
        # training_paths.append('config/temporal/parity/grid_search/heterogeneous_deterministic.yaml')
        # training_paths.append('config/temporal/parity/grid_search/heterogeneous_stochastic.yaml')

    d_set = {}
    d_set = {'task', 'delay', 'window_size'}
    i_set = {'connection', 'perturbation'}
    r_set = {'mode', 'k_avg', 'init'}
    factors = sorted([f'D_{x}' for x in d_set] + [f'I_{x}' for x in i_set] + [f'R_{x}' for x in r_set])
    data = list()
    for path in kq_and_gr_paths: # concat data
        df_i, _ = load_grid_search_data(config_paths=path)
        df_i = process_grid_search_data_kq_and_gr(df_i, d_set, i_set, r_set)
        data.append(df_i)
    df_metric = pd.concat(data, ignore_index=True)

    data = list()
    for path in training_paths: # concat data
        df_i, _ = load_grid_search_data(config_paths=path)
        df_i = process_grid_search_data(df_i, d_set, i_set, r_set)
        data.append(df_i)
    df_train = pd.concat(data, ignore_index=True)

    # Note: make sure all factors represent main variations s.t. we get normal distributions within the groups
    df_metric = df_metric[df_metric['D_delay'] == 5] # metric for prediction
    df_train = df_train[df_train['D_delay'] == delay] # delay in temporal dataset task
    df_train = df_train[df_train['D_window_size'] == window_size] # adjust to increase difficulty and reduce accuracy == 100%
    df_train = df_train[df_train['I_connection'] != 'out-0:b:1/b'] # terrible performance, no point in including 
    factors = list(df_train[factors].nunique()[df_train[factors].nunique() > 1].index)

    df = aggregate_and_merge_data(df_metric, df_train, factors)
    df, factors = fix_combo(df, factors)
    groups_dict = {k: v[variable].values for k, v in df.groupby('combo')}
    return df, factors, groups_dict

def load_train_data_only(variable):
    training_paths = list()

    # training_paths.append('config/temporal/density/grid_search/homogeneous_deterministic.yaml')
    training_paths.append('/code/config/temporal/density/grid_search/test_optimizer_and_readout_mode.yaml')

    d_set = {}
    d_set = {'task', 'delay', 'window_size'}
    i_set = {'connection', 'perturbation'}
    r_set = {'mode', 'k_avg', 'init'}
    t_set = {'optim', 'criterion'}
    o_set = {'mode'}
    # t_set = {}
    # o_set = {}
    factors = sorted([f'D_{x}' for x in d_set] + [f'I_{x}' for x in i_set] + [f'R_{x}' for x in r_set] + [f'T_{x}' for x in t_set] + [f'O_{x}' for x in o_set])

    data = list()
    for path in training_paths: # concat data
        df_i, _ = load_grid_search_data(config_paths=path)
        df_i = process_grid_search_data(df_i, d_set, i_set, r_set, t_set, o_set)
        data.append(df_i)
    df_train = pd.concat(data, ignore_index=True)

    # Note: make sure all factors represent main variations s.t. we get normal distributions within the groups
    factors = list(df_train[factors].nunique()[df_train[factors].nunique() > 1].index)

    df = df_train 
    df['combo'] = df.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)

    df, factors = fix_combo(df, factors)
    groups_dict = {k: v[variable].values for k, v in df.groupby('combo')}
    return df, factors, groups_dict

def fix_combo(df, factors):
    scores = [8, 2, 1, 0, 7] # manually set new order of factors to make most important factor first (put k_avg last)
    scores = [10, 0, 5, 1] # TODO temp override
    factors = sorted(factors, key=lambda item: scores[factors.index(item)], reverse=True)
    df['combo'] = df['combo'].apply(lambda x: tuple(sorted(x, key=lambda item: scores[x.index(item)], reverse=True)))
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    df['combo_no_k_avg'] = df['combo'].apply(lambda row: row[:-1])
    df['combo_no_k_avg_str'] = df['combo_no_k_avg'].apply(lambda t: "_".join(map(str, t)))
    return df, factors

def process_grid_search_data(df, d_set, i_set, r_set, t_set=set(), o_set=set()):
    flatten_params = lambda x: pd.concat([
        pd.Series({f"D_{k}": v for k, v in x.D.model_dump().items() if k in d_set}),
        pd.Series({f"I_{k}": v for k, v in x.M.I.model_dump().items() if k in i_set}),
        pd.Series({f"R_{k}": v for k, v in x.M.R.model_dump().items() if k in r_set}),
        pd.Series({f"T_{k}": str(v) for k, v in x.M.T.model_dump().items() if k in t_set}),
        pd.Series({f"O_{k}": v for k, v in x.M.O.model_dump().items() if k in o_set}),
    ])
    df_flattened_params = df['params'].apply(lambda p: flatten_params(p))
    df = pd.concat([df, df_flattened_params], axis=1)

    df['grid_search'] = df['params'].apply(lambda p: p.L.out_path.name)
    if df.iloc[0]['params'].L.T.loss:
        df['loss'] = df['params'].apply(lambda p: p.L.T.loss)
    if df.iloc[0]['accuracy'].L.T.accuracy:
        df['accuracy'] = df['params'].apply(lambda p: p.L.T.accuracy)
    df.drop(['params'], axis=1, inplace=True)
    return df

def process_grid_search_data_kq_and_gr(df, d_set, i_set, r_set): # TODO come back to this and compare KQ and GR to training?
    # row matrix style on KQ/GR/Delta so we make new columns to unpack this
    df_pivoted = df.pivot_table(index=['config', 'sample'], columns='metric', values='value').reset_index()
    df = pd.merge(df[['config', 'sample', 'params', 'spectral_radius']].groupby(['config', 'sample']).first(), 
        df_pivoted, 
        on=['config', 'sample'], 
        how='left')
    df = process_grid_search_data(df, d_set, i_set, r_set)
    return df

def aggregate_and_merge_data(df1, df2, factors):
    # max_values = df1.groupby(['D_delay', 'sample'])['delta'].idxmax() # max delta per delay-sample (over many k_avg)
    # max_subset = df1.loc[max_values]
    # max_subset['k_avg*'] = max_subset['k_avg'].mean() # average delta* over the grid_search samples

    # Note that dataset for KQ and GR dataframe, aka df1, should not be merged with df2 with dataset-based design choices as the metric is "dataset-agnostic"
    df1 = df1.convert_dtypes()
    df2 = df2.convert_dtypes()
    df1['combo'] = df1.apply(lambda row: tuple(row[feature] for feature in factors if feature[0:] != 'D_'), axis=1)
    df1 = df1[['combo', 'kq', 'gr', 'delta', 'spectral_radius']]
    df1 = df1.groupby('combo', as_index=False).mean(numeric_only=True)
    df2['combo'] = df2.apply(lambda row: tuple(row[feature] for feature in factors if feature[0:] != 'D_'), axis=1)
    df = pd.merge(df1, df2, on=['combo'], how='inner')
    df.columns = [col[:-2] if col.endswith('_x') else col for col in df.columns]
    df.drop([col for col in df.columns if col.endswith('_y')], axis=1, inplace=True)

    df['combo'] = df.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)
    return df

def graph_accuracy_vs_k_avg(out_path: Path, df: pd.DataFrame, success_thresh):
    out_path.mkdir(parents=True, exist_ok=True)
    df['R_k_avg_w_jitter'] = df['R_k_avg'] + np.random.uniform(-0.5, 0.5, size=len(df))
    df['design'] = df['combo_no_k_avg_str']
    
    fig = px.scatter(
        df, 
        x='R_k_avg_w_jitter', 
        y='accuracy',
        color='design',
        opacity=0.7,
        labels={
            'R_k_avg_w_jitter': 'R_k_avg',
            'accuracy': 'Accuracy'
        }
    )
    fig.update_traces(marker=dict(size=5))
    
    x_min = int(df['R_k_avg'].min())
    x_max = int(df['R_k_avg'].max())
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(x_min, x_max + 1, 1))
    )
    # fig.update_layout(showlegend=False)
    fig.write_html(out_path / 'scatter_accuracy_vs_k_avg.html')
    pio.write_image(fig, out_path / 'scatter_accuracy_vs_k_avg.svg', format='svg', width=1200, height=1600)
    return fig

def binary_stats_analysis(out_path: Path, df: pd.DataFrame, response: str, success_thresh: float):
    """
    Perform binary success analysis with perfect separation handling.
    Separates predictors (properties that should predict success) from 
    design choices (controllable factors to optimize).
    """
    out_path.mkdir(exist_ok=True, parents=True)
    df = df[['combo', response, 'delta', 'kq', 'gr', 'grid_search'] + factors]
    
    # Convert to binary success
    df["success"] = (df["accuracy"] > success_thresh).astype(int)
    
    # Separate predictors vs design choices
    predictors = ['delta', 'kq', 'gr']  # Properties that should predict success
    design_factors = [f for f in factors if f not in predictors]  # Actual design choices
    
    available_predictors = [p for p in predictors if p in df.columns]
    available_design_factors = [f for f in design_factors if f in df.columns]
    
    print("=" * 80)
    print("BINARY SUCCESS ANALYSIS WITH PERFECT SEPARATION HANDLING")
    print("=" * 80)
    print(f"\nThreshold for success: accuracy > {success_thresh}")
    print(f"Total samples: {len(df)}")
    print(f"Success rate: {df['success'].mean():.2%}")
    print(f"\nPredictors (should predict success): {available_predictors}")
    print(f"Design factors (choices we control): {available_design_factors}")
    
    # Analyze perfect separation
    perfect_separation_analysis(df)
    
    # 2. Analyze design choices
    print("\n" + "=" * 80)
    print("DESIGN CHOICE ANALYSIS (Fisher's Exact Tests)")
    print("=" * 80)
    fisher_results = analyze_design_choices_fisher(df, available_design_factors)
    
    # 3. Combination analysis for perfect success/failure
    print("\n" + "=" * 80)
    print("COMBINATION ANALYSIS")
    print("=" * 80)
    combo_results = analyze_combinations(df)
    
    # 4. Generate visualizations
    create_visualizations(df, available_design_factors, fisher_results, out_path)
    
    # 5. Save detailed results
    save_results(df, fisher_results, combo_results, out_path)
    
    return df, fisher_results, combo_results


def perfect_separation_analysis(df: pd.DataFrame):
    """Analyze which factor combinations lead to perfect separation."""
    
    print("\n" + "-" * 60)
    print("Perfect Separation Analysis")
    print("-" * 60)
    
    # Group by combo and calculate success rates
    combo_stats = df.groupby('combo_str').agg({
        'success': ['mean', 'count', 'sum']
    }).round(3)
    combo_stats.columns = ['success_rate', 'n_samples', 'n_successes']
    
    # Identify perfect separation cases
    perfect_success = combo_stats[combo_stats['success_rate'] == 1.0]
    perfect_failure = combo_stats[combo_stats['success_rate'] == 0.0]
    
    print(f"\nCombinations with perfect success (100%): {len(perfect_success)}")
    print(f"Combinations with perfect failure (0%): {len(perfect_failure)}")
    print(f"Combinations with mixed results: {len(combo_stats) - len(perfect_success) - len(perfect_failure)}")
    
    if len(perfect_success) > 0:
        print("\nTop combinations with perfect success:")
        print(perfect_success.head())
    
    if len(perfect_failure) > 0:
        print("\nTop combinations with perfect failure:")
        print(perfect_failure.head())
    
    return combo_stats

def analyze_design_choices_fisher(df: pd.DataFrame, 
                                 design_factors: List[str]) -> Dict[str, Any]:
    """
    Analyze design choices using Fisher's exact test for direct effects.
    """
    results = {}
    
    for factor in design_factors:
        print(f"\n{factor}:")
        print("-" * 40)
        
        # Get unique levels
        levels = df[factor].unique()
        
        if len(levels) == 2:
            # Binary factor - direct Fisher's exact test
            ct = pd.crosstab(df[factor], df['success'])
            
            try:
                oddsratio, pvalue = fisher_exact(ct)
                
                # Calculate success rates for each level
                success_rates = df.groupby(factor)['success'].agg(['mean', 'count'])
                
                print(f"  Fisher's exact test:")
                print(f"    Odds ratio: {oddsratio:.3f}")
                print(f"    p-value: {pvalue:.4f}")
                print(f"\n  Success rates by level:")
                for level in levels:
                    rate = success_rates.loc[level, 'mean']
                    n = success_rates.loc[level, 'count']
                    print(f"    {level}: {rate:.2%} (n={n})")
                
                results[factor] = {
                    'test': 'fisher_exact',
                    'oddsratio': oddsratio,
                    'pvalue': pvalue,
                    'success_rates': success_rates.to_dict(),
                    'contingency_table': ct
                }
                
            except Exception as e:
                print(f"  Error in Fisher's exact test: {e}")
                results[factor] = {'error': str(e)}
                
        else:
            # Multi-level factor - pairwise Fisher's exact tests
            print(f"  Multi-level factor ({len(levels)} levels)")
            print(f"  Performing pairwise Fisher's exact tests...")
            
            pairwise_results = []
            success_rates = df.groupby(factor)['success'].agg(['mean', 'count'])
            
            for level1, level2 in combinations(levels, 2):
                df_pair = df[df[factor].isin([level1, level2])]
                ct = pd.crosstab(df_pair[factor], df_pair['success'])
                
                try:
                    oddsratio, pvalue = fisher_exact(ct)
                    pairwise_results.append({
                        'level1': level1,
                        'level2': level2,
                        'oddsratio': oddsratio,
                        'pvalue': pvalue
                    })
                except:
                    pass
            
            # Sort by p-value
            pairwise_results = sorted(pairwise_results, key=lambda x: x['pvalue'])
            
            print(f"\n  Success rates by level:")
            for level in levels:
                rate = success_rates.loc[level, 'mean']
                n = success_rates.loc[level, 'count']
                print(f"    {level}: {rate:.2%} (n={n})")
            
            print(f"\n  Significant pairwise comparisons:")
            for res in pairwise_results:
                if res['pvalue'] < 0.05:
                    print(f"    {res['level1']} vs {res['level2']}: "
                          f"OR={res['oddsratio']:.3f}, p={res['pvalue']:.4f}")
            
            results[factor] = {
                'test': 'pairwise_fisher',
                'n_levels': len(levels),
                'pairwise_results': pairwise_results,
                'success_rates': success_rates.to_dict()
            }
    
    return results

def analyze_combinations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze which combinations of factors lead to perfect or near-perfect outcomes.
    """
    
    # Analyze by combination
    combo_analysis = df.groupby('combo_str').agg({
        'success': ['mean', 'count', 'sum'],
        'accuracy': ['mean', 'std']
    })
    combo_analysis.columns = ['success_rate', 'n_samples', 'n_successes', 
                              'accuracy_mean', 'accuracy_std']
    
    # Sort by success rate
    combo_analysis = combo_analysis.sort_values(['success_rate', 'accuracy_mean'], ascending=False)
    
    print("\nTop 5 combinations by success rate:")
    print(combo_analysis.head().to_string())
    
    print("\nBottom 5 combinations by success rate:")
    print(combo_analysis.tail().to_string())
    
    # Identify key patterns in successful combinations
    perfect_combos = combo_analysis[combo_analysis['success_rate'] == 1.0]
    failed_combos = combo_analysis[combo_analysis['success_rate'] == 0.0]
    
    results = {
        'combo_stats': combo_analysis,
        'n_perfect': len(perfect_combos),
        'n_failed': len(failed_combos),
        'n_mixed': len(combo_analysis) - len(perfect_combos) - len(failed_combos)
    }
    
    return results


def create_visualizations(df: pd.DataFrame, design_factors: List[str],
                         fisher_results: Dict, out_path: Path):
    """Create visualizations for the analysis results."""
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
    # 2. Success rates by design factors
    ax = axes[0]
    design_success_rates = []
    design_labels = []
    
    for factor in design_factors[:5]:  # Limit to first 5 for readability
        if factor in fisher_results:
            rates = fisher_results[factor].get('success_rates', {})
            if 'mean' in rates:
                for level, rate_info in rates['mean'].items():
                    design_success_rates.append(rate_info)
                    design_labels.append(f"{factor}_{level}")
    
    if design_success_rates:
        ax.bar(range(len(design_success_rates)), design_success_rates)
        ax.set_xticks(range(len(design_labels)))
        ax.set_xticklabels(design_labels, rotation=45, ha='right')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rates by Design Choices')
        ax.set_ylim([0, 1])
    
    # 3. Distribution of success rates across combinations
    ax = axes[1]
    combo_success_rates = df.groupby('combo_str')['success'].mean()
    ax.hist(combo_success_rates, bins=20, edgecolor='black')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Number of Combinations')
    ax.set_title('Distribution of Success Rates Across Combinations')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Perfect Failure')
    ax.axvline(x=1, color='g', linestyle='--', alpha=0.5, label='Perfect Success')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path / 'binary_analysis_plots.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of success rates
    combination_list = list(combinations(design_factors, 2))
    (out_path / 'heatmaps').mkdir(exist_ok=True, parents=True)

    for factor1, factor2 in combination_list:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # Changed 'axes' to 'ax'
        pivot_table = df.pivot_table(values='success',
                                    index=factor1,
                                    columns=factor2,
                                    aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax)
        ax.set_title(f'Success Rate Heatmap: {factor1} vs {factor2}')
        plt.tight_layout()
        plt.savefig(out_path / 'heatmaps' / f'{factor1}x{factor2}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    print(f"\nVisualizations saved to {out_path / 'binary_analysis_plots.png'}")

    create_level_based_heatmap(out_path, df, design_factors)

def create_level_based_heatmap(out_path, df, design_factors, value_col='success'):
    """
    Create a heatmap where rows and columns are individual factor levels,
    grouped by their parent factors
    """

    # Get all unique levels for each factor
    factor_levels = {}
    for factor in design_factors:
        factor_levels[factor] = sorted(df[factor].unique())
    
    # Flatten all levels with their parent factor info
    all_levels = []
    level_to_factor = {}
    factor_groups = {}
    
    for factor, levels in factor_levels.items():
        factor_groups[factor] = []
        for level in levels:
            level_str = str(level)
            all_levels.append(level_str)
            level_to_factor[level_str] = factor
            factor_groups[factor].append(level_str)
    
    n_levels = len(all_levels)
    print(f"Creating {n_levels}x{n_levels} heatmap")
    
    # Initialize the matrix
    heatmap_matrix = np.zeros((n_levels, n_levels))
    
    # Fill the matrix
    for i, level1 in enumerate(all_levels):
        for j, level2 in enumerate(all_levels):
            factor1 = level_to_factor[level1]
            factor2 = level_to_factor[level2]
            
            if i == j:
                # Diagonal: success rate when this level is present
                mask = (df[factor1].astype(str) == level1)
                if mask.sum() > 0:
                    heatmap_matrix[i, j] = df[mask][value_col].mean()
                else:
                    heatmap_matrix[i, j] = np.nan
            else:
                if factor1 == factor2:
                    # Same factor, different levels: set to NaN or 0 (they're mutually exclusive)
                    heatmap_matrix[i, j] = np.nan
                else:
                    # Different factors: success rate when both levels are present
                    mask = (df[factor1].astype(str) == level1) & (df[factor2].astype(str) == level2)
                    if mask.sum() > 0:
                        heatmap_matrix[i, j] = df[mask][value_col].mean()
                    else:
                        heatmap_matrix[i, j] = np.nan
    
    # Create labels with factor prefixes for clarity
    labels = [f"{level_to_factor[level]}:{level}" for level in all_levels]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create mask for NaN values
    mask = np.isnan(heatmap_matrix)
    
    # Plot the heatmap
    sns.heatmap(heatmap_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='RdYlGn',
                center=0.5,
                vmin=0, 
                vmax=1,
                annot=True,  # Show values
                fmt='.2f',
                cbar_kws={'label': 'Success Rate'},
                mask=mask,
                ax=ax)
    
    # Add group separators
    cumulative_pos = 0
    for factor in design_factors:
        group_size = len(factor_groups[factor])
        if cumulative_pos > 0:  # Don't draw line at the very beginning
            ax.axhline(cumulative_pos, color='white', linewidth=3)
            ax.axvline(cumulative_pos, color='white', linewidth=3)
        cumulative_pos += group_size
    
    plt.title('Success Rate Heatmap: All Factor Levels', fontsize=16, pad=20)
    plt.xlabel('Factor Levels', fontsize=12)
    plt.ylabel('Factor Levels', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Add factor group labels
    cumulative_pos = 0
    for factor in design_factors:
        group_size = len(factor_groups[factor])
        mid_pos = cumulative_pos + group_size / 2
        
        # Add text labels for factor groups
        ax.text(mid_pos, -1, factor, ha='center', va='top', fontweight='bold', fontsize=12)
        ax.text(-1, mid_pos, factor, ha='right', va='center', fontweight='bold', fontsize=12, rotation=90)
        
        cumulative_pos += group_size
    
    plt.tight_layout()
    plt.savefig(f'{out_path}/massive_heatmap.png', dpi=150, bbox_inches='tight')
    return heatmap_matrix
    

def save_results(df: pd.DataFrame, fisher_results: Dict,
                combo_results: Dict, out_path: Path):
    """Save detailed results to files."""
    
    # Save summary statistics
    with open(out_path / 'analysis_summary.txt', 'w') as f:
        f.write("BINARY SUCCESS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("\n\nFISHER'S EXACT TEST RESULTS (Design Choices):\n")
        f.write("-" * 40 + "\n")
        for factor, results in fisher_results.items():
            f.write(f"\n{factor}:\n")
            if 'pvalue' in results:
                f.write(f"  p-value: {results['pvalue']:.4f}\n")
                f.write(f"  Odds ratio: {results.get('oddsratio', 'N/A')}\n")
            elif 'pairwise_results' in results:
                f.write(f"  {results['n_levels']} levels analyzed\n")
                sig_pairs = [p for p in results['pairwise_results'] if p['pvalue'] < 0.05]
                f.write(f"  {len(sig_pairs)} significant pairwise comparisons\n")
        
        f.write("\n\nCOMBINATION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Combinations with perfect success: {combo_results['n_perfect']}\n")
        f.write(f"Combinations with perfect failure: {combo_results['n_failed']}\n")
        f.write(f"Combinations with mixed results: {combo_results['n_mixed']}\n")
    
    # Save combination statistics to CSV
    combo_results['combo_stats'].to_csv(out_path / 'combination_stats.csv')
    
    print(f"\nResults saved to {out_path}")
    print(f"  - analysis_summary.txt")
    print(f"  - combination_stats.csv")

def mixed_effects_by_level(df, response, success_thresh, categorical_factors, numerical_factor, out_path):
    """
    Run mixed effects logistic regression separately for each level of numerical factor.
    Returns DataFrame with all results for easy analysis and comparison.
    """
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Create binary success variable
    df = df.copy()
    df["success"] = (df[response] > success_thresh).astype(int)
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    df['combo_id'], _ = pd.factorize(df['combo_str'])
    
    all_results = []
    
    # Loop through each numerical level
    for level in sorted(df[numerical_factor].unique()):
        print(f"\n{'='*50}")
        print(f"ANALYSIS FOR {numerical_factor} = {level}")
        print(f"{'='*50}")
        
        # Filter data for this level
        df_level = df[df[numerical_factor] == level].copy()
        
        print(f"Data: {len(df_level)} trials, {df_level['combo_str'].nunique()} combos")
        success_rate = df_level['success'].mean()
        print(f"Success rate: {success_rate:.3f}")
        
        # Skip levels with perfect separation
        if success_rate == 0 or success_rate == 1:
            print(f"⚠️  Skipping level {level}: perfect separation (success rate = {success_rate})")
            print("   Cannot estimate coefficients with 0% or 100% success rate")
            continue
        
        try:
            # Ensure categorical factors are proper dtypes
            for factor in categorical_factors:
                if factor in df_level.columns:
                    df_level[factor] = df_level[factor].astype('category')
            
            # Create formula with categorical factors
            formula_terms = [f'C({factor})' for factor in categorical_factors if factor in df_level.columns]
            formula = f"success ~ {' + '.join(formula_terms)}"
            print(f"Formula: {formula}")
            
            # Fit GEE model (accounts for clustering)
            model = smf.gee(formula, groups=df_level['combo_id'], 
                           data=df_level, family=Binomial()).fit()
            
            print(f"QIC: {model.qic()[0]:.2f}")
            
            # Get reference categories
            reference_categories = {}
            for factor in categorical_factors:
                if factor in df_level.columns:
                    reference_categories[factor] = sorted(df_level[factor].unique())[0]
            
            print("\nReference categories (baseline for comparisons):")
            for factor, ref_cat in reference_categories.items():
                print(f"  {factor}: {ref_cat}")
            
            # Add reference categories to results (OR = 1.0)
            for factor, ref_category in reference_categories.items():
                all_results.append({
                    'numerical_factor': level,
                    'categorical_factor': factor,
                    'level': ref_category,
                    'is_reference': True,
                    'success_rate': success_rate,
                    'n_samples': len(df_level),
                    'OR': 1.0,
                    'p_value': np.nan,
                    'coefficient': 0.0,
                    'std_err': np.nan,
                    'QIC': model.qic()[0]
                })
            
            # Process model results
            print("\nSignificant effects (p < 0.05):")
            
            sig_effects = model.pvalues[model.pvalues < 0.05]
            has_significant = False
            
            for param in model.params.index:
                if param == 'Intercept':
                    continue
                    
                coef = model.params[param]
                p_val = model.pvalues[param]
                std_err = model.bse[param]
                odds_ratio = np.exp(coef)
                is_significant = p_val < 0.05
                
                if is_significant:
                    has_significant = True
                
                # Extract factor and level from parameter name
                if '[T.' in param:
                    factor_part = param.split(']')[0].split('[T.')[0].replace('C(', '').replace(')', '')
                    level_part = param.split('[T.')[1].split(']')[0]
                    
                    all_results.append({
                        'numerical_factor': level,
                        'categorical_factor': factor_part,
                        'level': level_part,
                        'is_reference': False,
                        'success_rate': success_rate,
                        'n_samples': len(df_level),
                        'OR': odds_ratio,
                        'p_value': p_val,
                        'coefficient': coef,
                        'std_err': std_err,
                        'QIC': model.qic()[0]
                    })
                    
                    if is_significant:
                        print(f"  {factor_part} = '{level_part}' vs reference: OR={odds_ratio:.3f}, p={p_val:.3f}")
            
            if not has_significant:
                print("  None")
        
        except Exception as e:
            print(f"Model failed: {e}")
            warnings.warn(f"Analysis failed for {numerical_factor}={level}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Sort for better readability
        results_df = results_df.sort_values(['numerical_factor', 'categorical_factor', 'is_reference'], 
                                          ascending=[True, True, False]).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print("SUMMARY - ALL RESULTS")
        print(f"{'='*60}")
        print(f"Total rows: {len(results_df)}")
        print(f"Significant effects: {len(results_df[results_df['p_value'] < 0.05])}")
        
        # Save all results
        results_df.to_csv(out_path / 'mixed_effects_all_results.csv', index=False)
        print(f"All results saved to: {out_path / 'mixed_effects_all_results.csv'}")
        
        # Save significant results only
        sig_results_df = results_df[
            (results_df['p_value'] < 0.05) | results_df['is_reference']
        ].copy()
        
        sig_results_df.to_csv(out_path / 'mixed_effects_significant_results.csv', index=False)
        print(f"Significant results saved to: {out_path / 'mixed_effects_significant_results.csv'}")
        print(f"Significant results rows: {len(sig_results_df)}")
        
        return results_df
    else:
        print("No results to save - all levels had perfect separation")
        return pd.DataFrame()


if __name__ == '__main__':
    out_path = Path('/out/temporal/stats/design_evaluation')
    response = 'accuracy'

    # # statistical evauluation
    # ####################################
    # for i in [1, 3, 5]:
    #     for j in [1, 3, 5]:

    #         success_thresh = 0.9
    #         path = out_path / 'density' / f'delay-{i}_window-{j}'
    #         print(path)
    #         print('#'*60)
    #         df, factors, groups_dict = load_custom_data(response, '0110', i, j)
    #         # polar_design_plot(path, df, factors, success_thresh, f'task:{path.parent.name}, delay:{i}, window:{j}')
    #         graph_accuracy_vs_k_avg(path, df, success_thresh)
    #         # df, fisher_results, combo_results = binary_stats_analysis(path, df, response, success_thresh)
    #         # results = mixed_effects_by_level(df=df, response='accuracy', success_thresh=success_thresh, out_path=path / 'mixed_effects_results', ascending=[1, 1, 0, 1, 1])

    #         success_thresh = 0.6
    #         path = out_path / 'parity' / f'delay-{i}_window-{j}'
    #         print(path)
    #         print('#'*60)
    #         df, factors, groups_dict = load_custom_data(response, '0101', i, j)
    #         # polar_design_plot(path, df, factors, success_thresh, f'task:{path.parent.name}, delay:{i}, window:{j}')
    #         graph_accuracy_vs_k_avg(path, df, success_thresh)
    #         # df, fisher_results, combo_results = binary_stats_analysis(path, df, response, success_thresh)
    #         # results = mixed_effects_by_level(df=df, response='accuracy', success_thresh=success_thresh, out_path=path / 'mixed_effects_results', ascending=[1, 1, 0, 1, 1])

    success_thresh = 0.9
    i = 3
    j = 5
    path = out_path / 'density' / 'test_optimizer_and_readout_mode' / f'delay-{i}_window-{j}'
    df, factors, groups_dict = load_train_data_only(response)
    graph_accuracy_vs_k_avg(path, df, success_thresh)