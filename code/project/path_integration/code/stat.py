from pathlib import Path
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.graph import calc_spectral_radius
import pandas as pd
from scipy.stats import f_oneway, levene, shapiro, kruskal
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols

def init_get_data_and_make_groups(in_path, out_path):
    out_path = Path(out_path)
    out_path = out_path / 'stats' 
    out_path.mkdir(parents=True, exist_ok=True)

    P = load_yaml_config(in_path)
    L = P.L

    file_path = L.out_path / 'log.h5'
    df = pd.read_hdf(file_path, key='df', mode='r')
    df['loss'] = df['loss'].apply(lambda x: x ** .5) # MSE to RMS
    col = 'loss' # TODO do this for parameter fields?
    df[col] = (df[col]-df[col].mean())/(df[col].std())
    df['model_params'] = df['params'].apply(lambda p: p.model)
    # input layer
    df['interleaving'] = df['model_params'].apply(lambda p: p.R.k_avg)
    # reservoir layer
    df['n_nodes'] = df['model_params'].apply(lambda p: p.R.n_nodes)
    df['k_avg'] = df['model_params'].apply(lambda p: p.R.k_avg)
    df['self_loops'] = df['model_params'].apply(lambda p: p.R.self_loops)
    df['init'] = df['model_params'].apply(lambda p: p.R.init)
    # extra
    # df['paths'] = df['params'].apply(lambda p: BooleanReservoir.make_load_paths(P.L.last_checkpoint))
    # df['spectral_radius'] = df['paths'].apply(lambda d: calc_spectral_radius(BooleanReservoir.load_graph(d['graph']))) # takes like 2hrs 5s/it

    vars = ['loss', 'interleaving', 'k_avg', 'self_loops', 'init']
    df = df[vars]

    df_encoded = pd.get_dummies(df, columns=df.columns[1:])
    df_melted = pd.melt(df_encoded, id_vars='loss', var_name='parameter', value_name='value')
    df_melted = df_melted[df_melted['value'] == True]
    df_melted.drop('value', axis=1, inplace=True)
    df_groups = df_melted.groupby('parameter')['loss'].apply(list)
    return df, df_melted, df_groups, vars

def perform_anova(df, metric):
    """Perform one-way ANOVA on the given metric."""
    model = ols(f'{metric} ~ C(config)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def anova_analysis(in_path, out_path):
    # compare between loss groups
    ####################################
    # samples: Indepenent
    # variables/factors: Independent & categorical, continuous variables follow normal distribution
    # one-way, two-way, n-way ANOVA: analysis of variance after changing one or more variables. Two-way consideres interaction effects.
    # Homogeneity of Variances: The groups should have approximately equal variances → Levene's Test
    # 1. normality: The dependent variable (loss) normally distributed within each group. → Shapiro-Wilk test or by examining Q-Q plots.
    #    normality not met: non-parametric tests like Kruskal-Wallis.

    # 1. Normality
    


    df, df_melted, df_groups, vars = init_get_data_and_make_groups(in_path, out_path)

    # samples sizes per category
    print('\n'.join([str((k, len(df_groups[k]))) for k in df_groups.index]))

    # # H0: is there a difference between the groups (parameter configurations)
    # anova = f_oneway(*df_groups) 
    # print(anova) 

    # how do the groups contribute (how do parameters affect loss)
    # Performing Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=df_melted['loss'], groups=df_melted['parameter'], alpha=0.05)
    print(tukey)

    # model_loss = ols('value ~ config', data=df_groups[df_groups['metric'] == 'loss']).fit()
    # table_loss = sm.stats.anova_lm(model_loss, typ=2)
    # print(table_loss)

    # # should this group be per combination or presence of a factor?
    # h_statistic, p_value = kruskal(*df_groups)
    # print("H-statistic:", h_statistic)
    # print("p-value:", p_value)


    # 


if __name__ == '__main__':
    # anova_analysis('/out/path_integration/grid_search/2D/initial_sweep/parameters.yaml', '/tmp')
    anova_analysis('/out/path_integration/grid_search/2D/initial_sweep/parameters.yaml', '/tmp')