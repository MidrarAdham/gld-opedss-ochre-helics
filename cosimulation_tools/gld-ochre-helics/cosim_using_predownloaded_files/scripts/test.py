'''
Author: Midrar Adham
Created: Sat Mar 21 2026
'''
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def initialize_history ():
    return {
        'std' : [],
        'beta' : [],
        'mean' : [],
        # 'prior' : [],
        'chunk' : [],
        'alpha' : [],
        'H_chunk' : [],
        'T_chunk' : [],
        'ci_upper' : [],
        'ci_lower' : [],
        # 'evidence' : [],
        'posterior' : [],
        # 'likelihood' : [],
    }

def create_binary_states (df : pd.DataFrame, threshold : float) -> pd.DataFrame:
    """
    convert power values to binary states (ON = 1, OFF=0)
    
    :param df: a one-week water heater load profile
    :type df: pd.DataFrame
    :return: df with added "state" column
    :rtype: DataFrame
    """
    df = df.copy ()
    df['state'] = (df[df.columns[1]] > threshold).astype('bool')
    return df

def cleanup_results_files (df : pd.DataFrame, col : str):
    df.loc[:, '# timestamp'] = df['# timestamp'].apply (lambda x: x.strip ('PST'))
    df.loc[:, '# timestamp'] = pd.to_datetime (df['# timestamp'])
    df.loc[:, col] = df[col].apply (lambda x: complex (x))
    df.loc[:, col] = df[col].apply(lambda x: x.real)
    return df

def calculate_stats (alpha : int, beta_param : int, ci_lower_thresh : float = 0.025, ci_upper_thresh : float = 0.9745) -> dict:
    """
    calculate_stats for prior and posterior.
    
    :param alpha: the first variable for the Beta function
    :type alpha: int
    :param beta_param: The second variable for the Beta function
    :type beta_param: int
    :return: mean, variance, std_dev, lower, upper, and width of the CI.
    :rtype: dict
    """
    # for a beta function, the mean can be calculated as follows:
    mean = (alpha) / (alpha + beta_param)
    # for a beta function, the variance can be calculated as follows:
    variance = (alpha * beta_param) / (((alpha + beta_param)**2) * (alpha+beta_param+1))
    
    std = np.sqrt (variance)
    
    # 95% confidence interval:
    ci_lower = beta.ppf (ci_lower_thresh, alpha, beta_param)
    ci_upper = beta.ppf (ci_upper_thresh, alpha, beta_param)
    ci_width = ci_upper - ci_lower

    # theta stat calculations
    return {
        'mean' : mean,
        'std' : std,
        'ci_lower' : ci_lower,
        'ci_upper' : ci_upper,
        'ci_width' : ci_width,
        'variance' : variance
    }

def bayesian_implementation (df : pd.DataFrame):
    # initialization of alpha and beta in a beta function
    initial_alpha = 1
    initial_beta = 1
    # generate theta values
    theta_values = np.linspace (0.001, 0.999, 1000)
    # Get the history to record the results:
    history = initialize_history ()
    # define loop parameters:
    window_size, num_chunks = 10, 144
    discount = 0.3
    for chunk_idx in range(num_chunks):
        start_index = chunk_idx * window_size
        # calculates heads and tails; the sum of ON and OFF states of DERs
        df_sliced = df.iloc [start_index: start_index+window_size]
        H = (df_sliced['state'] == 1).sum()
        T = (df_sliced['state'] == 0).sum()
        # posterior conjugate parameters:
        alpha_posterior = discount * initial_alpha + H
        beta_posterior = discount * initial_beta + T
        # Calculate statisitical parameters:
        stats = calculate_stats (
            alpha=alpha_posterior,
            beta_param=beta_posterior,
            ci_lower_thresh=0.05,
            ci_upper_thresh=0.95
            )
        
        # Calculate the posterior:
        posterior = beta.pdf (theta_values, alpha_posterior, beta_posterior)
        
        # Record history:
        history['chunk'].append(chunk_idx)
        history['H_chunk'].append(H)
        history['T_chunk'].append(T)
        history['std'].append(stats['std'])
        history['mean'].append(stats['mean'])
        history['posterior'].append(posterior)
        history['beta'].append(beta_posterior)
        history['alpha'].append(alpha_posterior)
        history['ci_lower'].append(stats['ci_lower'])
        history['ci_upper'].append(stats['ci_upper'])

        initial_alpha = alpha_posterior
        initial_beta = beta_posterior

    return history

def create_matrices_from_bayesian_results (df : pd.DataFrame, xfmr_df : pd.DataFrame):
    mean_matrix = pd.DataFrame (
        df['mean'].tolist (),
        index=df.index
    ).T

    ci_lower_matrix = pd.DataFrame (
        df['ci_lower'].tolist (),
        index=df.index
    ).T

    ci_upper_matrix = pd.DataFrame (
        df['ci_upper'].tolist (),
        index = df.index
    ).T

    feeder_demand = xfmr_df ['power_out'].values

    total = mean_matrix.sum(axis=1)  # shared denominator

    kw_mean  = mean_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_lower = ci_lower_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_upper = ci_upper_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    bayesian_matrices = {'kw_mean' : kw_mean,
                         'kw_lower' : kw_lower,
                         'kw_upper' : kw_upper
                         }

    return bayesian_matrices

def prepare_transformer_data (transformer_file_dir : str):
    df = pd.read_csv (f'{transformer_file_dir}/residential_transformer.csv', skiprows=8)
    # df = df.head (1440)
    df = df.iloc [1440:2880]
    df = cleanup_results_files (df=df, col = 'power_out')
    df = df.drop ('power_in', axis=1)
    df['# timestamp'] = pd.to_datetime(df['# timestamp'])
    df = df.set_index ('# timestamp')
    df = df.resample ("10min").mean()
    df = df.reset_index ()

    return df
    
def quantifying_error_metrics (y_true, y_pred):
    mae = round(mean_absolute_error (y_true=y_true, y_pred=y_pred), 2)
    rmse = round(np.sqrt (mean_squared_error (y_true=y_true, y_pred=y_pred)), 2)
    R_squared = round(r2_score (y_true=y_true, y_pred=y_pred), 2)
    mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    return mae, rmse, R_squared, mape, nrmse

def build_der_state_matrix (cosim_results_df : pd.DataFrame):
    
    state_matrix = {}
    for filename, df in cosim_results_df.items():
        df['# timestamp'] = pd.to_datetime (df['# timestamp'])
        states = df.set_index('# timestamp')['state']
        # states.index = pd.to_datetime(states.index)
        states = states.resample('10min').max()  # if any minute in chunk is ON, chunk is ON
        state_matrix[filename] = states.values

    state_matrix = pd.DataFrame(state_matrix)  # shape (144, 8)

    return state_matrix

def get_statistical_metrics_for_der_profiles (dfs : dict, bayesian_matrices : dict):

    for key, value in dfs.items():
        # ====================================
        # xfmr_demand is the transformer data
        # df is the single DER demand profile
        # ====================================

        stats = {}

        df = value.drop('state', axis=1)
        df = df.infer_objects ().set_index('# timestamp')
        df.index = pd.to_datetime(df.index)
        df = df.resample('10min').mean()
        df = df.reset_index ()

        y_true = pd.to_numeric(df['constant_power_12'], errors='coerce').to_numpy() / 1e3
        y_mean = pd.to_numeric(bayesian_matrices['kw_mean'][key], errors='coerce').to_numpy() / 1e3
        y_low  = pd.to_numeric(bayesian_matrices['kw_lower'][key], errors='coerce').to_numpy() / 1e3
        y_up   = pd.to_numeric(bayesian_matrices['kw_upper'][key], errors='coerce').to_numpy() / 1e3
        y_pred = y_mean

        mae, rmse, R_squared, mape, nrmse = quantifying_error_metrics (
            y_true=y_true, y_pred=y_pred
            )
        stats = {'y_true':y_true,
                 'y_mean':y_mean,
                 'y_low': y_low,
                 'y_up': y_up,
                 'y_pred': y_pred,
                 'mae': mae,
                 'rmse': rmse,
                 'R_squared': R_squared,
                 'mape': mape,
                 'nrmse': nrmse
                 }
        
        return stats
def get_wh_demand_from_feeder_demand (state_matrix : pd.DataFrame):
    pass

if __name__ == "__main__":

    cosim_results_dir = '../results/'
    cosim_total_house_xfmr_demand = f'{cosim_results_dir}total_house_consumption'
    cosim_wh_only_demand = f"{cosim_results_dir}wh_cosim/"

    cosim_results_files = [fname for fname in os.listdir (cosim_wh_only_demand) if 'ochre' in fname]
    # I'll change those in a bit
    wh_threshold = 5000.0
    hvac_threshold = 5000.0
    # Init log dict
    all_histories = {}
    cosim_results_df = {}
    # loop through the GLD results from the cosim to build the log dict
    for filename in cosim_results_files:
        df_month = pd.read_csv (cosim_results_dir+'wh_cosim/'+filename, skiprows=8)
        # df = df_month.head (1440)
        df = df_month.iloc [1440:2880]
        df = cleanup_results_files (df=df, col='constant_power_12')
        df = create_binary_states (df=df, threshold=wh_threshold)
        all_histories[filename] = bayesian_implementation (df=df)
        cosim_results_df [filename] = df

    df = pd.DataFrame (all_histories)
    df = df.transpose ()
    # Transformer information:
    full_house_xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_total_house_xfmr_demand)
    wh_only_xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_wh_only_demand)
    
    # prep the tiome col for the plots
    time_col = pd.to_datetime(full_house_xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')
    # Get the bayesian results
    bayesian_matrices = create_matrices_from_bayesian_results (df=df, xfmr_df=wh_only_xfmr_demand)

    resulting_stats = get_statistical_metrics_for_der_profiles (
        dfs=cosim_results_df,
        bayesian_matrices=bayesian_matrices
        )

    sns.set_style ('whitegrid')

    state_matrix = build_der_state_matrix (cosim_results_df= cosim_results_df)
    
    all_off_chunks = (state_matrix.sum(axis=1) == 0)
    
    all_off_times = time_col[all_off_chunks.values]

    background = full_house_xfmr_demand ['power_out'].copy()
    # We are keeoing only the OFF WHs states of the feeder demand:
    background [~all_off_chunks.values] = 0
    background_demand = background.infer_objects (copy=False).interpolate (method = 'linear')
    # background_demand = background.infer_objects (copy=False).rolling (
    #     window=12,
    #     center=True,
    #     min_periods=1
    # ).min()
    subtracted_wh_only_demand = full_house_xfmr_demand ['power_out'].values - background_demand.values
    subtracted_wh_only_demand = np.clip (subtracted_wh_only_demand, 0, None)
    
    fig, ax = plt.subplots (ncols=1, nrows=1, figsize=(16,6))
    
    # ax[0].plot (time_col, pd.to_numeric(background_demand)/1e3,
    #          label = 'feeder demand without WHs', alpha=0.5, linestyle='--', color='tab:red')
    
    # ax[0].plot (time_col, pd.to_numeric(full_house_xfmr_demand['power_out'])/1e3,
    #             label = 'Feeder Demand including WHs', color='tab:blue')
    
    # ax[0].plot (time_col, pd.to_numeric(subtracted_wh_only_demand)/1e3,
    #             label = 'WH demand portion of the feeder', color='black')
    # ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    
    bayesian_matrices['kw_mean']['total'] = bayesian_matrices['kw_mean'].sum (axis=1)
    bayesian_matrices['kw_mean']['total'] = pd.to_numeric (bayesian_matrices['kw_mean']['total'])/1e3
    full_house_xfmr_demand ['power_out'] = pd.to_numeric (full_house_xfmr_demand ['power_out'])/1e3

    # ax[1].plot (time_col, round(bayesian_matrices['kw_mean']['total'], 2),
    #          label = 'Diversified WH Predicted Demand [kW]', color='tab:blue')
    
    # ax[1].plot (time_col, round(full_house_xfmr_demand['power_out'], 2),
    #          label = 'Ground Truth - FH Feeder Demand [kW]', color='black', alpha=0.5)
    
    # ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    # # ax[1].legend ()

    # ax[2].plot (time_col, round(bayesian_matrices['kw_mean']['total'], 2),
    #          label = 'Diversified WH Predicted Demand [kW]', color='tab:blue')
    
    # ax[2].plot (time_col, pd.to_numeric(subtracted_wh_only_demand)/1e3,
    #             label = 'Ground Truth - WH Only Feeder Demand [kW]', color='black')

    # bayesian_matrices['kw_lower']['total'] = bayesian_matrices['kw_lower'].sum (axis=1)
    # bayesian_matrices['kw_upper']['total'] = bayesian_matrices['kw_upper'].sum (axis=1)
    
    # y_low  = pd.to_numeric(bayesian_matrices['kw_lower']['total'], errors='coerce').to_numpy() / 1e3
    # y_up   = pd.to_numeric(bayesian_matrices['kw_upper']['total'], errors='coerce').to_numpy() / 1e3
    # ax[2].fill_between (
    #     time_col, y_low, y_up,
    #     color = 'tab:blue',
    #     alpha=0.3, linewidth=0, label='95% CI'
    # )
    # ax[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    subtracted_wh_only_demand = [round(i/1e3, 2) for i in subtracted_wh_only_demand]
    wh_only_xfmr_demand['power_out'] = pd.to_numeric(wh_only_xfmr_demand['power_out'])/1e3
    ax.plot (time_col, subtracted_wh_only_demand,
             label = 'Subtracted Background from FH Xfmr Demand [kW]', color='tab:blue')
    
    ax.plot (time_col, round(wh_only_xfmr_demand['power_out'], 2),
             label = 'Ground Truth - Xfmr Demand : WHs Only [kW]', color='black', alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    ax.legend ()
    ax.set_xlabel('Time [HH:MM]', weight='bold')
    ax.set_ylabel('Feeder / WH Demand [kW]', weight='bold')
    # fig.supxlabel('Time [HH:MM]', weight='bold')
    # fig.supylabel('Feeder / WH Demand [kW]', weight='bold')

    plt.savefig ('./test.png')
    # plt.show()
    # print (~all_off_chunks.values)
    # print (all_off_chunks.values)