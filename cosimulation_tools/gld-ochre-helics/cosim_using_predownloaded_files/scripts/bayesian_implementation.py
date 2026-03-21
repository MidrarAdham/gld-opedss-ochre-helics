'''
Author: MidrarAdham
Created: Fri Mar 20 2026
'''
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# My scripts:
import figures_for_bayesian_implementation as figures


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
    # df['state'] = (df['tn_meter_4br_46:measured_real_power'] ==4500).astype(int)
    df['state'] = (df[df.columns[1]] >= threshold).astype(int)
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

    # kw_mean  = mean_matrix.multiply(feeder_demand, axis=0)
    # kw_lower = ci_lower_matrix.multiply(feeder_demand, axis=0)
    # kw_upper = ci_upper_matrix.multiply(feeder_demand, axis=0)


    return kw_mean, kw_lower, kw_upper

def prepare_transformer_data (transformer_file_dir : str):
    df = pd.read_csv (f'{transformer_file_dir}residential_transformer.csv', skiprows=8)
    # df = df.head (1440)
    df = df.iloc [1440:2880]
    df = cleanup_results_files (df=df, col = 'power_out')
    df = df.drop ('power_in', axis=1)
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

def try_different_day ():
    pass

if __name__ == "__main__":

    cosim_results_dir = '../results/wh_cosim/'
    cosim_results_files = [f for f in os.listdir (cosim_results_dir) if 'ochre' in f]
    # I'll change those in a bit
    wh_threshold = 5000.0
    hvac_threshold = 100.0
    # Init log dict
    all_histories = {}
    cosim_results_df = {}
    # loop through the GLD results from the cosim to build the log dict
    for filename in cosim_results_files:
        df_month = pd.read_csv (cosim_results_dir+filename, skiprows=8)
        # df = df_month.head (1440)
        df = df_month.iloc [1440:2880]
        df = cleanup_results_files (df=df, col='constant_power_12')
        # hvac_threshold = round(pd.to_numeric(df['constant_power_12']).max (), 2) - 3000
        df = create_binary_states (df=df, threshold=wh_threshold)
        all_histories[filename] = bayesian_implementation (df=df)
        cosim_results_df [filename] = df

    df = pd.DataFrame (all_histories)
    df = df.transpose ()
    # Transformer information:
    xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_results_dir)
    # Get the bayesian results
    kw_mean, kw_lower, kw_upper = create_matrices_from_bayesian_results (df=df, xfmr_df=xfmr_demand)
    # prep the tiome col for the plots
    time_col = pd.to_datetime(xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')

    sns.set_style ('whitegrid')

    for key, value in cosim_results_df.items():
        # ====================================
        # xfmr_demand is the transformer data
        # df is the single DER demand profile
        # ====================================

        stats = {}

        df = df.iloc [1440:2880]

        df = value.drop('state', axis=1)
        df = df.set_index('# timestamp')
        df.index = pd.to_datetime(df.index)
        df = df.resample('10min').mean()
        df = df.reset_index ()

        time_col = pd.to_datetime(xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')
        y_true = pd.to_numeric(df['constant_power_12'], errors='coerce').to_numpy() / 1e3
        y_mean = pd.to_numeric(kw_mean[key], errors='coerce').to_numpy() / 1e3
        y_low  = pd.to_numeric(kw_lower[key], errors='coerce').to_numpy() / 1e3
        y_up   = pd.to_numeric(kw_upper[key], errors='coerce').to_numpy() / 1e3
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
        figures.predicted_kw_each_der (df=df, stats=stats, filename=key)