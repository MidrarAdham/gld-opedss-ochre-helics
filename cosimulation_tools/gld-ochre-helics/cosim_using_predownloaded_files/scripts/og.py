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
from scipy.special import softmax
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
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
    # time_col = pd.to_datetime(xfmr_df['# timestamp']).dt.strftime ('%H:%M')

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

    # print(mean_matrix.div(total, axis=0))
    # # print(len(mean_matrix))
    # print(mean_matrix.div(total, axis=0).iloc[:,0])
    # # print(total)
    # quit()
    
    kw_mean  = mean_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_lower = ci_lower_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_upper = ci_upper_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    bayesian_matrices = {'kw_mean' : kw_mean,
                         'kw_lower' : kw_lower,
                         'kw_upper' : kw_upper
                         }

    return bayesian_matrices, mean_matrix # The mean matrix is returned for debugging

def prepare_transformer_data (transformer_file_dir : str, sampling_method : str):
    df = pd.read_csv (f'{transformer_file_dir}/residential_transformer.csv', skiprows=8)
    # df = df.head (1440)
    df = df.iloc [1440:2880]
    df = cleanup_results_files (df=df, col = 'power_out')
    df = df.drop ('power_in', axis=1)
    df['# timestamp'] = pd.to_datetime(df['# timestamp'])
    df = df.set_index ('# timestamp')
    if sampling_method == 'mean':
        df = df.resample ("10min").mean()
    else:
        df = df.resample ("10min").max()
    df = df.reset_index ()
    df['power_out'] = pd.to_numeric(df['power_out'], errors='coerce')

    return df
    
def quantifying_error_metrics (y_true, y_pred):
    mae = round(mean_absolute_error (y_true=y_true, y_pred=y_pred), 2)
    rmse = round(np.sqrt (mean_squared_error (y_true=y_true, y_pred=y_pred)), 2)
    R_squared = round(r2_score (y_true=y_true, y_pred=y_pred), 2)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    return mae, rmse, R_squared, nrmse

def build_der_state_matrix (cosim_results_df : pd.DataFrame):
    '''
    This is just resampling the bayesian results into 10-minutes intervals.
    '''
    
    state_matrix = {}
    for filename, df in cosim_results_df.items():
        df['# timestamp'] = pd.to_datetime (df['# timestamp'])
        states = df.set_index('# timestamp')['state']
        states.index = pd.to_datetime(states.index)
        states = states.resample('10min').mean()  # if any minute in chunk is ON, chunk is ON
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

        mae, rmse, R_squared, nrmse = quantifying_error_metrics (
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
                 'nrmse': nrmse
                 }
        
        return stats
def get_wh_demand_from_feeder_demand (state_matrix : pd.DataFrame):
    pass
def bayesian_cleaning_and_binary_states (
        cosim_results_dir : str,
        cosim_results_files : list,
        threshold : float,
        dir : str
        ):
    # Init log dict
    all_histories = {}
    cosim_results_df = {}

    for filename in cosim_results_files:
        df_month = pd.read_csv (cosim_results_dir+dir+filename, skiprows=8)
        # df = df_month.head (1440)
        df = df_month.iloc [1440:2880]
        df = cleanup_results_files (df=df, col='constant_power_12')
        df = create_binary_states (df=df, threshold=threshold)
        all_histories[filename] = bayesian_implementation (df=df)
        cosim_results_df [filename] = df
    
    return cosim_results_df, all_histories
if __name__ == "__main__":

    cosim_results_dir = '../results/'
    cosim_total_house_xfmr_demand = f'{cosim_results_dir}total_house_consumption'
    cosim_wh_only_demand = f"{cosim_results_dir}wh_cosim/"
    cosim_hvac_sim_dir = f"{cosim_results_dir}hvac_cosim/"
    wh_threshold = 5000.0
    hvac_threshold = 100.0
    combined_threshold = 0

    wh_cosim_results_files = [fname for fname in os.listdir (cosim_wh_only_demand) if 'ochre' in fname]
    hvac_cosim_results_files = [fname for fname in os.listdir (cosim_hvac_sim_dir) if 'ochre' in fname]
    
    wh_cosim_results_df, all_histories = bayesian_cleaning_and_binary_states (
        cosim_results_dir=cosim_results_dir,
        cosim_results_files=wh_cosim_results_files,
        threshold=wh_threshold,
        dir=cosim_wh_only_demand
    )
    # The wh_df includes all metrics that we got from the bayesian calculations (See history inside bayesian calculation function)
    # However, we're only using the CI intervals and the mean. We're not using anything else.
    wh_df = pd.DataFrame (all_histories)
    wh_df = wh_df.transpose ()
    
    hvac_cosim_results_df, all_histories = bayesian_cleaning_and_binary_states (
        cosim_results_dir=cosim_results_dir,
        cosim_results_files=hvac_cosim_results_files,
        threshold=hvac_threshold,
        dir=cosim_hvac_sim_dir
    )
    hvac_df = pd.DataFrame (all_histories)
    hvac_df = hvac_df.transpose ()

    # Transformer information:
    full_house_xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_total_house_xfmr_demand, sampling_method='mean')
    wh_only_xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_wh_only_demand,sampling_method='mean')
    hvac_only_xfmr_demand = prepare_transformer_data (transformer_file_dir=cosim_hvac_sim_dir,sampling_method='mean')
    
    combined_der_demand = wh_only_xfmr_demand.copy()
    combined_der_demand['power_out'] = (
        pd.to_numeric(wh_only_xfmr_demand['power_out'], errors='coerce') +
        pd.to_numeric(hvac_only_xfmr_demand['power_out'], errors='coerce')
        )

    # prep the tiome col for the plots
    time_col = pd.to_datetime(full_house_xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')
    
    hvac_df.index = ['hvac_' + idx for idx in hvac_df.index]
    wh_df.index = ['wh_' + idx for idx in wh_df.index]

    # Get the bayesian results
    combined_df = pd.concat ([wh_df, hvac_df])

    wh_bayesian_matrices, wh_mean_matrix = create_matrices_from_bayesian_results (df=wh_df, xfmr_df=full_house_xfmr_demand)
    hvac_bayesian_matrices, hvac_mean_matrix = create_matrices_from_bayesian_results (df=hvac_df, xfmr_df=full_house_xfmr_demand)
    print(hvac_mean_matrix)
    quit()
    combined_bayesian_matrices, combined_mean_matrix = create_matrices_from_bayesian_results (df=combined_df, xfmr_df=full_house_xfmr_demand)
    # hvac_bayesian_matrices = create_matrices_from_bayesian_results (df=hvac_df, xfmr_df=hvac_only_xfmr_demand)

    wh_cols = [c for c in wh_bayesian_matrices['kw_mean'].columns if c.startswith('wh_')]
    hvac_cols = [c for c in hvac_bayesian_matrices['kw_mean'].columns if c.startswith('hvac_')]


    wh_predicted_total = wh_bayesian_matrices['kw_mean'][wh_cols].sum(axis=1)
    hvac_predicted_total = hvac_bayesian_matrices['kw_mean'][hvac_cols].sum(axis=1)

    wh_ground_truth = pd.to_numeric(wh_only_xfmr_demand['power_out'], errors='coerce')
    hvac_ground_truth = pd.to_numeric(hvac_only_xfmr_demand['power_out'], errors='coerce')
    
    wh_share = combined_mean_matrix[wh_cols].sum(axis=1) / combined_mean_matrix.sum(axis=1)
    hvac_share = combined_mean_matrix[hvac_cols].sum(axis=1) / combined_mean_matrix.sum(axis=1)

    # Implementing the OLS:
    x_wh = combined_mean_matrix [wh_cols].sum (axis=1).values
    x_hvac = combined_mean_matrix [hvac_cols].sum (axis=1).values
    A = np.column_stack ([x_wh, x_hvac])
    b = pd.to_numeric (combined_der_demand['power_out'], errors='coerce').values
    kw_estimate, _, _, _ = np.linalg.lstsq (A, b, rcond=None)
    kw_wh, kw_hvac = kw_estimate
    print(f"Estimated kW per WH: {kw_wh:.1f} W")
    print(f"Estimated kW per HVAC: {kw_hvac:.1f} W")
    wh_predicted = kw_wh * x_wh
    hvac_predicted = kw_hvac * x_hvac

    print(f"WH predicted mean: {wh_predicted.mean():.1f} W")
    print(f"WH ground truth mean: {wh_only_xfmr_demand['power_out'].mean():.1f} W")
    print(f"HVAC predicted mean: {hvac_predicted.mean():.1f} W")
    print(f"HVAC ground truth mean: {hvac_only_xfmr_demand['power_out'].mean():.1f} W")
    
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax[0].plot(time_col, wh_predicted/1e3, label='WH predicted', color='tab:blue')
    ax[0].plot(time_col, pd.to_numeric(wh_only_xfmr_demand['power_out'])/1e3, 
            label='WH ground truth', color='black', alpha=0.7)
    ax[0].set_ylabel('Demand [kW]')
    ax[0].legend(frameon=False)
    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(20))

    ax[1].plot(time_col, hvac_predicted/1e3, label='HVAC predicted', color='tab:red')
    ax[1].plot(time_col, pd.to_numeric(hvac_only_xfmr_demand['power_out'])/1e3,
            label='HVAC ground truth', color='black', alpha=0.7)
    ax[1].set_ylabel('Demand [kW]')
    ax[1].legend(frameon=False)
    ax[1].xaxis.set_major_locator(ticker.MaxNLocator(20))

    plt.tight_layout()
    plt.show()
    # plt.savefig('./test.png')

    quit()
    # print(wh_share.head(10))
    # print(hvac_share.head(10))
    # quit()
    # fig, ax = plt.subplots(figsize=(16, 6))

    # ax.plot(time_col, wh_predicted_total/1e3, label='WH predicted', color='tab:blue')
    # ax.plot(time_col, wh_ground_truth/1e3, label='WH ground truth', color='black')
    # ax.plot(time_col, hvac_predicted_total/1e3, label='HVAC predicted', color='tab:red')
    # ax.plot(time_col, hvac_ground_truth/1e3, label='HVAC ground truth', color='tab:orange')

    # ax.legend(frameon=False)
    # ax.set_xlabel('Time [HH:MM]')
    # ax.set_ylabel('Demand [kW]')
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    # plt.tight_layout()
    # plt.savefig('./test.png')
    # quit ()

    # wh_resulting_stats = get_statistical_metrics_for_der_profiles (
    #     dfs=wh_cosim_results_df,
    #     bayesian_matrices=wh_bayesian_matrices
    #     )
    
    # hvac_resulting_stats = get_statistical_metrics_for_der_profiles (
    #     dfs=hvac_cosim_results_df,
    #     bayesian_matrices=hvac_bayesian_matrices
    #     )

    sns.set_style ('whitegrid')

    wh_state_matrix = build_der_state_matrix (cosim_results_df= wh_cosim_results_df)
    hvac_state_matrix = build_der_state_matrix (cosim_results_df= hvac_cosim_results_df)
    
    time_col = pd.to_datetime(wh_only_xfmr_demand['# timestamp']).dt.strftime ('%H:%M')
    wh_ground_truth = pd.to_numeric(wh_only_xfmr_demand['power_out'], errors='coerce') / 1e3
    
    wh_total_state = wh_state_matrix.sum(axis=1)
    hvac_total_state = hvac_state_matrix.sum(axis=1)
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

    # Top plot (small) → states
    ax1 = fig.add_subplot(gs[0])
    # Bottom plot (big) → kW
    ax2 = fig.add_subplot(gs[1])

    # ===== TOP: STEP PLOT =====
    # Stacked bars (individual WH states)
    hvac_state_matrix.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        width=0.9,
        alpha=0.6,
        legend=True  # avoid clutter
    )

    # Step line (total state)
    ax1.step(
        range(len(hvac_state_matrix)),
        hvac_state_matrix,
        where='mid',
        linewidth=0,
        color='black',
        label='Total State'
    )

    # ===== BOTTOM: LINE PLOT =====
    ax2.plot(
        time_col,
        hvac_bayesian_matrices['kw_mean'].sum(axis=1) / 1e3,
        linewidth=2,
        label='Predicted: WH Aggregated kW'
    )

    ax2.plot(
        time_col,
        hvac_mean_matrix.sum(axis=1),
        linewidth=2,
        label='Mean Matrix'
    )

    # ax2.plot(
    #     time_col,
    #     wh_ground_truth,
    #     linewidth=2,
    #     label='Ground Truth: WH Aggregated kW'
    # )

    ax2.plot(
        time_col,
        full_house_xfmr_demand ['power_out'] / 1e3,
        linewidth=2,
        label='Ground Truth: Full House Xfmr Demand'
    )

    ax2.set_ylabel("Power [kW]")
    ax2.set_xlabel("Time [HH:MM]")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Reduce x-axis clutter
    ax1.set_ylabel("State (Stacked)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax1.legend(
        title="Water Heaters",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        ncol=2,
        frameon=False
    )
    ax1.set_xticklabels([])
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax2.set_xlim (time_col.min(), time_col.max ())

    # Shared title
    fig.suptitle("Water Heater States and Aggregated Demand", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.show()
    # print(wh_state_matrix)
    quit()
    wh_state_matrix = build_der_state_matrix (cosim_results_df= wh_cosim_results_df)
    hvac_state_matrix = build_der_state_matrix (cosim_results_df= hvac_cosim_results_df)
    
    wh_all_off_chunks = (wh_state_matrix.sum(axis=1) == 0)

    hvac_all_off_chunks = (hvac_state_matrix.sum(axis=1) == 0)

    # quit()
    wh_off_indices = np.where(wh_all_off_chunks.values)[0]
    hvac_off_indices = np.where(hvac_all_off_chunks.values)[0]
    # off_values = full_house_xfmr_demand['power_out'].values[wh_off_indices]
    wh_off_indices = np.asarray(wh_off_indices, dtype=float)
    wh_off_values = np.asarray(full_house_xfmr_demand['power_out'].values[wh_off_indices.astype(int)], dtype=float)

    hvac_off_indices = np.asarray(hvac_off_indices, dtype=float)
    hvac_off_values = np.asarray(full_house_xfmr_demand['power_out'].values[hvac_off_indices.astype(int)], dtype=float)


    print(f"WH all-OFF chunks: {wh_all_off_chunks.sum()}")
    print(f"HVAC all-OFF chunks: {hvac_all_off_chunks.sum()}")
    # print(f"Both all-OFF chunks: {both_off_chunks.sum()}")
    # quit()

    wh_background = full_house_xfmr_demand['power_out'].copy()
    # We are keeoing only the OFF WHs states of the feeder demand:
    wh_background_demand = wh_background[~wh_all_off_chunks] = np.nan

    # wh_background_demand = np.interp(
    #     np.arange(144),
    #     wh_off_indices,
    #     wh_off_values
    #     )
    hvac_background = full_house_xfmr_demand['power_out'].copy()
    hvac_background_demand = hvac_background [~hvac_all_off_chunks.values]



    # hvac_background_demand = np.interp(
    #     np.arange(144),
    #     hvac_off_indices,
    #     hvac_off_values
    #     )
    
    subtracted_wh_only_demand = np.clip (wh_background_demand, 0, None)
    subtracted_wh_only_demand = full_house_xfmr_demand ['power_out'].values - wh_background_demand

    subtracted_hvac_only_demand = np.clip (hvac_background_demand, 0, None)
    subtracted_hvac_only_demand = full_house_xfmr_demand ['power_out'].values - hvac_background_demand
    
    wh_only_xfmr_demand['power_out'] = round(pd.to_numeric(wh_only_xfmr_demand['power_out'], errors='coerce') / 1e3, 2)
    hvac_only_xfmr_demand['power_out'] = round(pd.to_numeric(hvac_only_xfmr_demand['power_out'], errors='coerce') / 1e3, 2)
    time_col = pd.to_datetime(full_house_xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')
    
    full_house_xfmr_demand ['power_out'] = round(pd.to_numeric(full_house_xfmr_demand ['power_out'], errors='coerce') / 1e3, 2)

    wh_background_demand = np.round(wh_background_demand/1e3, 2)
    hvac_background_demand = np.round(pd.to_numeric(hvac_background_demand)/1e3, 2)

    fig, ax = plt.subplots (figsize=(16,6))
    ax.plot (time_col, wh_background_demand, color='tab:blue', label = 'background demand')
    ax.plot(time_col, full_house_xfmr_demand ['power_out'], color='black', alpha=0.5, label = 'full house feeder demand')
    ax.plot (time_col, hvac_only_xfmr_demand['power_out'], color = 'red', alpha=0.5, label='HVAC only feeder demand')
    ax.xaxis.set_major_locator (ticker.MaxNLocator (20))
    ax.set_xlim (time_col.min (), time_col.max ())
    # ax.grid ()
    ax.legend ()
    # plt.savefig('./test.png')
    plt.show()
    quit()

    # background_demand = background.infer_objects (copy=False).interpolate (method = 'linear')

    # subtracted_wh_only_demand = full_house_xfmr_demand ['power_out'].values - background_demand.values
    # subtracted_wh_only_demand = np.clip (subtracted_wh_only_demand, 0, None)
    
    # fig, ax = plt.subplots (ncols=1, nrows=1, figsize=(16,6))
    
    # wh_bayesian_matrices['kw_mean']['total'] = wh_bayesian_matrices['kw_mean'].sum (axis=1)
    # wh_bayesian_matrices['kw_mean']['total'] = pd.to_numeric (wh_bayesian_matrices['kw_mean']['total'])/1e3
    # full_house_xfmr_demand ['power_out'] = pd.to_numeric (full_house_xfmr_demand ['power_out'])/1e3

    # subtracted_wh_only_demand = [round(i/1e3, 2) for i in subtracted_wh_only_demand]
    # wh_only_xfmr_demand['power_out'] = pd.to_numeric(wh_only_xfmr_demand['power_out'])/1e3
    # ax.plot (time_col, subtracted_wh_only_demand,
    #          label = 'Subtracted Background from FH Xfmr Demand [kW]', color='tab:blue')
    
    # ax.plot (time_col, round(wh_only_xfmr_demand['power_out'], 2),
    #          label = 'Ground Truth - Xfmr Demand : WHs Only [kW]', color='black', alpha=0.5)
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    # ax.legend ()
    # ax.set_xlabel('Time [HH:MM]', weight='bold')
    # ax.set_ylabel('Feeder / WH Demand [kW]', weight='bold')
    # plt.savefig('./test.png')