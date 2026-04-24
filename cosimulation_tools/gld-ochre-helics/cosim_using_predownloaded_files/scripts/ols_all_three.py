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
        'posterior' : [],
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
    print(all_histories)
    quit()
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
    wh_state_matrix = build_der_state_matrix (cosim_results_df= wh_cosim_results_df)
    hvac_state_matrix = build_der_state_matrix (cosim_results_df= hvac_cosim_results_df)
    wh_all_off_chunks = (wh_state_matrix.sum(axis=1) == 0)
    wh_off_indices = np.where(wh_all_off_chunks.values)[0]
    background_demand = np.interp (
        np.arange (144),
        wh_off_indices, 
        full_house_xfmr_demand['power_out'].values [wh_off_indices]
    )
    estimated_combined_demand = np.clip (
        full_house_xfmr_demand['power_out'].values - background_demand, 0, None
    )

    estimated_combined_df = full_house_xfmr_demand.copy ()
    estimated_combined_df ['power_out'] = estimated_combined_demand
    combined_bayesian_matrices, combined_mean_matrix = create_matrices_from_bayesian_results (
        df=combined_df,
        xfmr_df=estimated_combined_df
        )

    # print(f"Estimated combined mean: {estimated_combined_demand.mean():.1f} W")
    # print(f"True combined mean: {combined_der_demand['power_out'].mean():.1f} W")
    # quit()
    x_wh = combined_mean_matrix [wh_cols].sum (axis=1).values
    x_hvac = combined_mean_matrix [hvac_cols].sum (axis=1).values
    
    background_constant = full_house_xfmr_demand['power_out'].values[wh_off_indices].mean()
    print(f"Estimated background constant: {background_constant:.1f} W")

    # Subtract background before OLS
    feeder_minus_background = full_house_xfmr_demand['power_out'].values - background_constant

    # OLS without intercept now
    A = np.column_stack([x_wh, x_hvac])
    b = feeder_minus_background

    kw_estimate, _, _, _ = np.linalg.lstsq (A, b, rcond=None)
    kw_wh, kw_hvac = kw_estimate
    print(f"Estimated kW per WH: {kw_wh:.1f} W")
    print(f"Estimated kW per HVAC: {kw_hvac:.1f} W")

    wh_predicted = kw_wh * x_wh
    
    # --- STEP 1: Keep your current OLS for WH ---
    # (Assuming you just ran the simultaneous OLS from the previous step)
    # wh_predicted = kw_wh * x_wh

    # --- STEP 2: Create the "Cleaner" Signal ---
    # We subtract the WH prediction from the raw feeder demand.
    # This 'y_hvac' should now mostly contain HVAC + Background.
    y_hvac = full_house_xfmr_demand['power_out'].values - wh_predicted

    # --- STEP 3: Second OLS just for HVAC and Baseline ---
    # We build a new A matrix without the WH column.
    A_hvac_only = np.column_stack([x_hvac, np.ones(len(x_hvac))])

    # Solve for HVAC kW and a refined Baseline
    hvac_estimate, _, _, _ = np.linalg.lstsq(A_hvac_only, y_hvac, rcond=None)
    kw_hvac_new, baseline_new = hvac_estimate

    # --- STEP 4: Update Results ---
    hvac_predicted_new = kw_hvac_new * x_hvac

    print(f"--- Sequential OLS Results ---")
    print(f"Refined kW per HVAC: {kw_hvac_new:.1f} W")
    print(f"Refined Constant Background: {baseline_new:.1f} W")
    print(f"New HVAC mean: {hvac_predicted_new.mean():.1f} W")
    print(f"Target HVAC mean: {hvac_only_xfmr_demand['power_out'].mean():.1f} W")
    quit()

    # print(f"WH predicted mean: {wh_predicted.mean():.1f} W")
    # print(f"WH ground truth mean: {wh_only_xfmr_demand['power_out'].mean():.1f} W")
    # print(f"HVAC predicted mean: {hvac_predicted.mean():.1f} W")
    # print(f"HVAC ground truth mean: {hvac_only_xfmr_demand['power_out'].mean():.1f} W")
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot A: The number of units ON
    ax1.stackplot(time_col, x_wh, x_hvac, labels=['WH Units On', 'HVAC Units On'], alpha=0.5)
    ax1.set_ylabel("Count of Units")
    ax1.legend(loc='upper left')

    # Plot Ax: The scaled power contribution
    ax2.stackplot(time_col, wh_predicted, hvac_predicted, labels=['WH Contribution', 'HVAC Contribution'], alpha=0.7)
    ax2.set_ylabel("Power (W)")
    ax2.legend(loc='upper left')

    # Plot y vs Ax: The Resulting Fit
    ax3.plot(time_col, b, color='black', alpha=0.3, label='Actual (y)')
    ax3.plot(time_col, wh_predicted, color='green', linestyle='--', label='OLS Prediction (Ax)')
    ax3.set_ylabel("Total Power (W)")
    ax3.legend(loc='upper left')
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(20))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig ('gem_code.png')
    plt.show()

    quit()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=True)

    ax.plot(time_col, wh_predicted/1e3, label='WH predicted', color='tab:blue')
    ax.plot(time_col, pd.to_numeric(wh_only_xfmr_demand['power_out'])/1e3, 
            label='WH ground truth', color='black', alpha=0.7)
    ax.set_ylabel('Demand [kW]')
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

    # ax[1].plot(time_col, hvac_predicted/1e3, label='HVAC predicted', color='tab:red')
    # ax[1].plot(time_col, pd.to_numeric(hvac_only_xfmr_demand['power_out'])/1e3,
    #         label='HVAC ground truth', color='black', alpha=0.7)
    # ax[1].set_ylabel('Demand [kW]')
    # ax[1].legend(frameon=False)
    # ax[1].xaxis.set_major_locator(ticker.MaxNLocator(20))

    plt.tight_layout()
    plt.savefig ('test.png')
    plt.show()
    quit()

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
    print('done')
    quit()