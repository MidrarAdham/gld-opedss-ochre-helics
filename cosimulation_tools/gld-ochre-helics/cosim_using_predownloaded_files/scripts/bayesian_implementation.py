import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


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
    df['state'] = (df[df.columns[1]] > threshold).astype(int)
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

def transformer_layer_data (df : pd.DataFrame, xfmr_df : pd.DataFrame):
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
    # kw_mean  = mean_matrix.div(mean_matrix.sum(axis=1), axis=0).multiply(feeder_demand, axis=0)
    # kw_lower = ci_lower_matrix.div(ci_lower_matrix.sum(axis=1), axis=0).multiply(feeder_demand, axis=0)
    # kw_upper = ci_upper_matrix.div(ci_upper_matrix.sum(axis=1), axis=0).multiply(feeder_demand, axis=0)
    total = mean_matrix.sum(axis=1)  # shared denominator

    kw_mean  = mean_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_lower = ci_lower_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)
    kw_upper = ci_upper_matrix.div(total, axis=0).multiply(feeder_demand, axis=0)

    return kw_mean, kw_lower, kw_upper

def prepare_transformer_data (df : pd.DataFrame):
    df = df.head (1440)
    df = cleanup_results_files (df=df, col = 'power_out')
    df = df.drop ('power_in', axis=1)
    df = df.set_index ('# timestamp')
    df = df.resample ("10min").mean()
    df = df.reset_index ()

    return df
    

cosim_results_dir = '../results/wh_cosim/'
cosim_results_files = [f for f in os.listdir (cosim_results_dir) if 'ochre' in f]
wh_threshold = 5000.0
hvac_threshold = 5000.0

all_histories = {}
cosim_results_df = {}
for filename in cosim_results_files:
    df = pd.read_csv (cosim_results_dir+filename, skiprows=8)
    df = df.head (1440)
    df = cleanup_results_files (df=df, col='constant_power_12')
    df = create_binary_states (df=df, threshold=wh_threshold)
    all_histories[filename] = bayesian_implementation (df=df)
    cosim_results_df [filename] = df

df = pd.DataFrame (all_histories)
df = df.transpose ()
xfmr_demand = pd.read_csv (f'{cosim_results_dir}residential_transformer.csv', skiprows=8)
xfmr_demand = prepare_transformer_data (df=xfmr_demand)
kw_mean, kw_lower, kw_upper = transformer_layer_data (df=df, xfmr_df=xfmr_demand)
time_col = pd.to_datetime(xfmr_demand ['# timestamp']).dt.strftime ('%H:%M')

sns.set_style ('whitegrid')
# fig, ax = plt.subplots (1,1, figsize = (16, 6))

# for key, value in cosim_results_df.items ():
#     fig, ax = plt.subplots (1,1, figsize = (16, 6))
#     df = value.drop ('state', axis=1)
#     df = df.set_index ('# timestamp')
#     df = df.resample ('10min').mean()
#     ax.plot (time_col, df['constant_power_12']/1e3, linewidth=2,
#              color='grey', label='Ground Truth [kW]')
#     ax.plot (time_col, kw_mean[key]/1e3, linewidth=2, color='gold', label='Mean [kW]')
#     # ax.plot (time_col, kw_lower[key]/1e3, linewidth=1, color='gold', label = 'CI_lower')
#     # ax.plot (time_col, kw_upper[key]/1e3, linewidth=1, color='red', label='CI_upper')
#     lower = np.asarray(kw_lower[key], dtype=float) / 1e3
#     upper = np.asarray(kw_upper[key], dtype=float) / 1e3

#     ax.fill_between(time_col, lower, upper, color='black', alpha=0.3, label='Confidence Interval')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
#     ax.set_xlabel ('Time [hh:mm]')
#     ax.set_ylabel (f'{key} [kW]')
#     ax.legend ()
    # plt.savefig (f'./{key}.png')

for key, value in cosim_results_df.items():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    df = value.drop('state', axis=1)
    df = df.set_index('# timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.resample('10min').mean()

    x = pd.to_datetime(time_col)
    y_true = pd.to_numeric(df['constant_power_12'], errors='coerce').to_numpy() / 1e3
    y_mean = pd.to_numeric(kw_mean[key], errors='coerce').to_numpy() / 1e3
    y_low  = pd.to_numeric(kw_lower[key], errors='coerce').to_numpy() / 1e3
    y_up   = pd.to_numeric(kw_upper[key], errors='coerce').to_numpy() / 1e3

    # Confidence band first so it stays behind the lines
    ax.fill_between(
        x, y_low, y_up,
        color='tab:blue',
        alpha=0.15,
        linewidth=0,
        label='95% CI'
    )

    # Main lines
    ax.plot(
        x, y_true,
        color='black',
        linewidth=2.4,
        label='Ground truth'
    )

    ax.plot(
        x, y_mean,
        color='tab:blue',
        linewidth=2.0,
        label='Predicted mean'
    )

    # Grid and spines
    ax.grid(True, which='major', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Axis labels and title
    ax.set_title(f'{key}', fontsize=14, pad=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power [kW]')

    # Time formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    # Legend
    ax.legend(frameon=False, loc='upper right')

    # Tight layout
    plt.tight_layout()
    plt.savefig (f'./{key}.png')