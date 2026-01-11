# %%
"""
This script runs sequential Bayesian and time-varying theta for multiple DERs (WHs).
"""
# %%
import numpy as np
import pandas as pd
from scipy.stats import beta
import bayesian_plots as baysian_vis
import bayesian_experiment as bayesian

# %%
def prepare_data (df : pd.DataFrame, start_index : int, window_size : int) -> pd.DataFrame:
    """
    Extract slice of the data for analysis
    
    :param df: df with state column
    :type df: pd.DataFrame
    :return: H number of ONs
    :rtype: int
    :return: T number of OFFs
    :rtype: int
    :return: n total observations
    :rtype: int
    """
    # Slice the data:
    # df_sliced = df.head (60)
    df_sliced = df.iloc[start_index: start_index+window_size]
    H = (df_sliced['state'] == 1).sum()
    T = (df_sliced['state'] == 0).sum()
    n = H + T

    return H, T, n, df_sliced

def calculate_stats (alpha : int, beta_param : int) -> dict:
    
    mean = (alpha) / (alpha+beta_param)
    variance = (alpha * beta_param) / ( 
        ((alpha+beta_param)**2) * (alpha + beta_param + 1)
        )
    
    std = np.sqrt (variance)
    ci_lower = beta.ppf (0.025, alpha, beta_param)
    ci_upper = beta.ppf (0.975, alpha, beta_param)
    
    return {
        'mean' : mean, 'variance' : variance, 'std' : std, 'ci_lower' : ci_lower, 'ci_upper' : ci_upper
    }

def sequential_bayesian_implementation (theta_values : np.array, df : pd.DataFrame,
                                        num_chunks : int, window_size : int):
    """
    There are two methods to calculate the posterior:
    1- Calculating every term individually - likelihood, prior, and normalization constant
    2- Or, we can use the posterior conjugate. Both return the same results!
    
    NOTE: posterior conjugate only works with beta prior and binomial likelihood
    """

    history = {
        'std' : [],
        'beta' : [],
        'mean' : [],
        'prior' : [],
        'chunk' : [],
        'alpha' : [],
        'H_chunk' : [],
        'T_chunk' : [],
        'ci_upper' : [],
        'ci_lower' : [],
        'evidence' : [],
        'posterior' : [],
        'likelihood' : [],
    }

    # choose prior parameters (removed the function, it's unccessary)
    alpha, beta_param = 1, 1

    for chunk_idx in range(num_chunks):
        
        start_idx = chunk_idx * window_size
        
        # df_sliced is used for debugging
        Hsliced, Tsliced, nsliced, df_slice = prepare_data (df=df,
                                                             start_index=start_idx,
                                                             window_size=window_size
                                                             )
        
        posterior_alpha = alpha + Hsliced
        posterior_beta = beta_param + Tsliced

        stats = calculate_stats (alpha=posterior_alpha, beta_param=posterior_beta)

        prior = bayesian.calculate_prior (theta_values=theta_values,
                                 alpha=alpha, beta_param=beta_param)
        
        likelihood = bayesian.calculate_likelihood (theta_values=theta_values,
                                           H=Hsliced, T=Tsliced, n=nsliced)
        
        evidence = bayesian.calculate_evidence (theta_values=theta_values,
                                       likelihood=likelihood,
                                       prior=prior)
        
        posterior = bayesian.calculate_posterior_conjugate (theta_values=theta_values,
                                                            alpha_posterior=posterior_alpha,
                                                            beta_posterior=posterior_beta)
    

        history['prior'].append(prior)
        history['chunk'].append(chunk_idx)
        history['H_chunk'].append(Hsliced)
        history['T_chunk'].append(Tsliced)
        history['std'].append(stats['std'])
        history['evidence'].append(evidence)
        history['mean'].append(stats['mean'])
        history['posterior'].append(posterior)
        history['beta'].append(posterior_beta)
        history['likelihood'].append(likelihood)
        history['alpha'].append(posterior_alpha)
        history['ci_lower'].append(stats['ci_lower'])
        history['ci_upper'].append(stats['ci_upper'])

        alpha = posterior_alpha
        beta_param = posterior_beta

    return history

def calculate_stats (alpha : int, beta_param : int) -> dict:
    
    mean = (alpha) / (alpha+beta_param)
    variance = (alpha * beta_param) / ( 
        ((alpha+beta_param)**2) * (alpha + beta_param + 1)
        )
    
    std = np.sqrt (variance)
    ci_lower = beta.ppf (0.025, alpha, beta_param)
    ci_upper = beta.ppf (0.975, alpha, beta_param)
    
    return {
        'mean' : mean, 'variance' : variance, 'std' : std, 'ci_lower' : ci_lower, 'ci_upper' : ci_upper
    }

def time_varying_theta (theta_values : np.array, df: pd.DataFrame):
    """
    calculate theta for each hour of the day. Each hour is analyzed independently with its own
    Bayesian terms (prior).
    
    :param theta_values: Array of theta values for PDF calculations
    :type theta_values: np.array
    :param df: Dataframe with WH information and state column
    :type df: pd.DataFrame
    """
    history = {
        'hour': [],
        'H_total': [],
        'T_total': [],
        'n_total': [],
        'alpha': [],
        'beta': [],
        'mean': [],
        'std': [],
        'ci_lower': [],
        'ci_upper': [],
        'posterior': []
    }

    df['Time'] = pd.to_datetime (df['Time'])

    for hour in range(24):
        hour_data = df[df['Time'].dt.hour == hour].copy()
        H_hour = (hour_data['state'] == 1).sum()
        T_hour = (hour_data['state'] == 0).sum()
        n_hour = len(hour_data)



        alpha_prior,beta_prior = 1, 1

        alpha_posterior = alpha_prior + H_hour
        beta_posterior = beta_prior + T_hour

        stats = calculate_stats (alpha=alpha_posterior, beta_param=beta_posterior)

        posterior_pdf = beta.pdf (theta_values, alpha_posterior, beta_posterior)

        history['hour'].append(hour)
        history['H_total'].append(H_hour)
        history['T_total'].append(T_hour)
        history['n_total'].append(n_hour)
        history['alpha'].append(alpha_posterior)
        history['beta'].append(beta_posterior)
        history['mean'].append(stats['mean'])
        history['std'].append(stats['std'])
        history['ci_lower'].append(stats['ci_lower'])
        history['ci_upper'].append(stats['ci_upper'])
        history['posterior'].append(posterior_pdf)
    
    return history

def plot_comparison_all_wh(all_histories):
    """
    Master comparison function - calls all other comparison functions
    """
    baysian_vis.plot_final_theta_comparison(all_histories)
    baysian_vis.plot_evolution_comparison(all_histories)
    baysian_vis.plot_all_posteriors_detailed (all_histories=all_histories)

def time_varying_theta_plots(all_history, theta_values):
    """
    Master as well - Generate all time-varying theta plots for multiple WHs
    
    :param all_history: Dict of {wh_name: history_dict}
    :param theta_values: Array for PDF calculation
    """
    baysian_vis.plot_theta_by_hour(all_history)
    baysian_vis.plot_posterior_evolution_by_hour(all_history, theta_values)
    # baysian_vis.plot_data_availability(all_history)
    baysian_vis.plot_uncertainty_by_hour(all_history)
# %%
if __name__ == '__main__':

    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
        input_paths = [s.strip() for s in input_paths if s.strip()]

     # Get the dataset input files:
    all_histories_seq = {}
    all_histories_time_varying_theta = {}

    # Set the theta values:
    theta_values = np.linspace (0.001, 0.999, 1000)

    # define loop parameters:
    window_size, num_chunks = 10, 144

    for idx, input_file in enumerate (input_paths):

        # Read the input files
        df = bayesian.load_wh_data (filepath=input_file)

        # Create binary states
        df = bayesian.create_binary_states (df=df, threshold=0.5)
        
        # Run the bayesian implementation:
        seq_history = sequential_bayesian_implementation (theta_values=theta_values,
                                                      df=df, num_chunks=num_chunks, window_size=window_size
                                                      )
        theta_history = time_varying_theta (theta_values=theta_values, df=df)
        
        bldg_id = input_file.split('/')[-3]
        all_histories_seq [bldg_id] = seq_history
        all_histories_time_varying_theta [bldg_id] = theta_history

    
    time_varying_theta_plots (all_history=all_histories_time_varying_theta, theta_values=theta_values)
    
    plot_comparison_all_wh (all_histories=all_histories_seq)
