# %%
# Import libs
import os
import time
import random

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import comb
import matplotlib.pyplot as plt
from pprint import pprint as pp
import matplotlib.ticker as ticker
from scipy.stats import beta, binom
import bayesian_experiment as bayesian
# %%

def initialize_history () -> dict:
    return {
        'hour': [],
        'H': [],
        'T': [],
        'alpha': [],
        'beta': [],
        'theta_mean': [],
        'theta_std': [],
        'ci_lower': [],
        'ci_upper': []
    }

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

def prepare_data (df : pd.DataFrame):
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

    H = (df['state'] == 1).sum()
    T = (df['state'] == 0).sum()
    n = H + T

    return H, T, n

def calculate_likelihood (theta_values : np.array, H, T, n):
    return binom.pmf (H, n, theta_values)

# %%
if __name__ == '__main__':
    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
    
    input_paths = [s.strip() for s in input_paths if s.strip()]
    
    df = pd.read_csv (input_paths[0], usecols=['Time', 'Water Heating Electric Power (kW)'])
    
    df['state'] = (df[df.columns[1]] > 4).astype(int)
    
    df['Time'] = pd.to_datetime(df['Time'])

    theta_values = np.linspace (0.001, 0.999, 1000)

    history = initialize_history ()

    # for date, group_df in grouped:

    for hour in range(24):

        # df_hour = group_df[group_df['Time'].dt.hour == hour]
        df_hour = df[df['Time'].dt.hour == hour]

        alpha = 1
        
        beta_param = 1

        H, T, n = prepare_data (df=df_hour)
        
        alpha_post = alpha + H
        
        beta_post = beta_param + T


        stats = calculate_stats (alpha= alpha_post, beta_param=beta_post)

        prior = bayesian.calculate_prior (theta_values=theta_values, alpha=alpha, beta_param=beta_param)

        likelihood = calculate_likelihood (theta_values=theta_values, H=H, T=T, n=n)

        evidence = bayesian.calculate_evidence (theta_values=theta_values, likelihood=likelihood, prior=prior)

        posterior = bayesian.calculate_posterior_conjugate (theta_values=theta_values,
                                                            alpha_posterior=alpha_post,
                                                            beta_posterior=beta_post)
        history['hour'].append(hour)
        history['H'].append(H)
        history['T'].append(T)
        history['alpha'].append(alpha_post)
        history['beta'].append(beta_post)
        history['theta_mean'].append(stats['mean'])
        history['theta_std'].append(stats['std'])
        history['ci_lower'].append(stats['ci_lower'])
        history['ci_upper'].append(stats['ci_upper'])

    plt.figure(figsize=(12, 6))
    plt.plot(history['hour'], history['theta_mean'], linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel(r'$\theta$ (Probability of ON)')
    plt.fill_between (history['hour'], history['ci_lower'], history['ci_upper'] ,alpha=0.3)
    plt.title('Water Heater Activity Throughout the Day')
    plt.grid(True)
    plt.savefig ('./time-varying-theta.png', dpi=150)
    plt.show()
