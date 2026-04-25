'''
Author: Midrar Adham
Created: Thu Apr 23 2026
'''
import numpy as np
import pandas as pd
from scipy.stats import beta

class BayesianEstimator:
    
    def __init__(self,
                 window_size : int = 10,
                 num_chunks : int = 144,
                 discount : float = 0.3,
                 initial_alpha : int = 1,
                 initial_beta : int = 1,
                 ci_lower : float = 0.05,
                 ci_upper : float = 0.95
                 ):
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.discount = discount
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
    
    def _initialize_history (self):
        return {
            'std' : [],
            'beta' : [],
            'mean' : [],
            'chunk' : [],
            'alpha' : [],
            'H_chunk' : [],
            'T_chunk' : [],
            'ci_upper' : [],
            'ci_lower' : [],
            'posterior' : [],
        }
    
    def _calculate_stats (self, alpha_posterior : int, beta_posterior : int) -> dict:
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
        mean = (alpha_posterior) / (alpha_posterior + beta_posterior)
        # for a beta function, the variance can be calculated as follows:
        variance = (alpha_posterior * beta_posterior) / (((alpha_posterior + beta_posterior)**2) * (alpha_posterior+beta_posterior+1))
        
        std = np.sqrt (variance)
        
        # 95% confidence interval:
        ci_lower = beta.ppf (self.ci_lower, alpha_posterior, beta_posterior)
        ci_upper = beta.ppf (self.ci_upper, alpha_posterior, beta_posterior)
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
    
    def _bayesian_core_engine (self, df : pd.DataFrame):
        # generate theta values
        beta_p = self.initial_beta
        alpha = self.initial_alpha
        theta_values = np.linspace (0.001, 0.999, 1000)
        # Get the history to record the results:
        history = self._initialize_history ()

        for chunk_idx in range(self.num_chunks):
            start_index = chunk_idx * self.window_size
            # calculates heads and tails; the sum of ON and OFF states of DERs
            df_sliced = df.iloc [start_index: start_index+self.window_size]
            H = (df_sliced['state'] == 1).sum()
            T = (df_sliced['state'] == 0).sum()
            # posterior conjugate parameters:
            alpha_posterior = self.discount * alpha + H
            beta_posterior = self.discount * beta_p + T
            
            # Calculate statisitical parameters:
            stats = self._calculate_stats (alpha_posterior = alpha_posterior, beta_posterior = beta_posterior)
            
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

            alpha = alpha_posterior
            beta_p = beta_posterior

        return history
    
    def fit (self, df : pd.DataFrame) -> dict:
        
        history = self._bayesian_core_engine (df=df)
        return history
    
    def fit_many (self, all_dfs : dict) -> dict:

        all_histories = {}

        for filename, df in all_dfs.items ():
            
            all_histories [filename] = self.fit (df=df)
        
        return all_histories