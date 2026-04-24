'''
Author: Midrar Adham
Created: Thu Apr 23 2026
'''
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class OrdinaryLeastSquare:
    
    def __init__(self):
        
        # define paths
        self.wh_demand_results_path = '../results/wh_cosim/'
        self.hvac_demand_results_path = '../results/hvac_cosim/'
        self.all_ders_demand_results_path = '../results/total_house_consumption/'

        # define variables:
        self.wh_threshold = 5000.0
        self.hvac_threshold = 100.0
    
    def _collect_files_from_directories (self, files_dir : str) -> list:
        '''
        append each csv file to a list and returns the path/filenames
        '''
        return [f'{files_dir}{fname}' for fname in os.listdir (files_dir) if 'ochre' in fname]

    def _initialize_history (self) -> dict:

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
    
    def _calculate_stats (self, alpha : int,
                         beta_param : int,
                         ci_lower_thresh : float = 0.025,
                         ci_upper_thresh : float = 0.9745
                         ) -> dict:

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
    
    def _bayesian_engine (self):
        # initialization of alpha and beta in a beta function
        initial_alpha = 1
        initial_beta = 1
        # generate theta values
        theta_values = np.linspace (0.001, 0.999, 1000)
        # Get the history to record the results:
        history = self._initialize_history ()
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
            stats = self._calculate_stats (
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
    
    def _read_files_create_binary_states (self, filename : str, threshold : float) -> pd.DataFrame:

        df = pd.read_csv (filename, header=0, names=['time', 'power_out'] ,skiprows=8)
        df.loc[:, 'time'] = df['time'].apply (lambda x: x.strip ('PST'))
        df.loc[:, 'time'] = pd.to_datetime (df['time'])
        df.loc[:, 'power_out'] = df['power_out'].apply (lambda x: complex (x))
        df.loc[:, 'power_out'] = df['power_out'].apply(lambda x: x.real)
        # create binary states
        df['state'] = (df[df.columns[1]] > threshold).astype('bool')

        return df

    def run_bayesian_analysis (self, results_path : str, threshold : float):


        """
        Create binary states for each DER and put all the dataframes in a single dictionary.
        """
        results_df = self._collect_files_from_directories (files_dir = results_path)

        for filename in results_df:
            # read each file and clean up the format
            df = self._read_files_create_binary_states (filename=filename)
            

if __name__ == '__main__':
    
    ols = OrdinaryLeastSquare ()
    ols.run_bayesian_analysis (results_path=ols.wh_demand_results_path, threshold=ols.wh_threshold)
    
    

