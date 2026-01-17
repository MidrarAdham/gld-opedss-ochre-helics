# %%
import statistics
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# My functions
import bayesian_plots as my_vis
import bayesian_experiment as my_bayesian
# %%
def initialize_history ():
    return {
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

def initialize_theta_history () -> dict:
    return {
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

def time_varying_theta (theta_values : np.array, df: pd.DataFrame):
    """
    calculate theta for each hour of the day. Each hour is analyzed independently with its own
    Bayesian terms (prior).
    
    :param theta_values: Array of theta values for PDF calculations
    :type theta_values: np.array
    :param df: Dataframe with WH information and state column
    :type df: pd.DataFrame
    """

    df['Time'] = pd.to_datetime (df['Time'])

    history = initialize_theta_history ()

    for hour in range(24):

        hour_data = df[df['Time'].dt.hour == hour].copy()
        # I could have called the prepare_data function in the bayesian import, but it is a pain to add the slice data info (ws & start index)
        H_hour = (hour_data['state'] == 1).sum()
        T_hour = (hour_data['state'] == 0).sum()
        n_hour = len(hour_data)

        alpha_prior,beta_prior = 1, 1

        alpha_posterior = alpha_prior + H_hour
        beta_posterior = beta_prior + T_hour

        stats = my_bayesian.calculate_stats (alpha=alpha_posterior, beta_param=beta_posterior)

        # posterior_pdf = beta.pdf (theta_values, alpha_posterior, beta_posterior)
        posterior_conj = my_bayesian.calculate_posterior_conjugate (theta_values=theta_values,
                                                                 beta_posterior=beta_posterior,
                                                                 alpha_posterior=alpha_posterior
                                                                 )

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
        history['posterior'].append(posterior_conj)
    
    return history
# %%
def sequential_bayesian_implementation (theta_values : np.array, df : pd.DataFrame,
                                        num_chunks : int, window_size : int) -> dict:
    
    """
    The sequential bayesian calculates the posterior evolution over the the day (or whatever the dataset length is).

    There are two methods to calculate the posterior:
    1- Calculating every term individually - likelihood, prior, and normalization constant
    2- Or, we can use the posterior conjugate. Both return the same results!
    
    NOTE: posterior conjugate only works with beta prior and binomial likelihood
    
    :param theta_values: distribution of theta values (probability in this case)
    :type theta_values: np.array
    :param df: df with state column
    :type df: pd.DataFrame
    :param num_chunks: The number of data slices to process
    :type num_chunks: int
    :param window_size: The number of window sizes to process
    :type window_size: int
    :return: history of the results data recorded at each chunk
    :rtype: dict
    """

    history = initialize_history ()

    # choose prior parameters (removed the function, it's unccessary)
    alpha, beta_param = 1, 1

    for chunk_idx in range(num_chunks):
        
        start_idx = chunk_idx * window_size
        
        # df_sliced is used for debugging
        Hsliced, Tsliced, nsliced, df_slice = my_bayesian.prepare_data (df=df, start_index=start_idx,
                                                                     window_size=window_size)
    
        posterior_alpha = alpha + Hsliced
        posterior_beta = beta_param + Tsliced

        stats = my_bayesian.calculate_stats (alpha=posterior_alpha, beta_param=posterior_beta,
                                          ci_lower_thresh=0.001, ci_upper_thresh=0.999)

        prior = my_bayesian.calculate_prior (theta_values=theta_values,
                                 alpha=alpha, beta_param=beta_param)
        
        likelihood = my_bayesian.calculate_likelihood (theta_values=theta_values,
                                           H=Hsliced, T=Tsliced, n=nsliced)
        
        evidence = my_bayesian.calculate_evidence (theta_values=theta_values,
                                       likelihood=likelihood,
                                       prior=prior)
        
        posterior = my_bayesian.calculate_posterior_conjugate (theta_values=theta_values,
                                                            alpha_posterior=posterior_alpha,
                                                            beta_posterior=posterior_beta
                                                            )
        
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

def homogenity_modeling (sequential_history : dict):
    """
    Scenario:
    - In a world full of water heaters, I have a subset (sample) that contains 27 WHs.
    - What is the probability that this subset will be on in a given time?
    - To calculate this probability, we need to find the optimal alpha and beta
    
    :param sequential_history: Description
    :type sequential_history: dict
    """
    final_thetas = extract_final_thetas (sequential_history=sequential_history)
    theta_values = [data['theta'] for data in final_thetas.values()]
    fitted_params = fit_beta_distribution (theta_values=theta_values)
    # my_vis.plot_heterogeneity_distribution (theta_values=theta_values, fitted_params=fitted_params)
    # my_vis.plot_qq_plot (theta_values=theta_values, fitted_params=fitted_params)
    # my_vis.analyze_percentiles (theta_values=theta_values, fitted_params=fitted_params)

def fit_beta_distribution (theta_values : list) -> dict:
    """
    What is the best alpha and beta parameters for the whole dataset (27 Whs)? 
    - We answer this question by calculating the "common".
    - From the common, we can calculate the initial alpha and beta
    - Now we ask ourselves a question, are these alpha and beta a good choice?
    - We can quantify the "goodness" of alpha and beta by using the negative log likelihood.
    - Here is how the negative likelihood works:
        - Start with an initial guess
        - Check if it is a good choice (is the negative likelihood close to zero?)
        - if not, change the alpha and beta in a very small manner
        - Check again. If not, change alpha and beta in a small manner.
        - Essentially, we are tryoing to minimze the negative log likelihood

    The common represent the alpha + beta. A higher common value says that 
    the distribution is more concentrated around the mean.
   
    :param theta_values: Description
    :type theta_values: list
    :return: Description
    :rtype: Any
    """
    theta_array = np.array(theta_values)
    mean_theta = np.mean (theta_array)
    var_theta = np.var (theta_array)
    common = ((mean_theta * (1-mean_theta)) / var_theta) - 1
    alpha_moment = common * mean_theta
    beta_moment = (1-mean_theta) * common

    # Optimize the alpha and beta choices

    def negative_log_likelihood (parameters):
        a, b = parameters
        if a <= 0 or b <= 0:
            return 1e10 # alpha and beta should never be less than or equal to zero, so punish if they are less than zero
        # if they aren't less than zero, then sum up the negative log likelihood
        return -np.sum(beta.logpdf(theta_array, a, b))
    
    result = minimize (negative_log_likelihood,
                       x0=[alpha_moment, beta_moment],
                       method='L-BFGS-B',
                       bounds=[(0.1,1000), (0.1, 1000)]
                       )
    alpha_mle, beta_mle = result.x

    return {
        'alpha_mom': alpha_moment,
        'beta_mom': beta_moment,
        'alpha_mle': alpha_mle,
        'beta_mle': beta_mle,
        'mean_empirical': mean_theta,
        'std_empirical': np.std(theta_array)
        }
    


def extract_final_thetas (sequential_history) -> dict:
    final_thetas = {}

    for wh_name, history in sequential_history.items():
        final_theta = history['mean'][-1]
        final_alpha = history['alpha'][-1]
        final_beta = history['beta'][-1]
    
        final_thetas[wh_name] = {
            'theta': final_theta,
            'alpha': final_alpha,
            'beta': final_beta
            }
    return final_thetas


# %%
if __name__ == '__main__':

    # 
    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
    
    input_paths = [s.strip() for s in input_paths if s.strip()]

    time_changing_theta_history = {}
    sequential_bayesian_history = {}

    for input_path in input_paths:

        df = my_bayesian.load_wh_data (input_path)

        # Create binary states
        df = my_bayesian.create_binary_states (df=df, threshold=5)

        # Set the theta values:
        theta_values = np.linspace (0.001, 0.999, 1000)

        # define loop parameters:
        window_size, num_chunks = 10, 144
            
        bldg_id = input_path.split('/')[-3]

        # Run the bayesian implementation:
        seq_step_history = sequential_bayesian_implementation (theta_values=theta_values,
                                                    df=df, num_chunks=num_chunks,
                                                    window_size=window_size)
        
        theta_step_history = time_varying_theta (theta_values=theta_values, df=df)
        
        sequential_bayesian_history [bldg_id] = seq_step_history
        time_changing_theta_history [bldg_id] = theta_step_history
    
    # extract_final_thetas (sequential_history=sequential_bayesian_history)
    homogenity_modeling (sequential_history=sequential_bayesian_history)
    