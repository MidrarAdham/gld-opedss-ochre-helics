# %%
# Import libs
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
from scipy.special import comb # calculates combinations (N choose K)
# %%
def load_wh_data (filepath : str) -> pd.DataFrame:
    """
    Read a csv file using pandas. This method read df with:

            'Time', 'Water Heating Electric Power (kW)'
    
    :param input_path: A string indicating the csv file Path
    :type input_path: str
    :return: A dataframe
    :rtype: DataFrame
    """

    df = pd.read_csv (filepath, usecols = ['Time', 'Water Heating Electric Power (kW)'])
    return df

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
    :return: sliced dataframe
    :rtype: pd.DataFrame
    """
    # Slice the data:
    # df_sliced = df.head (60)
    df_sliced = df.iloc[start_index: start_index+window_size]
    H = (df_sliced['state'] == 1).sum()
    T = (df_sliced['state'] == 0).sum()
    n = H + T

    return H, T, n, df_sliced

def choose_prior (alpha : int = 1, beta_param : int = 1) -> dict:
    """
    Set the prior distribution P(theta) = Beta (alpha, beta)
    
    :param alpha: prior evidence for ON (WH is ON)
    :type alpha: int
    :param beta_param: prior evidence for OFF (WH is OFF)
    :type beta_param: int
    """
    return {'alpha' : alpha, 'beta': beta_param}

def calculate_likelihood (theta_values : np.array, H : int, T : int, n : int) -> np.array:
    """
    Calculate P(X | theta) for different theta values
    
    P(X | theta) = (n choose H) theta^H (1-theta)^n-H
    
    :param theta_values: array of theta values to evaluate (i.e., np.linspace)
    :type theta_values: np.array

    :param H: number of ONs
    :type H: int
    :param T: number of OFFs
    :type T: int
    :param n: total observations
    :type n: int
    :return: array of likelihood values
    :rtype: np.array
    """
    binom_coeff = comb (N=n, k=H, exact=True)
    likelihood = binom_coeff * (theta_values**H) * ((1-theta_values)**T)

    return likelihood

def calculate_prior (theta_values : np.array, alpha : int, beta_param : int)->np.array:
    """
    Calculate the prior = P(theta) = Beta (alpha, beta)
    
    :param theta_values: distribution of theta values
    :type theta_values: np.array
    :param alpha: prior evidence for ON (WH is ON)
    :type alpha: int
    :param beta: prior evidence for OFF (WH is OFF)
    :type beta: int
    :return: prior array
    :rtype: np.array
    """
    prior = beta.pdf (theta_values, alpha, beta_param)
    return prior

def calculate_evidence (theta_values : np.array, likelihood : np.array, prior : np.array) -> float:
    """
    calculate the evidence = P(X) = ∫ P(X | theta) * P(theta) dtheta

    :param theta_values: distribution of theta values
    :type theta_values: np.array
    :param likelihood: calculated likelihood
    :type likelihood: np.array
    :param prior: calculated prior
    :type prior: np.array
    :return evidence P(theta): evidence single number
    :rtype: float
    """
    inegrand = likelihood * prior
    evidence = np.trapz (inegrand, theta_values)
    return evidence

# def calculate_posterior (theta_values:np.array, likelihood:np.array, prior:np.array, evidence:float) -> np.array:
def calculate_posterior (likelihood:np.array, prior:np.array, evidence:float) -> np.array:
    """
    Calculate the posterior = P(theta | X) = [P(X | theta) * P(theta)]/P(X)
    :param theta_values: distribution of theta values
    :type theta_values: np.array
    :param likelihood: calculated likelihood
    :type likelihood: np.array
    :param prior: calculated prior
    :type prior: np.array
    :param evidence P(theta): evidence single number
    :rtype: float
    :return posterior: P(X | theta)
    :rtype: np.array
    """
    posterior = (likelihood * prior)/(evidence)
    return posterior

def calculate_posterior_conjugate (theta_values, alpha_posterior, beta_posterior):
    """
    The current method is we calculate the prior, likelihood, and normlization factor. However, method 2 says:
    posterior = beta.pdf (theta_values, alpha+H, beta+T), that's it!
    :param H: number of ONs
    :type H: int
    :param T: number of OFFs
    :type T: int
    :param alpha: prior evidence for ON (WH is ON)
    :type alpha: int
    :param beta: prior evidence for OFF (WH is OFF)
    :type beta: int
    :return updated alpha : alpha + H
    :rtype int
    :return updated beta_param : beta_param + T
    :rtype int
    """
    # alpha_posterior = alpha + H
    # beta_posterior = beta_param + T

    posterior = beta.pdf (theta_values, alpha_posterior, beta_posterior)

    return posterior

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


def bayesian_implementation (H: int, T: int, n: int):
    """
    There are two methods to calculate the posterior:
    1- Calculating every term individually - likelihood, prior, and normalization constant
    2- Or, we can use the posterior conjugate. Both return the same results!
    
    NOTE: posterior conjugate only works with beta prior and binomial likelihood
    """
    prior_params = choose_prior (alpha=10, beta_param=1)

    theta_values = np.linspace (0.001, 0.999, 1000)

    likelihood = calculate_likelihood (theta_values=theta_values, H=H, T=T, n=n)

    # plot_likelihood (theta_values=theta_values, likelihood=likelihood, H=H, n=n)

    # prior is expected be flat, because alpha = 1 and beta = 1
    prior = calculate_prior (theta_values=theta_values, alpha=prior_params['alpha'], beta_param=prior_params['beta'])

    evidence = calculate_evidence (theta_values=theta_values, likelihood=likelihood, prior=prior)
    
    posterior = calculate_posterior (likelihood=likelihood, prior=prior, evidence=evidence)
    
    alpha_posterior = prior_params['alpha'] + H
    beta_posterior = prior_params['beta'] + T

    posterior_conj = calculate_posterior_conjugate (theta_values=theta_values,
                                                    alpha_posterior=alpha_posterior,
                                                    beta_posterior=beta_posterior
                                                    )

    return posterior_conj
# %%
if __name__ == '__main__':

    # 
    root = Path (__file__).resolve ().parents[3]
    
    dataset_dir = root / 'cosimulation_tools' / 'dss-ochre-helics' / 'profiles' / 'one_week_wh_data'

     # Get the dataset input files:
    input_files = [file for file in dataset_dir.iterdir()]

    # Read the input files
    df = load_wh_data (filepath=input_files[0])

    # Create binary states
    df = create_binary_states (df=df, threshold = 4500.0)

    # initialize the state of the given profile
    H, T, n, df_sliced = prepare_data (df=df, start_index=0, window_size=60)
    
    posterior = bayesian_implementation (H=H, T=T, n=n)