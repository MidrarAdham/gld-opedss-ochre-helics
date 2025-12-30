# %%
# Import libs
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import comb # calculates combinations (N choose K)
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
# %%
def load_wh_data (filepath : str) -> pd.DataFrame:
    """
    Read a csv file using pandas
    
    :param input_path: A string indicating the csv file Path
    :type input_path: str
    :return: A dataframe
    :rtype: DataFrame
    """
    return pd.read_csv (filepath)

def create_binary_states (df : pd.DataFrame) -> pd.DataFrame:
    """
    convert power values to binary states (ON = 1, OFF=0)
    
    :param df: a one-week water heater load profile
    :type df: pd.DataFrame
    :return: df with added "state" column
    :rtype: DataFrame
    """
    df['state'] = (df['tn_meter_4br_46:measured_real_power'] ==4500).astype(int)
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
    Calculate the posterior = P(X | theta) = [P(theta | X) * P(theta)]/P(X)
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

def calculate_posterior_conjugate (H, T, alpha, beta_param):
    """
    If we want to skip using the integration in calculating the evidence, we can use apparently.

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
    alpha_posterior = alpha + H
    beta_posterior = beta_param + T

    return alpha_posterior, beta_posterior

def plot_likelyhood (theta_values, likelihood, H, n):
    plt.figure (figsize=(10, 6))
    plt.plot (theta_values, likelihood, linewidth=2,
              label=f'P(theta | X) = ({n} choose {H}) * theta ^ {H} * (1-theta)^{n-H}')
    plt.xlabel ('theta (Probability of ON)', fontsize=12)
    plt.ylabel ('P(X | theta)', fontsize=12)
    plt.title ('Likelihood function', fontsize=14, fontweight='bold')
    plt.axvline (x=H/n, color='red', linestyle='--',label=f'Max at theta = {H/n:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    return plt

def plot_bayesian (theta_values, prior, likelihood, posterior, H, T, n):
    fig, axes = plt.subplots (3, 1, figsize=(10,12))
    print(prior)

    axes[0].plot(theta_values, prior, linewidth=2, color='blue')
    axes[0].set_ylim(0,1.5)
    axes[0].set_title('Prior: P(θ)', fontweight='bold')
    axes[0].set_xlabel('θ')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_values, likelihood, linewidth=2, color='green')
    axes[1].set_title(f'Likelihood: P(X|theta) where X = {H} ONs, {T} OFFs', 
                      fontweight='bold')
    axes[1].set_xlabel('theta')
    axes[1].set_ylabel('P(X|theta)')
    axes[1].axvline(x=H/n, color='red', linestyle='--', label=f'MLE = {H/n:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(theta_values, posterior, linewidth=2, color='red')
    axes[2].set_title('Posterior: P(θ|X)', fontweight='bold')
    axes[2].set_xlabel('θ')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig ('./test.png')
    return plt


def bayesian_implementation (H: int, T: int, n: int):
    """
    
    """
    prior_params = choose_prior (alpha=1, beta_param=1)

    theta_values = np.linspace (0.001, 0.999, 1000)

    likelihood = calculate_likelihood (theta_values=theta_values, H=H, T=T, n=n)

    # plot_likelyhood (theta_values=theta_values, likelihood=likelihood, H=H, n=n)

    # prior is expected be flat, because alpha = 1 and beta = 1
    prior = calculate_prior (theta_values=theta_values, alpha=prior_params['alpha'], beta_param=prior_params['beta'])

    evidence = calculate_evidence (theta_values=theta_values, likelihood=likelihood, prior=prior)
    
    posterior = calculate_posterior (likelihood=likelihood, prior=prior, evidence=evidence)

    plot_bayesian (theta_values=theta_values, prior=prior, likelihood=likelihood, posterior=posterior, H=H, T=n-H, n=n)

    return posterior
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
    df = create_binary_states (df=df)

    # initialize the state of the given profile
    H, T, n, df_sliced = prepare_data (df=df, start_index=0, window_size=60)
    
    posterior = bayesian_implementation (H=H, T=T, n=n)