# %%
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

    return H, T, n

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
        
        Hsliced, Tsliced, nsliced = prepare_data (df=df,
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

def plot_sequential_learning (history):
    fig, axes = plt.subplots (3,1, figsize=(12,12))
    chunks = history['chunk']

    # Plot the mean over the chunks
    axes[0].plot(chunks, history['mean'], linewidth=2, markersize=8, color='blue')
    axes[0].set_xlabel('Chunks number', fontsize=12)
    axes[0].set_ylabel(f'Estimate {r"$\theta$"} mean', fontsize=12)
    axes[0].set_title ('How chunks mean evolve over time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline (y=history['mean'][-1], color='skyblue', linestyle='--',
                     label=f'final {r"$\theta$"} estimate = {history["mean"][-1]:.3f}')
    axes[0].legend()

    # Plot the uncertainity over time
    # The uncertainity here is the std
    axes[1].plot (chunks, history['std'], linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Chunks Number', fontsize=12)
    axes[1].set_ylabel('Std', fontsize=12)
    axes[1].set_title ('How Uncertanity (Std) Decreases with more data (Chnuks)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Fill the coincidence intervals
    axes[2].fill_between (chunks, history['ci_lower'], history['ci_upper'],
                          alpha=0.1, color='purple', label='95% CI')
    axes[2].plot(chunks, history['mean'], linewidth=2, markersize=8, color='purple', label='Mean Estimate')
    axes[2].set_xlabel ('Chunk Number', fontsize=12)
    axes[2].set_ylabel (r'$\theta$', fontsize=12)
    axes[2].set_title ("Credible Interval Narrowing over time", fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig ('./Sequential_Bayesian.png', dpi=150, bbox_inches='tight')

def plot_posterior_evolution (history):
    theta_values = np.linspace (0.001, 0.999, 1000)
    prior_pdf = beta.pdf (theta_values, 1, 1)

    fig, ax = plt.subplots (figsize=(12,8))

    ax.plot(theta_values, prior_pdf[:len(theta_values)],
            linewidth=2, linestyle='--', color='gray',
            label=f'Initial Prior: Beta ({r"$\alpha = 1, \beta = 1$ "})')
    
    colors = plt.cm.viridis(np.linspace (0, 1, len(history['chunk'])))
    

    # ax.set_ylim (0, 2)
    ax.legend()

    for idx, chunk in enumerate (history['chunk']):
        alpha_post = history['alpha'][idx]
        beta_post = history['beta'][idx]

        posterior_pdf = history['posterior'][idx]
        # print(len(theta_values))
        
        ax.plot(theta_values, posterior_pdf,
                linewidth=2, color=colors[idx],
                label = f"After Chunk {chunk}: Beta ({r'$\alpha = $'}{alpha_post}, {r'$\beta = $'}{beta_post})"
                )
        
    ax.set_xlabel (r"probability of ON ($\theta$)")
    ax.set_ylabel ('Probability Density (PDF)')
    ax.set_title ('Posterior Distribution Evolution', fontsize=14, fontweight='bold')
    ax.legend (fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout ()
    plt.savefig ('./posterior_evolution.png', dpi=150)
# %%
if __name__ == '__main__':

    # 
    root = Path (__file__).resolve ().parents[3]
    
    dataset_dir = root / 'cosimulation_tools' / 'dss-ochre-helics' / 'profiles' / 'one_week_wh_data'

     # Get the dataset input files:
    input_files = [file for file in dataset_dir.iterdir()]

    # Read the input files
    df = bayesian.load_wh_data (filepath=input_files[0])

    # Create binary states
    df = bayesian.create_binary_states (df=df)

    # Set the theta values:
    theta_values = np.linspace (0.001, 0.999, 1000)

    # define loop parameters:
    window_size, num_chunks = 10, 6
        
    # Run the bayesian implementation:
    history = sequential_bayesian_implementation (theta_values=theta_values,
                                                  df=df, num_chunks=num_chunks, window_size=window_size)
    
    plot_sequential_learning (history=history)

    plot_posterior_evolution (history=history)

