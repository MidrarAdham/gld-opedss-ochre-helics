# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
from scipy.special import comb
import matplotlib.pyplot as plt
from pprint import pprint as pp
import matplotlib.ticker as ticker
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

def plot_final_theta_comparison(all_histories):
    """
    Compare final θ estimates across all water heaters
    """
    wh_names = list(all_histories.keys())
    final_thetas = [all_histories[wh]['mean'][-1] for wh in wh_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(wh_names)), final_thetas, color='steelblue', alpha=0.7)
    plt.axhline(y=np.mean(final_thetas), color='red', linestyle='--', 
                linewidth=2, label=f'Average θ = {np.mean(final_thetas):.4f}')
    
    plt.xlabel('Water Heater', fontsize=12)
    plt.ylabel(f'Final {r"$\theta = $"} Estimate', fontsize=12)
    plt.title(f'Final {r"$\theta = $"} Estimates Across All Water Heaters', fontsize=14, fontweight='bold')
    plt.xticks(range(len(wh_names)), wh_names, rotation=90, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../results/theta_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_all_posteriors_detailed(all_histories):
    """
    Plot all final posterior distributions with detailed statistics
    """
    theta_values = np.linspace(0.001, 0.999, 1000)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    for idx, (wh_name, history) in enumerate(all_histories.items()):
        # Final posterior parameters
        final_alpha = history['alpha'][-1]
        final_beta = history['beta'][-1]
        
        # Calculate statistics
        mean_theta = final_alpha / (final_alpha + final_beta)
        
        # Calculate posterior PDF
        posterior_pdf = beta.pdf(theta_values, final_alpha, final_beta)
        
        # Create detailed label
        label = (f'{wh_name}: '
                f'Beta({r"$\alpha$"}={final_alpha}, {r"$\beta$"}={final_beta}) | '
                f'{r"$\theta$"}={mean_theta:.3f}')
        
        # Plot
        ax.plot(theta_values, posterior_pdf, 
                linewidth=2.5, 
                color=colors[idx],
                label=label,
                alpha=0.8)
        
        # Mark the mean
        ax.axvline(x=mean_theta, color=colors[idx], 
                  linestyle=':', alpha=0.4, linewidth=1.5)
    
    ax.set_xlabel(f'{r"$\theta$"} (Probability of ON)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title('Posterior Distributions Comparison: All Water Heaters', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    # ax.set_xlim(0, 0.4)  # Adjust based on your data
    
    plt.tight_layout()
    plt.savefig('../results/all_posteriors_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_evolution_comparison(all_histories):
    """
    Compare how θ evolves over chunks for all water heaters
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    for idx, (wh_name, history) in enumerate(all_histories.items()):
        chunks = history['chunk']
        
        # Plot 1: Mean evolution
        axes[0].plot(chunks, history['mean'], 
                    linewidth=2, marker='o', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
        
        # Plot 2: Uncertainty evolution
        axes[1].plot(chunks, history['std'], 
                    linewidth=2, marker='s', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
        
        # Plot 3: Credible interval width
        ci_width = np.array(history['ci_upper']) - np.array(history['ci_lower'])
        axes[2].plot(chunks, ci_width, 
                    linewidth=2, marker='^', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
    
    # Formatting
    axes[0].set_ylabel(f'{r"$\theta = $"} Estimate', fontsize=12)
    axes[0].set_title(f'{r"$\theta = $"} Evolution Across Water Heaters', fontsize=14, fontweight='bold')
    axes[0].legend(loc='right', fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Standard Deviation', fontsize=12)
    axes[1].set_title('Uncertainty Evolution', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Chunk Number', fontsize=12)
    axes[2].set_ylabel('95% CI Width', fontsize=12)
    axes[2].set_title('Credible Interval Width Evolution', fontsize=14, fontweight='bold')
    axes[2].legend(loc='best', fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/evolution_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison_all_wh(all_histories):
    """
    Master comparison function - calls all other comparison functions
    """
    # Plot 1: Final theta comparison
    plot_final_theta_comparison(all_histories)
    
    # Plot 2: Evolution comparison
    plot_evolution_comparison(all_histories)

    plot_all_posteriors_detailed (all_histories=all_histories)
    
    # Print statistical summary
    # print_statistical_summary(all_histories)
# %%
if __name__ == '__main__':

    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
        input_paths = [s.strip() for s in input_paths if s.strip()]

     # Get the dataset input files:
    all_histories = {}

    # Set the theta values:
    theta_values = np.linspace (0.001, 0.999, 1000)

    # define loop parameters:
    window_size, num_chunks = 10, 50

    for idx, input_file in enumerate (input_paths):

        # Read the input files
        df = bayesian.load_wh_data (filepath=input_file)

        # Create binary states
        df = bayesian.create_binary_states (df=df, threshold=0.5)
        
        # Run the bayesian implementation:
        history = sequential_bayesian_implementation (theta_values=theta_values,
                                                      df=df, num_chunks=num_chunks, window_size=window_size
                                                      )
        bldg_id = input_file.split('/')[-3]

        all_histories [bldg_id] = history
    
    plot_comparison_all_wh (all_histories=all_histories)