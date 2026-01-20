import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_likelihood (theta_values, likelihood, H, n):
    plt.figure (figsize=(10, 6))
    plt.plot (theta_values, likelihood, linewidth=2,
              label=f'P(X | theta) = ({n} choose {H}) * theta ^ {H} * (1-theta)^{n-H}')
    plt.xlabel ('theta (Probability of ON)', fontsize=12)
    plt.ylabel ('P(X | theta)', fontsize=12)
    plt.title ('Likelihood function', fontsize=14, fontweight='bold')
    plt.axvline (x=H/n, color='red', linestyle='--',label=f'Max at theta = {H/n:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    return plt

def plot_bayesian (theta_values, prior, likelihood, posterior, H, T, n, stats : dict):
    
    fig, axes = plt.subplots (3, 1, figsize=(10,12))
    axes[0].plot(theta_values, prior, linewidth=2, color='blue')
    axes[0].set_xlim(0,1)
    axes[0].set_title('Prior: P(θ)', fontweight='bold')
    axes[0].set_xlabel('θ')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    axes[1].plot(theta_values, likelihood, linewidth=2, color='green')
    axes[1].set_title(f'Likelihood: P(X|theta) where X = {H} ONs, {T} OFFs', 
                      fontweight='bold')
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel('theta')
    axes[1].set_ylabel('P(X|theta)')
    axes[1].axvline(x=H/n, color='red', linestyle='--', label=f'MLE = {H/n:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    axes[2].plot(theta_values, posterior, linewidth=2, color='red')
    axes[2].set_xlim(0,1)
    axes[2].set_title('Posterior: P(θ|X)', fontweight='bold')
    axes[2].set_xlabel('θ')
    axes[2].set_ylabel('Density')
    axes[2].plot ([stats['ci_lower'], stats['ci_upper']], [0,0], linewidth=8,
                  marker='|', markersize=15,
                  label=f"Confidence Interval={stats['ci_width']:.3f}\nstd={stats['std']:.3f}")

    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    return plt

def plot_evolution_comparison(all_histories):
    """
    Compare how theta evolves over chunks for all water heaters
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    # print(all_histories['32387']['chunk'])
    
    # from pprint import pprint as pp

    # pp(all_histories)
    # quit()

    for idx, (wh_name, history) in enumerate(all_histories.items()):
        chunks = history['chunk']

        
        # Mean evolution
        axes[0].plot(chunks, history['mean'], 
                    linewidth=2, marker='o', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
        
        # Uncertainty evolution
        axes[1].plot(chunks, history['std'], 
                    linewidth=2, marker='s', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
        
        # Credible interval width
        ci_width = np.array(history['ci_upper']) - np.array(history['ci_lower'])
        axes[2].plot(chunks, ci_width, 
                    linewidth=2, marker='^', markersize=4,
                    color=colors[idx], label=wh_name, alpha=0.7)
    
    # Formatting shit
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

def plot_theta_by_hour(history):
    """
    Plot theta estimate for each hour of day
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history)))
    
    for idx, (wh_name, history) in enumerate(history.items()):
        hours = history['hour']
        theta_mean = history['mean']
        
        ax.plot(hours, theta_mean, marker='o', linewidth=2,
                markersize=6, color=colors[idx], 
                label=wh_name, alpha=0.7)
    
    ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{r"$\theta$"} (Probability of ON)', fontsize=14, fontweight='bold')
    ax.set_title(f'Time-Varying {r"$\theta$"}: Comparison Across Water Heaters', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
    ax.legend(loc='best', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/theta_comparison_all_wh.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_posterior_evolution_by_hour(all_history, theta_values):
    """
    Show how posterior distribution changes throughou the day
    """
    n_wh = len(all_history)
    n_cols = 3
    n_rows = (n_wh + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_wh > 1 else [axes]
    
    # Color map for different hours of the daty
    colors = plt.cm.twilight(np.linspace(0, 1, 24))

    for wh_idx, (wh_name, history) in enumerate(all_history.items()):
        ax = axes[wh_idx]
        
        # Plot each hour's posterior for this WH
        for hour_idx, hour in enumerate(history['hour']):
            posterior_pdf = history['posterior'][hour_idx]
            theta_mean = history['mean'][hour_idx]
            
            ax.plot(theta_values, posterior_pdf, 
                    linewidth=1.5, color=colors[hour_idx],
                    alpha=0.6)
        
        ax.set_title(f'{wh_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel(r'$\theta$', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_xlim(0, 0.3)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_wh, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Posterior Distributions by Hour: All Water Heaters', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../results/posteriors_by_hour_all_wh.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_data_availability(all_history):
    """
    Show data availability averaged across all water heaters. THIS IS A USELESS PLOT
    
    :param all_history: Dict of {wh_name: history_dict}
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Aggregate data across all WHs
    n_wh = len(all_history)
    avg_n_total = np.zeros(24)
    avg_H_total = np.zeros(24)
    avg_T_total = np.zeros(24)
    
    for wh_name, history in all_history.items():
        avg_n_total += np.array(history['n_total'])
        avg_H_total += np.array(history['H_total'])
        avg_T_total += np.array(history['T_total'])
    
    avg_n_total /= n_wh
    avg_H_total /= n_wh
    avg_T_total /= n_wh
    
    hours = range(24)
    
    # Total observations per hour (average)
    axes[0].bar(hours, avg_n_total, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Avg Total Observations', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Average Data Availability by Hour (n={n_wh} WHs)', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # ON vs OFF counts (average)
    axes[1].bar(hours, avg_H_total, label='Avg ONs', 
               color='orange', alpha=0.7)
    axes[1].bar(hours, avg_T_total, bottom=avg_H_total,
               label='Avg OFFs', color='lightblue', alpha=0.7)
    axes[1].set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Avg Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Average ON vs OFF Counts by Hour', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../results/data_availability_by_hour.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_uncertainty_by_hour(all_history):
    """
    Show uncertainty for all water heaters across hours
    
    :param all_history: Dict of {wh_name: history_dict}
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_history)))
    
    for idx, (wh_name, history) in enumerate(all_history.items()):
        hours = history['hour']
        std_values = history['std']
        ci_width = np.array(history['ci_upper']) - np.array(history['ci_lower'])
        
        # STD
        axes[0].plot(hours, std_values, marker='o', linewidth=2,
                    markersize=4, color=colors[idx], 
                    label=wh_name, alpha=0.7)
        
        # CI width
        axes[1].plot(hours, ci_width, marker='s', linewidth=2,
                    markersize=4, color=colors[idx], 
                    label=wh_name, alpha=0.7)
    
    # Formatting
    axes[0].set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    axes[0].set_title('Std Deviation by Hour: All Water Heaters', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
    axes[0].legend(loc='best', fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('95% CI Width', fontsize=12, fontweight='bold')
    axes[1].set_title('95% CI Width by Hour: All Water Heaters', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
    axes[1].legend(loc='best', fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/uncertainty_by_hour_all_wh.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_heterogeneity_distribution(theta_values, fitted_params):
    """
    Plot histogram of observed thetas with fitted Beta distribution
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    theta_array = np.array(theta_values)
    theta_range = np.linspace(0.001, 0.999, 1000)
    
    # Extract fitted parameters
    alpha_mle = fitted_params['alpha_mle']
    beta_mle = fitted_params['beta_mle']
    
    # first polt: Histogram + PDF
    axes[0].hist(theta_array, bins=20, density=True, 
                alpha=0.6, color='steelblue', edgecolor='black',
                label='Observed θ values')
    
    # Fittted Beta distribution
    fitted_pdf = beta.pdf(theta_range, alpha_mle, beta_mle)
    axes[0].plot(theta_range, fitted_pdf, 'r-', linewidth=3,
                label=f'Fitted: Beta({r"$\alpha$"}={alpha_mle:.2f}, {r"$\beta$"}={beta_mle:.2f})')
    
    mean_theta = alpha_mle / (alpha_mle + beta_mle)
    axes[0].axvline(mean_theta, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {mean_theta:.4f}')
    
    axes[0].set_xlabel(r'$\theta$ (Probability of ON)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    axes[0].set_title('Heterogeneity in Water Heater Behavior', 
                     fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Empirical CDF vs Fitted CDF
    sorted_theta = np.sort(theta_array)
    empirical_cdf = np.arange(1, len(sorted_theta) + 1) / len(sorted_theta)
    fitted_cdf = beta.cdf(sorted_theta, alpha_mle, beta_mle)
    
    axes[1].plot(sorted_theta, empirical_cdf, 'o-', 
                linewidth=2, markersize=6, color='steelblue',
                label='Actual CDF', alpha=0.7)
    axes[1].plot(sorted_theta, fitted_cdf, 'r-', 
                linewidth=3, label='Fitted CDF', alpha=0.8)
    
    axes[1].set_xlabel(r'$\theta$ (Probability of ON)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    axes[1].set_title('Goodness of Fit: Actual vs Fitted CDF', 
                     fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/heterogeneity_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_qq_plot(theta_values, fitted_params):
    """
    Q-Q plot to assess goodness of fit
    """
    theta_array = np.array(theta_values)
    alpha_mle = fitted_params['alpha_mle']
    beta_mle = fitted_params['beta_mle']
    
    # Theoretical quantiles
    n = len(theta_array)
    theoretical_quantiles = beta.ppf(np.linspace(0.01, 0.99, n), 
                                          alpha_mle, beta_mle)
    
    # Empirical quantiles
    empirical_quantiles = np.sort(theta_array)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Q-Q plot
    ax.scatter(theoretical_quantiles, empirical_quantiles, 
              s=80, alpha=0.6, color='steelblue', edgecolors='black')
    
    # 45-degree line (perfect fit)
    min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
    max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='Perfect Fit')
    
    ax.set_xlabel('Theoretical Quantiles (Fitted Beta)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Quantiles (Observed)', 
                 fontsize=14, fontweight='bold')
    ax.set_title(f'Q-Q Plot: Beta({r"$\alpha$"}={alpha_mle:.2f}, β={beta_mle:.2f})', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/qq_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_percentiles(theta_values, fitted_params):
    """
    Analyze and visualize percentiles of the distribution
    """
    theta_array = np.array(theta_values)
    alpha_mle = fitted_params['alpha_mle']
    beta_mle = fitted_params['beta_mle']
    
    # Calculate percentiles
    percentiles = [0.001, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.999]
    empirical_pct = np.percentile(theta_array, percentiles)
    fitted_pct = beta.ppf(np.array(percentiles)/100, alpha_mle, beta_mle)

    for i, pct in enumerate(percentiles):
        diff = empirical_pct[i] - fitted_pct[i]
        print(f"{pct}th{'':<12} {empirical_pct[i]:<15.4f} "
              f"{fitted_pct[i]:<15.4f} {diff:<15.4f}")
    print(f"{'='*70}\n")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(percentiles))
    width = 0.35
    
    ax.bar(x_pos - width/2, empirical_pct, width, 
          label='Actual', color='steelblue', alpha=0.7)
    ax.bar(x_pos + width/2, fitted_pct, width,
          label='Fitted', color='orange', alpha=0.7)
    
    ax.set_xlabel('Percentile', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\theta$ Value', fontsize=14, fontweight='bold')
    ax.set_title('Percentile Comparison: Actual vs Fitted', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../results/percentile_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

