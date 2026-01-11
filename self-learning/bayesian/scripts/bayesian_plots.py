import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

def plot_final_theta_comparison(all_histories):
    """
    Compare final theta estimates across all water heaters
    """
    wh_names = list(all_histories.keys())
    final_thetas = [all_histories[wh]['mean'][-1] for wh in wh_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(wh_names)), final_thetas, color='steelblue', alpha=0.7)
    plt.axhline(y=np.mean(final_thetas), color='red', linestyle='--', 
                linewidth=2, label=f'Average {f"$\theta$"} = {np.mean(final_thetas):.4f}')
    
    plt.xlabel('WH Bldg ID', fontsize=12)
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
    Compare how theta evolves over chunks for all water heaters
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
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