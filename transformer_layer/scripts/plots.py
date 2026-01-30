import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import load_allocations_api as api

def check_file (method : str):
    file_dir = './results/'
    input_paths = []

    for files in os.listdir (file_dir):
    
        if method in files:
            input_paths.append(file_dir+files)
    
    if input_paths:
        return input_paths

def plot_method1_results (filename):

    results = pd.read_csv (filename[0])
    # Use capsize to make the error bars less annoying
    # Maybe remove markers from the DF plot, they are not useful
    df_25 = results[results['kva'] == 25.0]
    df_50 = results[results['kva'] == 50.0]
    df_75 = results[results['kva'] == 75.0]

    fig, ax = plt.subplots (2,1, figsize = (12,10))

    ax[0].errorbar (
        df_25['n_customers'], df_25['avg_utilization'],
        yerr = df_25['std_utilization'], marker = 'o',
        label = '25 kVA', capsize=5
    )

    ax[0].errorbar(df_50['n_customers'], df_50['avg_utilization'], 
                   yerr=df_50['std_utilization'], 
                   marker='s', label='50 kVA', capsize=5)
    ax[0].errorbar(df_75['n_customers'], df_75['avg_utilization'], 
                   yerr=df_75['std_utilization'], 
                   marker='^', label='75 kVA', capsize=5)
    
    ax[0].axhline(y=1.0, color='r', linestyle='--', label='100% Utilization')
    ax[0].set_xlabel('Number of Customers')
    ax[0].set_ylabel('Utilization Factor')
    ax[0].set_title('Transformer Utilization vs Number of Customers')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Plot 2: Diversity Factor
    ax[1].plot(df_25['n_customers'], df_25['avg_diversity_factor'], 
               marker='o', label='25 kVA')
    ax[1].plot(df_50['n_customers'], df_50['avg_diversity_factor'], 
               marker='s', label='50 kVA')
    ax[1].plot(df_75['n_customers'], df_75['avg_diversity_factor'], 
               marker='^', label='75 kVA')
    
    ax[1].set_xlabel('Number of Customers')
    ax[1].set_ylabel('Diversity Factor')
    ax[1].set_title('Diversity Factor vs Number of Customers')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./method1.png')
    plt.show()

def load_survey_plot (data, regr_results):

    data_df = pd.read_csv (data)
    regr_df = pd.read_csv (regr_results)

    kwh = data_df['kwh'].to_list()
    kw = data_df['kw'].to_list()


    a = regr_df['intercept_a'].iloc[0]
    b = regr_df['slope_b'].iloc[0]
    r_squared = regr_df['r_squared'].iloc[0]


    kwh_range = np.linspace (min(kwh), max(kwh), 100)
    kw_peak = a + b * kwh_range

    plt.figure (figsize = (10,6))
    plt.scatter (kwh, kw, alpha=0.6, s=50, label='Actual Data')
    plt.plot (kwh_range, kw_peak, 'r-', linewidth=2, label=f'Regression: kW = {a:.2f} + {b:.2f} x kWh')
    plt.xlabel('Energy Consumption (kWh) - 24 hours')
    plt.ylabel('Maximum Demand (kW)')
    plt.title('Load Survey: Maximum Demand vs Energy Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f"R^2 = {r_squared:.3f}", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    # plt.savefig('./method.png')
    plt.show()

def transformer_load_management_plot (data, regr_results):
    # data_results = pd.read_csv(data)
    regr_results = pd.read_csv (regr_results)

    fig, axes = plt.subplots (1, 3, figsize = (18, 10))

    for idx, filename in enumerate(data):
        
        ax = axes[idx]

        df = pd.read_csv (filename)
        # print(df)
        # quit()

        kwh_list = df['transformer_kwh'].to_list()
        kw_list = df['max_diversified_kw'].to_list()
        kva = df['kva_rating'].iloc[0]

        regression_full_results = regr_results[regr_results['kva']==kva]

        a = regression_full_results['intercept'].iloc[0]
        b = regression_full_results['slope'].iloc[0]
        r2 = regression_full_results['r_squared'].iloc[0]
        n_customers = regression_full_results['n_customers'].iloc[0]
        pred_error = regression_full_results['residual_std'].iloc[0]

        ax.scatter (kwh_list, kw_list, alpha=0.6, s=50, label='actual data')

        kwh_range = np.linspace (min(kwh_list), max(kwh_list), 100)
        kw_peak = a + b * kwh_range

        ax.plot (kwh_range, kw_peak, 'r-', linewidth=2, label = f'kW = {a:.2f} + {b:.3f} x kWh')
        ax.fill_between (kwh_range,
                         kw_peak - pred_error,
                         kw_peak + pred_error,
                         alpha = 0.2,
                         color = 'red',
                         label = f'{r"$\pm$"}{pred_error:.1f} kW prediction error')
        
        ax.set_xlabel('Transformer Total kWh (24 hour)')
        ax.set_ylabel('Max Diversified Demand (kW)')
        ax.set_title (f'{kva} kVA Transformer ({n_customers} customers)')
        ax.legend ()
        ax.grid (True, alpha=0.3)

        stats_text = f"{r"$R^2$"} = {r2:.3f}\n"
        
        stats_text += f"kWh: {regression_full_results['kwh_mean'].iloc[0]:.0f} {r"$\pm$"} {regression_full_results['kwh_std'].iloc[0]:.0f}\n"
        stats_text += f"kW: {regression_full_results['kw_mean'].iloc[0]:.1f} {r"$\pm$"} {regression_full_results['kw_std'].iloc[0]:.1f}"


        # ax.text(0.05, 0.95, [regression_full_results.to_json()[0]],
        #         transform=ax.transAxes, 
        #         fontsize=9, 
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle ('Method 3: Transformer Load Management (TLM)', fontsize=14)
    plt.tight_layout()
    plt.show ()


def plot_method4_allocation(results):
    """
    Visualize Method 4: Metered Feeder Allocation
    
    Shows:
    1. Bar chart of allocated load per transformer
    2. Utilization factors
    3. Summary statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    allocations = results['transformer_allocations']
    
    # Extract data
    transformer_ids = [a['transformer_id'] for a in allocations]
    kva_ratings = [a['kva_rating'] for a in allocations]
    allocated_kw = [a['allocated_kw'] for a in allocations]
    utilizations = [a['utilization_factor'] for a in allocations]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ============================================================
    # Plot 1: Allocated Load per Transformer
    # ============================================================
    
    # Color by transformer size
    colors = ['lightcoral' if kva == 25.0 else 'lightblue' if kva == 50.0 else 'lightgreen' 
              for kva in kva_ratings]
    
    bars1 = ax1.bar(range(len(transformer_ids)), allocated_kw, color=colors, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Transformer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Allocated Load (kW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Method 4: Load Allocation ({results["n_transformers"]} Transformers, '
                  f'{results["total_customers"]} Customers)\n'
                  f'Metered Demand: {results["metered_demand_kw"]:.2f} kW | '
                  f'Allocation Factor: {results["allocation_factor"]:.4f}',
                  fontsize=13, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(transformer_ids)))
    ax1.set_xticklabels(transformer_ids, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, kw) in enumerate(zip(bars1, allocated_kw)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{kw:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='75 kVA'),
        Patch(facecolor='lightblue', edgecolor='black', label='50 kVA'),
        Patch(facecolor='lightcoral', edgecolor='black', label='25 kVA')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # ============================================================
    # Plot 2: Utilization Factors
    # ============================================================
    
    bars2 = ax2.bar(range(len(transformer_ids)), 
                    [u * 100 for u in utilizations],  # Convert to percentage
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add 100% reference line
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                label='100% Utilization Limit', zorder=5)
    
    ax2.set_xlabel('Transformer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Utilization Factor (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Transformer Utilization Factors', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(transformer_ids)))
    ax2.set_xticklabels(transformer_ids, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, max(110, max([u*100 for u in utilizations]) + 10)])
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add value labels on bars
    for bar, util in zip(bars2, utilizations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{util*100:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./method4.png')
    plt.show()
    
    # ============================================================
    # Print Summary Statistics
    # ============================================================
    # print("\n" + "="*60)
    # print("METHOD 4 SUMMARY STATISTICS")
    # print("="*60)
    # print(f"Total Customers: {results['total_customers']}")
    # print(f"Metered Demand: {results['metered_demand_kw']:.2f} kW")
    # print(f"Number of Transformers: {results['n_transformers']}")
    # print(f"Total Transformer Capacity: {results['total_transformer_kva']:.2f} kVA")
    # print(f"Allocation Factor: {results['allocation_factor']:.4f}")
    # print(f"Average Utilization: {np.mean(utilizations)*100:.2f}%")
    # print(f"All transformers within limits: {'YES ✓' if all(u <= 1.0 for u in utilizations) else 'NO ✗'}")
    # print("="*60)


def plot_method4_allocation(trials_file, clusters_file):
    """
    Plot Method 4: Cluster-Based Transformer Sizing
    
    Args:
        trials_file: path to method4_cluster_trials.csv
        clusters_file: path to method4_cluster_assignments.csv
    """
    
    # Load data
    trials_df = pd.read_csv(trials_file)
    clusters_df = pd.read_csv(clusters_file)
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Transformer mix distribution
    ax2 = fig.add_subplot(gs[0, 1])  # Cluster size distribution
    ax3 = fig.add_subplot(gs[1, 0])  # Installed capacity per trial
    ax4 = fig.add_subplot(gs[1, 1])  # Number of transformers per trial
    ax5 = fig.add_subplot(gs[2, 0])  # Cluster peak vs size (scatter)
    ax6 = fig.add_subplot(gs[2, 1])  # Transformer choice by cluster size
    
    # Get summary statistics
    mean_n25 = trials_df['n_25kva'].mean()
    mean_n50 = trials_df['n_50kva'].mean()
    mean_n75 = trials_df['n_75kva'].mean()
    mean_installed = trials_df['installed_kva_total'].mean()
    mean_clusters = trials_df['n_clusters'].mean()
    n_customers = trials_df['n_total_customers'].iloc[0]
    cluster_min = trials_df['cluster_min'].iloc[0]
    cluster_max = trials_df['cluster_max'].iloc[0]
    
    # ============================================================
    # Plot 1: Average Transformer Mix (Pie Chart)
    # ============================================================
    
    sizes = [mean_n25, mean_n50, mean_n75]
    labels = [f'25 kVA\n(avg: {mean_n25:.1f})', 
              f'50 kVA\n(avg: {mean_n50:.1f})', 
              f'75 kVA\n(avg: {mean_n75:.1f})']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    # Filter non-zero
    non_zero_sizes = [s for s in sizes if s > 0]
    non_zero_labels = [l for l, s in zip(labels, sizes) if s > 0]
    non_zero_colors = [c for c, s in zip(colors, sizes) if s > 0]
    
    ax1.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title(f'Average Transformer Mix\n(Avg: {mean_n25+mean_n50+mean_n75:.1f} transformers)', 
                  fontweight='bold', fontsize=11)
    
    # ============================================================
    # Plot 2: Cluster Size Distribution (Histogram)
    # ============================================================
    
    ax2.hist(clusters_df['cluster_size_houses'], bins=range(cluster_min, cluster_max+2),
             edgecolor='black', alpha=0.7, color='steelblue', align='left')
    ax2.set_xlabel('Houses per Cluster', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title(f'Cluster Size Distribution\n(Range: {cluster_min}-{cluster_max} houses)', 
                  fontweight='bold', fontsize=11)
    ax2.set_xticks(range(cluster_min, cluster_max+1))
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # Plot 3: Installed Capacity per Trial (Box + Violin)
    # ============================================================
    
    parts = ax3.violinplot([trials_df['installed_kva_total']], positions=[1],
                           showmeans=True, showmedians=True)
    
    ax3.axhline(mean_installed, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_installed:.1f} kVA')
    ax3.set_ylabel('Installed Capacity (kVA)', fontweight='bold')
    ax3.set_title(f'Installed Capacity Distribution\nAcross {len(trials_df)} Trials', 
                  fontweight='bold', fontsize=11)
    ax3.set_xticks([1])
    ax3.set_xticklabels(['All Trials'])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # Plot 4: Number of Transformers per Trial (Stacked Bar)
    # ============================================================
    
    trial_nums = trials_df['trial']
    
    ax4.bar(trial_nums, trials_df['n_25kva'], color='lightcoral', 
            label='25 kVA', edgecolor='black')
    ax4.bar(trial_nums, trials_df['n_50kva'], 
            bottom=trials_df['n_25kva'],
            color='lightblue', label='50 kVA', edgecolor='black')
    ax4.bar(trial_nums, trials_df['n_75kva'], 
            bottom=trials_df['n_25kva'] + trials_df['n_50kva'],
            color='lightgreen', label='75 kVA', edgecolor='black')
    
    ax4.set_xlabel('Trial Number', fontweight='bold')
    ax4.set_ylabel('Number of Transformers', fontweight='bold')
    ax4.set_title('Transformer Count by Trial', fontweight='bold', fontsize=11)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # Plot 5: Cluster Peak vs Cluster Size (Scatter)
    # ============================================================
    
    ax5.scatter(clusters_df['cluster_size_houses'], 
                clusters_df['cluster_peak_kw'],
                alpha=0.5, s=30, c='steelblue')
    
    ax5.set_xlabel('Cluster Size (houses)', fontweight='bold')
    ax5.set_ylabel('Cluster Peak Demand (kW)', fontweight='bold')
    ax5.set_title('Peak Demand vs Cluster Size', fontweight='bold', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Add trend line
    from scipy import stats as sp_stats
    slope, intercept, r_value, _, _ = sp_stats.linregress(
        clusters_df['cluster_size_houses'], 
        clusters_df['cluster_peak_kw']
    )
    x_line = np.array([cluster_min, cluster_max])
    y_line = slope * x_line + intercept
    ax5.plot(x_line, y_line, 'r--', linewidth=2, 
             label=f'Trend: y={slope:.2f}x+{intercept:.2f}')
    ax5.legend()
    
    # ============================================================
    # Plot 6: Transformer Choice by Cluster Size (Stacked Bar)
    # ============================================================
    
    # Count transformer choices by cluster size
    choice_data = []
    for size in range(cluster_min, cluster_max + 1):
        subset = clusters_df[clusters_df['cluster_size_houses'] == size]
        n_25 = (subset['chosen_transformer_kva'] == 25.0).sum()
        n_50 = (subset['chosen_transformer_kva'] == 50.0).sum()
        n_75 = (subset['chosen_transformer_kva'] == 75.0).sum()
        choice_data.append({'size': size, '25kVA': n_25, '50kVA': n_50, '75kVA': n_75})
    
    choice_df = pd.DataFrame(choice_data)
    x_pos = np.arange(len(choice_df))
    
    ax6.bar(x_pos, choice_df['25kVA'], color='lightcoral', 
            label='25 kVA', edgecolor='black')
    ax6.bar(x_pos, choice_df['50kVA'], bottom=choice_df['25kVA'],
            color='lightblue', label='50 kVA', edgecolor='black')
    ax6.bar(x_pos, choice_df['75kVA'], 
            bottom=choice_df['25kVA'] + choice_df['50kVA'],
            color='lightgreen', label='75 kVA', edgecolor='black')
    
    ax6.set_xlabel('Cluster Size (houses)', fontweight='bold')
    ax6.set_ylabel('Count', fontweight='bold')
    ax6.set_title('Transformer Selection by Cluster Size', fontweight='bold', fontsize=11)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(choice_df['size'])
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Overall title
    plt.suptitle(f'Method 4: Cluster-Based Transformer Sizing\n'
                 f'{n_customers} Customers | {len(trials_df)} Trials | '
                 f'Cluster Size: {cluster_min}-{cluster_max} houses',
                 fontsize=14, fontweight='bold')
    
    plt.savefig('./method4_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

# regr_results = None
# data = None

filename = check_file (method='method4')

# for f in filename:
#     if 'regression' in f:
#         regr_results = f
#     else:
#         data = f

# ================= Diversified Peak Method ==================
# plot_method1_results (filename=filename)
# ================= Diversified Peak Method ==================

# ------------------------------------------------------------

# ==================== Load Survey Method ====================
# load_survey_plot (data=data, regr_results=regr_results)
# ==================== Load Survey Method ====================

# ------------------------------------------------------------

# ================ Transformer Load Management ===============

# data = [data for data in filename if not "regression" in data]
# regr = [data for data in filename if "regression" in data]

# transformer_load_management_plot (data = data, regr_results=regr[0])
# ================ Transformer Load Management ===============
# ------------------------------------------------------------
# ================ Metered Feeder Max. Demand ================

# plot_method4_allocation (results=results)

filename = check_file (method='method4')
trials_file = [data for data in filename if "cluster_trials" in data and data.endswith('csv')]
clusters_file = [data for data in filename if "cluster_assignments" in data and data.endswith('csv')]


plot_method4_allocation (trials_file=trials_file[0], clusters_file=clusters_file[0])