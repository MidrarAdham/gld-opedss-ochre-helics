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

# def plot_method4_allocation(plan_file, allocation_file, summary_file):
#     """
#     Plot Method 4: Metered Feeder Allocation
    
#     Args:
#         data: path to method4_summary.csv
#         stat: path to method4_trials.csv
#     """
    
#     # Load data
#     summary_df = pd.read_csv(data)
#     trials_df = pd.read_csv(stat)
    
#     # Create figure with 4 subplots
#     fig = plt.figure(figsize=(16, 12))
#     gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
#     ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
#     ax2 = fig.add_subplot(gs[1, 0])  # Middle left
#     ax3 = fig.add_subplot(gs[1, 1])  # Middle right
#     ax4 = fig.add_subplot(gs[2, :])  # Bottom row, full width
    
#     # Extract data
#     metered_kw = trials_df['metered_demand_kw']
#     util = trials_df['utilization_factor']
#     alloc_factor = trials_df['allocation_factor_kw_per_kva']
    
#     mean_kw = summary_df['metered_kw_mean'].iloc[0]
#     std_kw = summary_df['metered_kw_std'].iloc[0]
#     p95_kw = summary_df['metered_kw_p95'].iloc[0]
#     mean_util = summary_df['util_mean'].iloc[0]
#     p95_util = summary_df['util_p95'].iloc[0]
    
#     # Plot 1: Metered Demand Distribution
#     ax1.hist(metered_kw, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
#     ax1.axvline(mean_kw, color='red', linestyle='--', linewidth=2, 
#                 label=f'Mean: {mean_kw:.2f} kW')
#     ax1.axvline(p95_kw, color='orange', linestyle='--', linewidth=2, 
#                 label=f'95th percentile: {p95_kw:.2f} kW')
    
#     ax1.set_xlabel('Metered Demand (kW)', fontweight='bold')
#     ax1.set_ylabel('Frequency', fontweight='bold')
#     ax1.set_title(f'Metered Demand Distribution ({trials_df["n_total_customers"].iloc[0]} customers, {len(trials_df)} trials)\n'
#                   f'Mean: {mean_kw:.2f} {r"$\pm$"} {std_kw:.2f} kW',
#                   fontweight='bold')
#     ax1.legend()
#     ax1.grid(axis='y', alpha=0.3)
    
#     # Plot 2: Utilization Factor Distribution
#     ax2.hist(util, bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
#     ax2.axvline(mean_util, color='red', linestyle='--', linewidth=2, 
#                 label=f'Mean: {mean_util:.2%}')
#     ax2.axvline(1.0, color='darkred', linestyle='-', linewidth=2, 
#                 label='100% Limit')
#     ax2.axvline(p95_util, color='orange', linestyle='--', linewidth=2, 
#                 label=f'95th: {p95_util:.2%}')
    
#     ax2.set_xlabel('Utilization Factor', fontweight='bold')
#     ax2.set_ylabel('Frequency', fontweight='bold')
#     ax2.set_title('System Utilization Distribution', fontweight='bold')
#     ax2.legend()
#     ax2.grid(axis='y', alpha=0.3)
    
#     # Plot 3: Allocation Factor Distribution
#     mean_af = alloc_factor.mean()
    
#     ax3.hist(alloc_factor, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
#     ax3.axvline(mean_af, color='red', linestyle='--', linewidth=2, 
#                 label=f'Mean: {mean_af:.4f}')
    
#     ax3.set_xlabel('Allocation Factor (kW/kVA)', fontweight='bold')
#     ax3.set_ylabel('Frequency', fontweight='bold')
#     ax3.set_title('Allocation Factor Distribution', fontweight='bold')
#     ax3.legend()
#     ax3.grid(axis='y', alpha=0.3)
    
#     # Plot 4: Metered Demand Across Trials
#     ax4.plot(trials_df['trial'], metered_kw, marker='o', linestyle='-', 
#              color='steelblue', label='Metered Demand (kW)')
#     ax4.axhline(mean_kw, color='red', linestyle='--', linewidth=1.5, 
#                 label=f'Mean: {mean_kw:.2f} kW')
#     ax4.fill_between(trials_df['trial'], mean_kw - std_kw, mean_kw + std_kw,
#                      alpha=0.2, color='red', label=f'{r"$\pm$"}1 Std Dev')
    
#     ax4.set_xlabel('Trial Number', fontweight='bold')
#     ax4.set_ylabel('Metered Demand (kW)', fontweight='bold')
#     ax4.set_title('Metered Demand Across Trials', fontweight='bold')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.suptitle('Method 4: Metered Feeder Maximum Demand', 
#                  fontsize=15, fontweight='bold')
    
#     plt.savefig('./method4.png', dpi=300, bbox_inches='tight')
#     plt.show()

def plot_method4_allocation(plan_file, allocation_file, summary_file):
    """
    Plot Method 4: Metered Feeder Allocation with Cost Optimization
    
    Args:
        plan_file: path to method4_transformer_plan.csv
        allocation_file: path to method4_allocation_results.csv
        summary_file: path to method4_summary.csv
    """
    
    # Load data
    plan_df = pd.read_csv(plan_file)
    allocation_df = pd.read_csv(allocation_file)
    summary_df = pd.read_csv(summary_file)
    
    # Create figure with 5 subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Top left - Pie chart
    ax2 = fig.add_subplot(gs[0, 1])  # Top right - Capacity
    ax3 = fig.add_subplot(gs[1, 0])  # Middle left - Cost breakdown
    ax4 = fig.add_subplot(gs[1, 1])  # Middle right - Optimization metrics
    ax5 = fig.add_subplot(gs[2, :])  # Bottom - Allocations
    
    # Extract key values
    n_25 = plan_df['n_25kva'].iloc[0]
    n_50 = plan_df['n_50kva'].iloc[0]
    n_75 = plan_df['n_75kva'].iloc[0]
    
    required_kva = plan_df['required_installed_kva'].iloc[0]
    installed_kva = plan_df['installed_kva'].iloc[0]
    overbuild_kva = plan_df['overbuild_kva'].iloc[0]
    proxy_cost = plan_df['proxy_cost'].iloc[0]
    metered_kw = plan_df['metered_demand_kw'].iloc[0]
    pf = plan_df['pf'].iloc[0]
    uf_target = plan_df['uf_target'].iloc[0]
    
    # Parse weights from string if needed
    weights_str = plan_df['weights'].iloc[0]
    if isinstance(weights_str, str):
        import ast
        weights = ast.literal_eval(weights_str)
    else:
        weights = {25.0: 1.0, 50.0: 2.2, 75.0: 3.6}  # Default
    
    # ============================================================
    # Plot 1: Transformer Mix (Pie Chart)
    # ============================================================
    
    sizes = [n_25, n_50, n_75]
    labels = [f'{n_25}× 25 kVA', f'{n_50}× 50 kVA', f'{n_75}× 75 kVA']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    explode = (0.05, 0.05, 0.05)
    
    # Only plot non-zero values
    non_zero_sizes = [s for s in sizes if s > 0]
    non_zero_labels = [l for l, s in zip(labels, sizes) if s > 0]
    non_zero_colors = [c for c, s in zip(colors, sizes) if s > 0]
    non_zero_explode = [e for e, s in zip(explode, sizes) if s > 0]
    
    ax1.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
            autopct='%1.1f%%', explode=non_zero_explode, shadow=True, startangle=90)
    ax1.set_title(f'Transformer Mix\n(Total: {n_25 + n_50 + n_75} transformers)', 
                  fontweight='bold', fontsize=11)
    
    # ============================================================
    # Plot 2: Capacity Overview (Bar Chart)
    # ============================================================
    
    categories = ['Required\nkVA', 'Installed\nkVA']
    values = [required_kva, installed_kva]
    colors_bar = ['orange', 'green']
    
    bars = ax2.bar(categories, values, color=colors_bar, edgecolor='black', 
                   linewidth=2, alpha=0.7)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} kVA',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.text(1, installed_kva * 0.5, 
             f'Overbuild:\n{overbuild_kva:.1f} kVA\n({overbuild_kva/required_kva*100:.1f}%)',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_ylabel('Capacity (kVA)', fontweight='bold', fontsize=10)
    ax2.set_title('Capacity Planning', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================
    # Plot 3: Cost Breakdown (Stacked Bar)
    # ============================================================
    
    cost_25 = n_25 * weights[25.0]
    cost_50 = n_50 * weights[50.0]
    cost_75 = n_75 * weights[75.0]
    
    ax3.bar('Total', cost_25, color='lightcoral', label='25 kVA', edgecolor='black')
    ax3.bar('Total', cost_50, bottom=cost_25, color='lightblue', label='50 kVA', edgecolor='black')
    ax3.bar('Total', cost_75, bottom=cost_25+cost_50, color='lightgreen', label='75 kVA', edgecolor='black')
    
    ax3.set_ylabel('Proxy Cost (weighted units)', fontweight='bold', fontsize=10)
    ax3.set_title(f'Cost Breakdown\nTotal Proxy Cost: {proxy_cost:.2f}', 
                  fontweight='bold', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim([0, proxy_cost * 1.1])
    
    # Add cost weights as text
    weights_text = f"Weights:\n25 kVA: {weights[25.0]:.1f}\n50 kVA: {weights[50.0]:.1f}\n75 kVA: {weights[75.0]:.1f}"
    ax3.text(0.5, proxy_cost * 0.5, weights_text,
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ============================================================
    # Plot 4: Optimization Metrics (Table-style)
    # ============================================================
    
    ax4.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Metered Demand', f'{metered_kw:.2f} kW'],
        ['Feeder Peak', f'{metered_kw/pf:.2f} kVA'],
        ['Target Utilization', f'{uf_target*100:.0f}%'],
        ['Required Capacity', f'{required_kva:.2f} kVA'],
        ['Installed Capacity', f'{installed_kva:.2f} kVA'],
        ['Overbuild', f'{overbuild_kva:.2f} kVA ({overbuild_kva/required_kva*100:.1f}%)'],
        ['Proxy Cost', f'{proxy_cost:.2f}'],
        ['Total Transformers', f'{n_25 + n_50 + n_75}'],
        ['Actual Utilization', f'{(metered_kw/pf)/installed_kva*100:.1f}%']
    ]
    
    table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax4.set_title('Optimization Metrics', fontweight='bold', fontsize=11, pad=20)
    
    # ============================================================
    # Plot 5: Individual Transformer Allocations
    # ============================================================
    
    transformer_ids = allocation_df['transformer_id'].tolist()
    allocated_kw = allocation_df['allocated_kw'].tolist()
    utilizations = allocation_df['utilization_factor'].tolist()
    kva_ratings = allocation_df['kva_rating'].tolist()
    
    bar_colors = ['lightcoral' if kva == 25.0 else 'lightblue' if kva == 50.0 else 'lightgreen' 
                  for kva in kva_ratings]
    
    bars5 = ax5.bar(range(len(transformer_ids)), allocated_kw, color=bar_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax5.set_xlabel('Transformer', fontweight='bold', fontsize=10)
    ax5.set_ylabel('Allocated Load (kW)', fontweight='bold', fontsize=10)
    ax5.set_title(f'Load Allocation per Transformer | '
                  f'Allocation Factor: {metered_kw/installed_kva:.4f} kW/kVA',
                  fontweight='bold', fontsize=11)
    ax5.set_xticks(range(len(transformer_ids)))
    ax5.set_xticklabels(transformer_ids, rotation=45, ha='right', fontsize=7)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add utilization labels
    for i, (bar, util) in enumerate(zip(bars5, utilizations)):
        height = bar.get_height()
        color = 'green' if util <= 1.0 else 'red'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{util*100:.0f}%',
                ha='center', va='bottom', fontsize=6, color=color, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='75 kVA'),
        Patch(facecolor='lightblue', edgecolor='black', label='50 kVA'),
        Patch(facecolor='lightcoral', edgecolor='black', label='25 kVA')
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Overall title
    plt.suptitle(f'Method 4: Optimized Metered Feeder Allocation\n'
                 f'Cost-Optimized Configuration for {summary_df["total_customers_simulated"].iloc[0]} Customers',
                 fontsize=14, fontweight='bold')
    
    plt.savefig('./method4.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # Print Summary
    # ============================================================
    print("\n" + "="*70)
    print("METHOD 4: COST-OPTIMIZED FEEDER ALLOCATION")
    print("="*70)
    print(f"Customers: {summary_df['total_customers_simulated'].iloc[0]}")
    print(f"Metered Demand: {metered_kw:.2f} kW")
    print(f"\nOptimization Objective: Minimize (Overbuild, Cost, Count)")
    print(f"  Cost Weights: 25kVA={weights[25.0]:.1f}, 50kVA={weights[50.0]:.1f}, 75kVA={weights[75.0]:.1f}")
    print(f"\nOptimal Configuration:")
    print(f"  {n_75}× 75 kVA (cost: {cost_75:.2f})")
    print(f"  {n_50}× 50 kVA (cost: {cost_50:.2f})")
    print(f"  {n_25}× 25 kVA (cost: {cost_25:.2f})")
    print(f"  Total: {n_25 + n_50 + n_75} transformers")
    print(f"  Proxy Cost: {proxy_cost:.2f}")
    print(f"\nCapacity:")
    print(f"  Required: {required_kva:.2f} kVA")
    print(f"  Installed: {installed_kva:.2f} kVA")
    print(f"  Overbuild: {overbuild_kva:.2f} kVA ({overbuild_kva/required_kva*100:.1f}%)")
    print(f"\nActual Utilization: {(metered_kw/pf)/installed_kva*100:.1f}%")
    print("="*70)

filename = check_file (method='method4')



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
plan_file = [data for data in filename if "transformer_plan" in data and data.endswith('csv')]
allocation_file = [data for data in filename if "allocation_results" in data and data.endswith('csv')]
summary_file = [data for data in filename if "summary" in data and data.endswith('csv')]

plot_method4_allocation (plan_file[0], allocation_file[0], summary_file[0])