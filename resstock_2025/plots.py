import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import load_allocations_api as api

def check_file (method : str):
    file_dir = './results/'
    for files in os.listdir (file_dir):
        if method in files:
            return pd.read_csv (file_dir+files)
        else:
            return False

def plot_method1_results (results):
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

def load_survey_plot (kw, kwh, regr_results):
    a = regr_results['intercept_a']
    b = regr_results['slope_b']

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
    
    # Add R² to plot
    plt.text(0.05, 0.95, f"R² = {regr_results['r_squared']:.3f}", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    # plt.savefig('./method.png')
    plt.show()

def transformer_load_management_plot (regr_results, data_results):
    fig, axes = plt.subplots (1, 3, figsize = (18, 10))

    for idx, kva in enumerate(regr_results):
        ax = axes[idx]

        kwh_list = data_results[kva]['transformer_kwh']
        kw_list = data_results[kva]['max_diversified_kw']

        a = regr_results[kva]['intercept']
        b = regr_results[kva]['slope']
        r2 = regr_results[kva]['r_squared']
        n_customers = regr_results[kva]['n_customers']

        # Stats:
        stats = regr_results[kva]['statistics']
        pred_error = stats['residual_std']

        ax.scatter(kwh_list, kw_list, alpha=0.6, s=50, label='Actual Data')

        kwh_range = np.linspace(min(kwh_list), max(kwh_list), 100)
        kw_peak = a + b * kwh_range

        ax.plot(kwh_range, kw_peak, 'r-', linewidth=2, label=f'kW = {a:.2f} + {b:.3f} x kWh')
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
        
        stats_text += f"kWh: {stats['kwh_mean']:.0f} {r"$\pm$"} {stats['kwh_std']:.0f}\n"
        stats_text += f"kW: {stats['kw_mean']:.1f} {r"$\pm$"} {stats['kw_std']:.1f}"


        ax.text(0.05, 0.95, stats_text, 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.suptitle('Method 3: Transformer Load Management', fontsize=14)
    plt.tight_layout()
    # plt.savefig('./method3.png')
    plt.show()

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

data = check_file (method='method1')
# ================= Diversified Peak Method ==================
plot_method1_results (results=data)
# ================= Diversified Peak Method ==================
# ------------------------------------------------------------
# ==================== Load Survey Method ====================
dataset_dir = f"{os.getcwd()}/datasets/cosimulation/"
# kw, kwh = api.method2_load_survey (dataset_dir=dataset_dir)
# regr_results = api.linear_regr(kwh= kwh, kw=kw)
# load_survey_plot (kw=kw, kwh=kwh, regr_results=regr_results)
# ==================== Load Survey Method ====================
# ------------------------------------------------------------
# ================ Transformer Load Management ===============
results = api.method3_transformer_load_management (dataset_dir=dataset_dir)
# regression_res = api.method3_regr (results=results)
# transformer_load_management_plot (regr_results=regression_res, data_results=results)
# ================ Transformer Load Management ===============
# ------------------------------------------------------------
# ================ Metered Feeder Max. Demand ================
# results = api.method4_metered_feeder_max_demand(dataset_dir=dataset_dir, n_total_customers=300)
# plot_method4_allocation (results=results)