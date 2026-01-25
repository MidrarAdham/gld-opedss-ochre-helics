# %%
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp
import matplotlib.ticker as ticker
# My functions
import bayesian_plots as my_vis
import bayesian_experiment as my_bayesian
import sequential_bayesian as my_seq_bayesian

# %%
def collect_profiles_in_one_df(input_paths):
    x = []

    for input_path in input_paths:
        bldg_id = input_path.split('/')[-3]
        df = my_bayesian.load_wh_data(filepath=input_path)
        df = df.head(1440)

        time_col = df['Time']
        df = df.drop('Time', axis=1)

        # Rename columns to include building ID
        df.columns = [f"bldg_{bldg_id}_{col}" for col in df.columns]

        x.append(df)

    dfs = pd.concat(x, axis=1)
    dfs['Time'] = pd.to_datetime(time_col)

    return dfs

def calculate_diversified_demand (dfs):
    time_col = dfs['Time']
    dfs = dfs.drop('Time',axis=1)

    row_sums = dfs.sum (axis=1)
    dfs['diversified_demand (kVA)'] = row_sums
    dfs['Time'] = time_col.to_list()
    return dfs

def plot_dfs (df : pd.DataFrame, column : str):

    sns.set_theme (style='darkgrid', context='notebook', color_codes = True)
    fig, ax  = plt.subplots (figsize=(16,10))

    ax.plot (df['Time'], df[column], color = 'tab:gray', linewidth = 2, marker='o')

    ax.xaxis.set_major_locator (ticker.MaxNLocator(10))

    plt.show()

def calculate_apparent_power (input_paths : list[str]) -> list[pd.DataFrame]:
    dfs = []
    bldg_ids = []
    for input_path in input_paths:
        bldg_id = input_path.split('/')[-3]
        bldg_ids.append(bldg_id)
        cols = ['Time', 'Total Electric Power (kW)', 'Total Reactive Power (kVAR)', 'Water Heating Electric Power (kW)']
        df = pd.read_csv(input_path, usecols=cols)
        # cols = ['Time', f'{bldg_id} Total Electric Power (kW)', f'{bldg_id} Total Reactive Power (kVAR)', f'{bldg_id} Water Heating Electric Power (kW)']
        df = df.rename (columns= {'Total Electric Power (kW)' : f'{bldg_id} Total Electric Power (kW)',
                                  'Total Reactive Power (kVAR)' : f'{bldg_id} Total Reactive Power (kVAR)',
                                  'Water Heating Electric Power (kW)' : f'{bldg_id} Water Heating Electric Power (kW)'}
                                  )
        df = df.head(1440)
        df[f'{bldg_id} kva'] = np.sqrt(df[f'{bldg_id} Total Electric Power (kW)'].pow(2) + df[f'{bldg_id} Total Reactive Power (kVAR)'].pow(2))
        dfs.append(df)
    
    return dfs, bldg_ids

def smallest_size_for_peak(peak_kva):
    if peak_kva <= limits[25.0] + 1e-9:
        return 25.0
    if peak_kva <= limits[50.0] + 1e-9:
        return 50.0
    if peak_kva <= limits[75.0] + 1e-9:
        return 75.0
    return None

def pack_bucket(dfs, cols):
    remaining = cols.copy()
    txs = []
    tx_idx = 0

    while remaining:
        s = pd.Series(0.0, index=dfs.index)
        members, skipped = [], []

        for col in remaining:
            s_new = s + dfs[col]
            if s_new.max() <= limits[75.0] + 1e-9:
                s = s_new
                members.append(col)
            else:
                skipped.append(col)

        if not members:
            bad = remaining[0]
            raise ValueError(f"Can't place '{bad}' in 75kVA@80%. Check its peak or NaNs.")

        peak = float(s.max())
        size = smallest_size_for_peak(peak)

        txs.append({
            "tx_in_bucket": f"T{tx_idx:02d}",
            "size": size,
            "peak_div_kva": peak,
            "members": members
        })
        tx_idx += 1
        remaining = skipped

    return txs

def random_buckets(cols, bucket_size=8, seed=0):
    cols = cols.copy()
    random.Random(seed).shuffle(cols)
    return [cols[i:i+bucket_size] for i in range(0, len(cols), bucket_size)]

# %%
if __name__ == "__main__":
    with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
        input_paths = f.readlines ()
    
    input_paths = [s.strip() for s in input_paths if s.strip()]
    
    dfs, bldg_ids = calculate_apparent_power (input_paths=input_paths)

    allocated_loads = {'25kva': [],'50kva': [],'75kva': []}

    time_col = (dfs[0])['Time']

    dfs_conca = []


    for df in dfs:

        df = df.drop ('Time', axis=1)
        dfs_conca.append(df)

    
    dfs = pd.concat (dfs_conca, axis=1)

    dfs['Time'] = time_col.to_list()
    time_col = pd.to_datetime(dfs['Time']).dt.strftime('%H:%M')

    kva_cols = [col for col in dfs.columns if col.endswith('kva')]
    watts_cols = [col for col in dfs.columns if col.endswith('(kW)')]

    peaks = dfs[kva_cols].max()
    

    loading = 0.8
    limits = {25.0: 25*loading, 50.0: 50*loading, 75.0: 75*loading}  # 20/40/60

    # Play around with bucket size. A smaller bucket size priotrize smaller transformers
    bucket_size = 4
    seed = 7

    buckets = random_buckets(kva_cols, bucket_size=bucket_size, seed=seed)

    all_txs = []
    for b_id, cols in enumerate(buckets):
        txs = pack_bucket(dfs, cols)
        for tx in txs:
            tx["bucket_id"] = b_id
        all_txs.extend(txs)

    counts = pd.Series([t["size"] for t in all_txs]).value_counts().sort_index()
    

    kwatts_peak_div = dfs[watts_cols].sum(axis=1).max()
total_transformer_size = sum(t["size"] for t in all_txs)
AF_sys = kwatts_peak_div / total_transformer_size

df_txs = pd.DataFrame(all_txs)
grouper = df_txs.groupby(by='bucket_id')

sns.set_theme(style='darkgrid', context='notebook', color_codes=True, font_scale=1.0)

for bucket_id, bucket_txs in grouper:
    n_txs_in_bucket = len(bucket_txs)
    
    if n_txs_in_bucket == 1:
        ncols, nrows = 1, 1
    elif n_txs_in_bucket <= 2:
        ncols, nrows = 2, 1
    elif n_txs_in_bucket <= 4:
        ncols, nrows = 2, 2
    else:
        ncols = 3
        nrows = math.ceil(n_txs_in_bucket / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    if n_txs_in_bucket == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    bucket_total_capacity = bucket_txs['size'].sum()
    bucket_all_members = [m for sublist in bucket_txs['members'].tolist() for m in sublist]
    bucket_total_load = dfs[bucket_all_members].sum(axis=1)
    bucket_peak = bucket_total_load.max()

    for i, (_, tx_info) in enumerate(bucket_txs.iterrows()):
        ax = axes[i]
        
        tx_load_kva = dfs[tx_info['members']].sum(axis=1)
        tx_size = tx_info['size']
        tx_capacity = tx_size * loading
        peak_kva = tx_load_kva.max()
        utilization = (peak_kva / tx_capacity) * 100
        allocated_kw = AF_sys * tx_size
        
        ax.plot(time_col, tx_load_kva, linewidth=2, color='tab:blue', label='Demand')
        ax.axhline(y=tx_capacity, color='r', linestyle='--', linewidth=1.5, 
                   label=f'Capacity @ {loading*100:.0f}%')
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
        ax.set_title(f'{tx_info["tx_in_bucket"]} ({tx_size:.0f} kVA)\n'
                     f'{len(tx_info["members"])} buildings | Peak: {peak_kva:.1f} kVA | '
                     f'Util: {utilization:.1f}%', fontsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel('Apparent Power [kVA]')
        ax.set_xlabel('Time [hh:mm]')
        
        stats_text = (f"Allocated: {allocated_kw:.1f} kW\n"
                      f"AF_sys: {AF_sys:.3f}\n"
                      f"Peak: {peak_kva:.1f} kVA")
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    for j in range(n_txs_in_bucket, len(axes)):
        axes[j].axis('off')
    
    # Add overall figure title with bucket summary
    fig.suptitle(f'Bucket {bucket_id} Summary: {n_txs_in_bucket} Transformers | '
                 f'Total Capacity: {bucket_total_capacity:.0f} kVA | '
                 f'Peak Demand: {bucket_peak:.1f} kVA',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure for this bucket
    # plt.savefig(f'../results/bucket_{bucket_id}_transformers.png', dpi=300, bbox_inches='tight')
    # print(f"Saved: {filename}")
    
    # plt.show()

    fig, ax = plt.subplots(figsize=(14, 6))
    total_feeder_kva = dfs[kva_cols].sum(axis=1)
    ax.plot(time_col, total_feeder_kva, linewidth=2, color='navy', label='Total Feeder Demand')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.set_title('Total Feeder Diversified Demand', fontsize=14, fontweight='bold')
    ax.set_ylabel('Apparent Power [kVA]')
    ax.set_xlabel('Time [hh:mm]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/feeder_demand_profile.png', dpi=300, bbox_inches='tight')
    