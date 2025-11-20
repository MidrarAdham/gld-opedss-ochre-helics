#%%
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from pprint import pprint as pp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%%
def filter_datasets(dataset_path):
    '''
    This function looks into the ResStock datasets, checks buildings with no in.schedules.csv file,
    and ignore them. It also remove building IDs with empty in.schedules.csv

    The fuction returns a list of building IDs that have in.schedules.csv file

    TODO: Remove this function from here. It takes so much time. Also, this is bad implementation
    '''
    exists = []
    root = Path(dataset_path)
    for entry in root.iterdir():
        up00 = entry / 'up00'
        target_file = up00 / 'in.schedules.csv'
        # try:

        if target_file.is_file() and target_file.stat().st_size > 0:
            exists.append(entry.name)

    # for folder in os.listdir(dataset_path):
    #     upgrade_path = os.path.join(dataset_path, folder, 'up00')
    #     if os.path.isdir(upgrade_path):
    #         target_file = os.path.join(upgrade_path, 'in.schedules.csv')
    #         # Instead of reading the file to see if it's empty, check its size and skip if it's zero
    #         if os.path.isfile(target_file) and os.path.getsize(target_file) > 0:
    #             "Checking if in.schedules.csv is empty or not"
    #             bldg_id = upgrade_path.split('/')[-2]
    #             exists.append(bldg_id)
    # exists is just a list of building IDs.
    return exists

# %%

def bar_plots (ldc):
    '''
    This function plots the maximum demand vs percentage of the time
    '''
    df = ldc.copy()

    df["duration_bin"] = df["duration_fraction"] * 100
    df["duration_bin"] = pd.cut(df["duration_fraction"], bins=96, labels=False)

    bin_means = df.groupby("duration_bin")["P_sorted_kW"].mean().reset_index()
    bin_means["P_sorted_kW"] = bin_means["P_sorted_kW"]/1000

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=bin_means,
                x="duration_bin",
                y="P_sorted_kW")
    

    # plt.xticks([], [])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.set_xlabel("Percent of Time (%)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Load Duration Curve")
    plt.tight_layout()
    plt.savefig('./ldc.png')
    plt.show()
# %%
def line_plots (df: pd.DataFrame, x: str, y:str, title: str):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x=x, y=y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

#%%
def calculate_daily_metrics (df: pd.DataFrame, p_max_summary: list, bldg: str, up: str):
    '''
    This function calculates the load factor for every customer. [Add more context. Why?].
    
    Load factor: It is a term used to describle a load. It is the ratio of the average demand
    to the max demand.
    
    Steps (Cite William Kersting):

    1) Identify the peak power and the time where the power is at its peak
    2) Calculate the average power
    3) Get the energy, or calculate it if needed.
    4) Calculate the load factor = avergage power / peak power
    '''
    idx_peak = df['Total Electric Power (kW)'].idxmax()
    p_peak = df.loc[idx_peak, 'Total Electric Power (kW)']
    t_peak = df.loc[idx_peak, 'Time']
    t_sim = (df['Time'].iloc[-1] - df['Time'].iloc[0]).total_seconds() / 3600
    p_avg = df['Total Electric Energy (kWh)'].sum() / t_sim
    load_factor = p_avg / p_peak

    p_max_summary.append({
        'upgrade': up,
        'bldg_id': bldg,
        'daily_avg [kW]': p_avg,
        'peak_time': t_peak,
        'daily_peak_power [kW]': p_peak,
        'daily_load_factor': load_factor,
        'daily_energy_energy [kWh]': df['Total Electric Energy (kWh)'].sum()
    })
    return p_max_summary

# %%

def calculate_diversified_demand(df: pd.DataFrame, group_p_sum: None,group_time_index: None):
    '''
    Diversified demand is the sum of all loads at each time step. For isntance:
    Time    |   Sum of 30 houses [kW]
    15:00   |   20
    15:15   |   60
    15:30   |   35

    and so forth.
    '''
    p = df['Total Electric Power (kW)']
    
    if group_p_sum is None:
        group_p_sum = p.copy()
        group_time_index = df['Time']
    else:
        group_p_sum += p
    
    return group_p_sum, group_time_index
# %%
def calculate_load_duration_curve (p_diversified, t_diversified):
    '''
    Defintion:
    The LDC is the fraction of time wherein the load is above some X kW value.

    Steps:
    1) Calculate the diversified demand (already calculated)
    2) Sort it in a descending order.
    3) Then we can see the times wherein the demand is at max and descending for all houses.

    INPUTS:
    p_diversified and t_diversified are the diversified demand as per Kerstings. 
        - See calculate_diversified_demand() for more info
    
    The LDC typically shows the peak demand vs the percentage of time that that peak happened.
    '''

    p_div_sorted = np.sort(p_diversified)[::-1]
    if len(t_diversified) > 1:
        dt_hours = (t_diversified.iloc[1] - t_diversified.iloc[0]).total_seconds()/3600
    else:
        dt_hours = 0.0
    
    n = len(p_div_sorted)
    total_time_hours = n*dt_hours

    # let's compute the fraction of time:

    duration_fraction = np.arange(1, n+1)/n

    duration_hours = duration_fraction * total_time_hours

    return pd.DataFrame({
        "P_sorted_kW": p_div_sorted,
        'duration_fraction': duration_fraction,
        'duration_hours': duration_hours
    })

# %%
def calculate_diversity_factor (df, max_div_demand):
    '''
    Each customer has a different peak than the others. We want to know where would the peak of all
    customers fall within the load duration curve.

    STEPS:

    1) Calculate the non-coincidental demand:
        The sum of all the peak powers from houses, known as the Non-coincident Demand
    2) Cacluate the diversity factor: 
        This is simply the ratio of the maximum non-coincident demand to the maximum diversified demand
    '''
    # calculate the non-coincidental demand:
    noncoincidental_demand = df['daily_peak_power [kW]'].sum()

    diversity_factor = noncoincidental_demand / max_div_demand
    coincidence_factor = max_div_demand / noncoincidental_demand

    return diversity_factor, coincidence_factor

# %%

def accumulated_diversity_factor (load_profiles: list, summary_df: pd.DataFrame):
    '''
    Similar to calculate_diversity_factor(). However, it calculates the accumulated DF for plotting purposes.
    '''
    accum_diversity_factor = []
    for n in range(1, len(load_profiles)+1):

        # calculate the non-coincidental demand:
        p_sum_n = np.sum(load_profiles[:n], axis=0)

        # Get the peak of the non-coincidental demand:
        p_diversified_n = p_sum_n.max()

        # Maximum non-coincidental demand is the sum of the daily peak kW:
        sum_peaks_n = summary_df['daily_peak_power [kW]'].iloc[:n].sum()

        # 4) calculate diversity factor:
        diversity_factor_n = sum_peaks_n /p_diversified_n

        accum_diversity_factor.append ({
            'n_customers': n,
            'diversity_factor': diversity_factor_n
        })

    return pd.DataFrame(accum_diversity_factor)

# %%
def calculate_utilization_factor (max_diversified_demand, pf = 0.9, transformer_rating = 15):
    '''
    The utilization factor is an indication of how well the capacity of a given electrical device is being utilized. Consider a transformer rated for 15 kVA and assuming
    a power factor of 0.9, the maximum kVA demand can be calculated using the maximum diversified demand/pf. As such, the utilization factor becomes:

    Utilization factor = Maximum kVA demand / Transformer rating, where:

        Maximum kVA demand = Maximum diversified demand / Power factor
    '''
    max_kva_demand = max_diversified_demand / pf
    utilization_factor = max_kva_demand / transformer_rating

    return utilization_factor
    

#%%
if __name__ == '__main__':
    input_paths = []
    dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/cosimulation'
    bldgs = filter_datasets(dataset_path=dataset_dir)
    # random.shuffle(bldgs)
    bldgs = bldgs[:500]
    # bldgs = ['298']
    upgrades = ['up00']
    for upgrade in upgrades:
        for bldg in bldgs:
            input_path = os.path.join(dataset_dir, bldg, upgrade)
            input_paths.append(input_path)
    
    sns.set_theme(style="whitegrid", context="talk")

    load_summary = []
    load_profiles = []
    diversified_demand_p = None
    diversified_demand_time = None

    keep_cols = ['Time', 'Total Electric Power (kW)', 'Total Reactive Power (kVAR)',
                 'Total Electric Energy (kWh)', 'Total Reactive Energy (kVARh)' ]
    
    for input_path in input_paths:
        bldg = input_path.split('/')[-2]
        up = input_path.split('/')[-1]
        target_file = input_path+f"/out_{input_path.split('/')[-2]}_{input_path.split('/')[-1]}.csv"
        df = pd.read_csv(target_file, usecols=keep_cols)
        df['Time'] = pd.to_datetime(df['Time'])
        load_profiles.append(df['Total Electric Power (kW)'].to_numpy())

        load_summary = calculate_daily_metrics (df=df, 
                              p_max_summary=load_summary,
                              bldg=bldg,
                              up=up)
        
        diversified_demand_p, diversified_demand_time = calculate_diversified_demand(df=df, 
                                                    group_p_sum=diversified_demand_p,
                                                    group_time_index=diversified_demand_time)
    
    if diversified_demand_p is not None:
        summary_df = pd.DataFrame(load_summary)
        # Here is the Diversified demand reporting:
        idx_group_peak = np.argmax(diversified_demand_p) # This is a column of P values. grab the max P value index
        p_max_diversified = diversified_demand_p[idx_group_peak] # This is the P value at the index
        t_max_diversified = diversified_demand_time.iloc[idx_group_peak] # this is the time of the max P value occurred

        ldc = calculate_load_duration_curve(p_diversified=diversified_demand_p, t_diversified=diversified_demand_time)
        bar_plots(ldc=ldc)
        
        diversity_factor, coincidence_factor = calculate_diversity_factor(
            df=summary_df, max_div_demand=p_max_diversified)

    diversity_factor_n_df = accumulated_diversity_factor (load_profiles=load_profiles,
                                                          summary_df=summary_df)
    line_plots(df=diversity_factor_n_df, x='n_customers', y='diversity_factor', 
               title='Diversity Factor - Loads: 500')
    plt.show()