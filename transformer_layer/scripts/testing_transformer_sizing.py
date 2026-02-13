import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
from config import load_config
import matplotlib.pyplot as plt
from pprint import pprint as pp
import matplotlib.ticker as ticker


def build_dataset_dir (bldg_ids : list, prefix : str, suffix : str):
    """
    Given a list of building IDs passed, this function creates a full path for each building ID
    
    :param bldg_ids: a list of building IDs
    :type bldg_ids: list
    :param prefix: the prefix of the path where the bldg ids are located
    :type prefix: str
    :param suffix: the suffix of the path wherein the bldg ids are located
    :type suffix: str
    """
    finalized_dataset_dir = []
    for bldg_id in bldg_ids:
        finalized_dataset_dir.append(f"{prefix}/{bldg_id}{suffix}")
    
    return finalized_dataset_dir

def get_list_full_dataset (dataset_dir : str) -> list:
    """
    Get the list of all the files in the given directory
    
    :param dataset_dir: directory where all the csv files exists
    :type dataset_dir: str
    """
    x = [filename for filename in os.listdir (dataset_dir)]
    x.remove ('409590')
    x.remove ('355669')
    shuffeled_list = random.sample (x, k=len (x))
    return shuffeled_list


def concatenate_load_profiles (dataset_dir : str):
    """
    To concatenate dataframes, we execute the following steps:
    1) append all load profiles in a single list
    2) concatenate that list into a single dataframe
    
    :param dataset_dir: the directory wherein the dataset is located
    :type dataset_dir: str
    """
    dfs = []

    
    for idx, filename in enumerate(dataset_dir):

        bldg_name = (filename.split('/'))[-3]
        keep_cols = ['Time', 'Total Electric Power (kW)','Total Reactive Power (kVAR)']
        # keep_cols = ['Time', 'Total Electric Power (kW)']

        df = pd.read_csv (filename, usecols=keep_cols)

        df = df.head(1440)

        if idx != 0:
            df = df.drop('Time', axis=1)

        df = df.rename (columns={
            'Total Electric Power (kW)' : f'{bldg_name}: Total Electric Power (kW)',
            'Total Reactive Power (kVAR)' : f'{bldg_name}: Total Reactive Power (kVAR)'
            })

        dfs.append(df)

    return pd.concat (dfs, axis=1)

def calculate_diverisifed_demand (df : pd.DataFrame):
    """
    Sum up each row of the DataFrame to get the Maximum Diversified Demand
    
    :param df: concatenated DataFrame
    :type df: pd.DataFrame
    """
    kw_cols = [col for col in df.columns if 'kW' in col]
    kvar_cols = [col for col in df.columns if 'kVAR' in col]
    
    df['Total_kW'] = df[kw_cols].sum(axis=1)
    df['Total_kVAR'] = df[kvar_cols].sum(axis=1)
    
    df['Diversified Demand (kVA)'] = np.sqrt(
        df['Total_kW']**2 + df['Total_kVAR']**2
    )
    
    return df

def calculate_daily_peak_demand (df : pd.DataFrame):
    df['block'] = df.index // 15
    block_avg = df.groupby ('block')['Diversified Demand (kVA)'].mean ()

    daily_peak = block_avg.max ()
    
    return daily_peak

def plotting(results: dict):
    sns.set_theme(context='notebook')
    ncols = 2
    nrows = math.ceil(len(results) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))
    axes = axes.flatten()

    for i, (trial_id, values) in enumerate(results.items()):
        
        df = results[trial_id]['dataframe']  # Changed from [0]
        peak_demand = results[trial_id]['peak_demand']  # Get the peak demand
        
        num_houses = [col.split(':')[0] for col in df.columns]
        num_houses = len(list(set(num_houses))) - 2
        df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M')

        ax = axes[i]
        ax.plot(df['Time'], df['Diversified Demand (kVA)'].round(decimals=2), 
                linewidth=2, color='blue', label='Apparent Power (kVA)')
        
        # Add horizontal line showing the peak demand
        ax.axhline(y=peak_demand, color='red', linestyle='--', linewidth=2, 
                   label=f'Daily Peak: {peak_demand:.2f} kVA')
        
        ax.set_title(f'Diversified Demand of {num_houses} Bldgs (kVA)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
        ax.tick_params(axis='x', rotation=45)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Diversified Demand (kVA)', fontweight='bold')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":

    cfg = load_config ()
    dataset_dir = cfg["data"]["dataset_dir"]
    prefix = dataset_dir
    suffix = '/up00/ochre.csv'

    # abnormal_buildings ()



    mc_trials = 8
    results_dict = {}

    
    for trial in range(mc_trials):

        possible_bldgs_75kva = random.randint (8,10)

        random_choice_list = random.choices (get_list_full_dataset (dataset_dir=dataset_dir), k=possible_bldgs_75kva)
        
        final_dataset_dir = build_dataset_dir (bldg_ids=random_choice_list, prefix=prefix, suffix=suffix)

        df = concatenate_load_profiles (dataset_dir=final_dataset_dir)

        df = calculate_diverisifed_demand (df=df)

        peak_demand = calculate_daily_peak_demand (df=df)

        if not trial in results_dict:
            results_dict[trial] = {}
        
        results_dict[trial]['dataframe'] = df
        results_dict[trial]['peak_demand'] = peak_demand

    plotting (results=results_dict)