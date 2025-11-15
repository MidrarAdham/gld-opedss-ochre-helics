#%%
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
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
    missing = []
    exists = []
    for folder in os.listdir(dataset_path):
        upgrade_path = os.path.join(dataset_path, folder, 'up00')
        if os.path.isdir(upgrade_path):
            target_file = os.path.join(upgrade_path, 'in.schedules.csv')
            if os.path.isfile(target_file):
                "Checking if in.schedules.csv is empty or not"
                try:
                    df = pd.read_csv(target_file)
                    bldg_id = upgrade_path.split('/')[-2]
                    exists.append(bldg_id)
                except pd.errors.EmptyDataError:
                    missing.append(bldg_id)
            else:
                "Checking if in.schedules.csv exists"
                missing.append(bldg_id)
    # exists is just a list of building IDs.
    return exists

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
#%%
if __name__ == '__main__':
    input_paths = []
    dataset_dir = '/home/deras/gld-opedss-ochre-helics/datasets/cosimulation'
    bldgs = filter_datasets(dataset_path=dataset_dir)
    # bldgs = ['298']
    upgrades = ['up00']
    for upgrade in upgrades:
        for bldg in bldgs:
            input_path = os.path.join(dataset_dir, bldg, upgrade)
            input_paths.append(input_path)
    
    sns.set_theme(style="whitegrid", context="talk")

    load_summary = []
    keep_cols = ['Time', 'Total Electric Power (kW)', 'Total Reactive Power (kVAR)',
                 'Total Electric Energy (kWh)', 'Total Reactive Energy (kVARh)' ]
    for input_path in input_paths:
        bldg = input_path.split('/')[-2]
        up = input_path.split('/')[-1]
        target_file = input_path+f"/out_{input_path.split('/')[-2]}_{input_path.split('/')[-1]}.csv"
        df = pd.read_csv(target_file, usecols=keep_cols)
        df['Time'] = pd.to_datetime(df['Time'])
        bldg = input_path.split('/')[-2]
        load_summary = calculate_daily_metrics (df=df, 
                              p_max_summary=load_summary,
                              bldg=bldg,
                              up=up)
        pp(load_summary)
        quit()