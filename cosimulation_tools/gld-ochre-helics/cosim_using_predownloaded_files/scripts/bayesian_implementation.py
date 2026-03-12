import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
from scipy.special import comb # calculates combinations (N choose K)

def create_binary_states (df : pd.DataFrame, threshold : float) -> pd.DataFrame:
    """
    convert power values to binary states (ON = 1, OFF=0)
    
    :param df: a one-week water heater load profile
    :type df: pd.DataFrame
    :return: df with added "state" column
    :rtype: DataFrame
    """
    # df['state'] = (df['tn_meter_4br_46:measured_real_power'] ==4500).astype(int)
    df['state'] = (df[df.columns[1]] > threshold).astype(int)
    return df

def cleanup_results_files (df : pd.DataFrame):
    df = df.head (1440)
    df['# timestamp'] = df['# timestamp'].apply (lambda x: x.strip ('PST'))
    df['# timestamp'] = pd.to_datetime (df['# timestamp'])
    df['constant_power_12'] = df['constant_power_12'].apply (lambda x: complex (x))
    df['constant_power_12'] = df['constant_power_12'].apply (lambda x: x.real)
    return df

cosim_results_dir = '../results/'
cosim_results_files = [f for f in os.listdir (cosim_results_dir) if 'ochre' in f]

for filename in cosim_results_files:
    df = pd.read_csv (cosim_results_dir+filename, skiprows=8)
    df = cleanup_results_files (df=df)
    print(df)