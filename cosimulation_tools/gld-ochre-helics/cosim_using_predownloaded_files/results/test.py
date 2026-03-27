'''
Author: Midrar Adham
Created: Tue Mar 24 2026
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

total_house_dir = './total_house_consumption'
only_wh_dir = './wh_cosim'
only_hvac_dir = './hvac_cosim'


def cleanup_results_files (df : pd.DataFrame, col : str):
    df.loc[:, '# timestamp'] = df['# timestamp'].apply (lambda x: x.strip ('PST'))
    df.loc[:, '# timestamp'] = pd.to_datetime (df['# timestamp'])
    df.loc[:, col] = df[col].apply (lambda x: complex (x))
    df.loc[:, col] = df[col].apply(lambda x: x.real)
    return df

def prepare_transformer_data (transformer_file_dir : str, sampling_method : str):
    df = pd.read_csv (f'{transformer_file_dir}/residential_transformer.csv', skiprows=8)
    # df = df.head (1440)
    df = df.iloc [1440:2880]
    df = cleanup_results_files (df=df, col = 'power_out')
    df = df.drop ('power_in', axis=1)
    df['# timestamp'] = pd.to_datetime(df['# timestamp'])
    df = df.set_index ('# timestamp')
    if sampling_method == 'mean':
        df = df.resample ("10min").mean()
    else:
        df = df.resample ("10min").max()
    df = df.reset_index ()

    return df


total_house_xfmr = [f for f in os.listdir (total_house_dir) if 'residential' in f]
only_wh_xfmr = [f for f in os.listdir (only_wh_dir) if 'residential' in f]
only_hvac_xfmr = [f for f in os.listdir (only_hvac_dir) if 'residential' in f]

total_house_df = pd.read_csv (total_house_dir+'/'+total_house_xfmr[0], skiprows=8)
only_wh_df = pd.read_csv (only_wh_dir+'/'+only_wh_xfmr[0], skiprows=8)
only_hvac_df = pd.read_csv (only_hvac_dir+'/'+only_hvac_xfmr[0], skiprows=8)

total_house_df = prepare_transformer_data (total_house_dir, sampling_method='mean')
only_wh_df = prepare_transformer_data (only_wh_dir, sampling_method='mean')
only_hvac_df = prepare_transformer_data (only_hvac_dir, sampling_method='mean')

time_col = pd.to_datetime(only_wh_df ['# timestamp']).dt.strftime ('%H:%M')

only_wh_df['power_out'] = round (pd.to_numeric (only_wh_df['power_out'], errors='coerce') / 1e3, 2)
only_hvac_df['power_out'] = round (pd.to_numeric (only_hvac_df['power_out'], errors='coerce') / 1e3, 2)
total_house_df['power_out'] = round (pd.to_numeric (total_house_df['power_out'], errors='coerce') / 1e3, 2)
# total_house_df['power_out'] = round(total_house_df['power_out'] / 1e3, 2)
total_house_df['no_hvac_wh'] = total_house_df['power_out'] - (only_hvac_df['power_out'] + only_wh_df['power_out'])


fig, ax = plt.subplots (figsize=(16,6))
ax.plot (time_col, total_house_df['power_out'], color='tab:blue', label='full house')
ax.plot (time_col, total_house_df['no_hvac_wh'], color='tab:red', label='full house without hvac and wh')
# ax.plot (time_col, only_hvac_df['power_out'], color='tab:red', label='hvac demand')
# ax.plot(time_col, only_wh_df['power_out'], color='black', label='wh demand')
ax.xaxis.set_major_locator (ticker.MaxNLocator (20))
ax.set_xlim (time_col.min (), time_col.max ())
ax.legend ()
ax.grid()
plt.savefig('./test.png')
quit()