'''
Author: MidrarAdham
Created: Fri Mar 20 2026
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

results_dir = '../results/hvac_cosim/'

ders_df = [pd.read_csv (results_dir+f, skiprows=8) for f in os.listdir (results_dir) if not 'transformer' in f]

xfmr_df = pd.read_csv (results_dir+'residential_transformer.csv', skiprows=8)

def cleanup_results_files (df : pd.DataFrame, col : str):
    df.loc[:, '# timestamp'] = df['# timestamp'].apply (lambda x: x.strip ('PST'))
    df.loc[:, '# timestamp'] = pd.to_datetime (df['# timestamp'])
    df.loc[:, col] = df[col].apply (lambda x: complex (x))
    df.loc[:, col] = df[col].apply(lambda x: x.real)
    return df



for idx, df in enumerate(ders_df):
    df = cleanup_results_files(df=df, col='constant_power_12')

    df = df.iloc[1440:2880].copy()

    df['# timestamp'] = pd.to_datetime(df['# timestamp'])
    df = df.set_index('# timestamp')

    # 10-minute bin averages
    df_avg = df.resample('10min').mean().reset_index()

    time = df_avg['# timestamp']
    power = pd.to_numeric(df_avg['constant_power_12']) / 1e3

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    # 10 minutes in matplotlib date units (days)
    ten_min_width = 10 / 1440

    # Bar shows each 10-minute bin
    ax.bar(
        time,
        power,
        width=ten_min_width,
        align='edge',
        alpha=0.7,
        label='10-min average power',
        edgecolor = 'black',
        linewidth=0.5
    )

    # Step plot matches binned data better than a regular line
    ax.step(
        time,
        power,
        where='post',
        color='black',
        linewidth=1.8,
        label='Binned profile'
    )

    max_power = power.max()
    ax.axhline(
        max_power,
        linestyle='--',
        color='red',
        alpha=0.25,
        label=f'Max HVAC Power = {max_power:.2f} kW'
    )

    ax.set_xlabel('Time [HH:MM]')
    ax.set_ylabel('Real Power [kW]')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.grid(True, alpha=0.4)
    ax.legend(frameon=False, loc='best')
    plt.tight_layout()
    plt.show()

