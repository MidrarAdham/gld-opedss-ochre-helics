'''
Author: MidrarAdham
Created: Mon Mar 09 2026
'''
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

filenames = [f for f in os.listdir ('../results/') if f.endswith ('.csv') and 'residential' not in f]

fig, ax = plt.subplots (1,1, figsize=(16,6))
sns.set_context ('poster')
colors = plt.cm.viridis(np.linspace(0, 1, len(filenames)))

def fix_time_power (df : pd.DataFrame, col_name : str):
    df['# timestamp'] = df['# timestamp'].apply (lambda x: x.replace ('PST',''))
    df['# timestamp'] = pd.to_datetime(df['# timestamp']).dt.strftime ('%H:%M')
    df[col_name] = df[col_name].apply(complex)
    df[col_name] = df[col_name].apply(lambda x: abs (x))

    return df

for idx, filename in enumerate(filenames):
    df = pd.read_csv (f'../results/{filename}',skiprows=8)
    df = df.head(1440)
    col_name = 'constant_power_12'
    df = fix_time_power (df=df, col_name=col_name)
    time_col = df['# timestamp']
    df[col_name] = round(df[col_name]/1000, 2)
    power_col = df[col_name]
    ax.plot (time_col, power_col, linewidth = 2, color=colors[idx], label = filename)
    ax.xaxis.set_major_locator (ticker.MaxNLocator (nbins=22))
    ax.set_xlim (min (time_col), max(time_col))
    # ax.set_ylim (min (power_col), max(power_col)+5)

ax.set_xlabel ('Time [hour:minutes]')
ax.set_ylabel ('Apparent Power [kVA]')
ax.grid (True)
# ax.legend ()
# plt.savefig ('test.png')