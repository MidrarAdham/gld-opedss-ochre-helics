'''
Author: Midrar Adham
Created: Wed Mar 25 2026
'''
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

load_paths_file = './load_paths.json'
with open (load_paths_file, 'r') as f:
    data = json.load (f)

dfs = []
for key, value in data.items ():
    df = pd.read_csv(value, parse_dates=["Time"])
    df = df.iloc[1440:2880]
    time_col = df['Time']
    df = df[['Total Electric Power (kW)', 'Water Heating Electric Power (kW)']]
    df['background'] = df['Total Electric Power (kW)'] - df['Water Heating Electric Power (kW)']
    dfs.append (df)


df = pd.concat([df['background'] for df in dfs], axis=1)
df_total = pd.concat([df['Total Electric Power (kW)'] for df in dfs], axis=1)
df['div.sum'] = df.sum(axis=1)
df_total['div.sum'] = df_total.sum(axis=1)

# time_col = pd.to_datetime (time_col).dt.strftime ('%H:%M')

fig, ax = plt.subplots (figsize=(16,6))

df['div.sum'] = pd.to_numeric(df['div.sum'], errors='coerce')
df['div.sum'] = round(df['div.sum'], 2)

df_total['div.sum'] = pd.to_numeric (df_total['div.sum'], errors='coerce')
df_total['div.sum'] = round (df_total['div.sum'], 2)

df.insert (column = 'Time',value = time_col, loc=0)
df = df.set_index ('Time')
df = df.resample ('10min').mean ()
df = df.reset_index ()


df_total.insert (column = 'Time',value = time_col, loc=0)
df_total = df_total.set_index ('Time')
df_total = df_total.resample ('10min').mean ()
df_total = df_total.reset_index ()

# df = df.head (134)
# df_total = df_total.head (134)

x = df_total['div.sum'] - df['div.sum']

df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = df['Time'].dt.strftime ('%H:%M')

ax.plot (df['Time'], df['div.sum'], color='tab:blue', label='Background feeder demand')
ax.plot (df['Time'], df_total['div.sum'], color='black', label='Full house feeder demand')
ax.plot (df['Time'], x, color='tab:red', label = 'Diff bet full house and background feeder demand')
ax.xaxis.set_major_locator (ticker.MaxNLocator (20))
ax.set_xlim (df['Time'].min (), df['Time'].max ())
ax.grid ()
ax.legend ()
plt.show()