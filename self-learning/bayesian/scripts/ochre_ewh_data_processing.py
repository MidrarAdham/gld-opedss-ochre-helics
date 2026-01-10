import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import bayesian_experiment as bayesian

with open ('./filtered_dataset.txt', 'r', encoding='utf-8-sig') as f:
    input_paths = f.readlines ()
    input_paths = [s.strip() for s in input_paths if s.strip()]


fig, ax = plt.subplots(figsize=(10,6))

for f in input_paths:

    # bayesian.
    
    df = pd.read_csv (f, usecols = ['Time', 'Water Heating Electric Power (kW)'])

    # df = df.head(1400)

    max_kw = df['Water Heating Electric Power (kW)'].max()

    mean_kw = df['Water Heating Electric Power (kW)'].mean()

    median_kw = df['Water Heating Electric Power (kW)'].median()

    min_kw = df['Water Heating Electric Power (kW)'].min()

    print(f'max: {max_kw}\nmean: {mean_kw}\nmedian: {median_kw}\nmin: {min_kw}')
    mask = df['Water Heating Electric Power (kW)'].between (0,5.0, inclusive='neither')
    dff = df['Water Heating Electric Power (kW)'][mask]
    # print(dff)

    # print(df)
    
    print("="*40)

    # quit()
    
    # watts = df[df.columns[1]]
    
    # df['Time'] = pd.to_datetime (df['Time']).dt.strftime ('%d %H:%M')
    
    # ax.plot(df['Time'], watts)

    # ax.xaxis.set_major_locator (ticker.MaxNLocator (20))

    # plt.show()