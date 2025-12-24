import os
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# GLD stuff
xfmr = pd.read_csv('./residential_transformer.csv', header=8)
gld_load = pd.read_csv('./ochre_load.csv', header=8)
ochre = pd.read_csv('./cosimulation/bldg0112631/up00/House_1.csv')

# print(xfmr)
# quit()
# xfmr = xfmr.drop(xfmr.index[:7])
# print(xfmr.drop(xfmr.index[:11]))
# quit()

# quit()
xfmr['# timestamp'] = xfmr['# timestamp'].apply(lambda x: x.replace('PST', ''))
xfmr['# timestamp'] = xfmr['# timestamp'].apply(lambda x: x.replace('PDT', ''))

gld_load['# timestamp'] = gld_load['# timestamp'].apply(lambda x: x.replace('PST', ''))
gld_load['# timestamp'] = gld_load['# timestamp'].apply(lambda x: x.replace('PDT', ''))

print(f"MAX -->:\n ochre load: {ochre['Time'].max()}\ngld load: {gld_load['# timestamp'].max()}\ngld xfmr: {xfmr['# timestamp'].max()}")

# quit()

xfmr['# timestamp'] = pd.to_datetime(xfmr['# timestamp']).dt.strftime('%m-%d %H:%M')

gld_load['# timestamp'] = pd.to_datetime(gld_load['# timestamp']).dt.strftime('%m-%d %H:%M')

ochre['Time'] = pd.to_datetime(ochre['Time']).dt.strftime('%m-%d %H:%M')


xfmr['real_polar_kva'] = xfmr['power_in'].apply(lambda x: np.real(complex(x))/1e3)
xfmr['imag_polar_kva'] = xfmr['power_in'].apply(lambda x: np.imag(complex(x)))

gld_load['real_polar_kva'] = gld_load['constant_power_12'].apply(lambda x: np.real(complex(x)/1e3))

load_kw = ochre['Total Electric Power (kW)']



def axes_processing (ax: np.ndarray, titles: list) -> np.ndarray:
    for i in range(len(ax)):

        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=30))
        ax[i].tick_params(axis='x', rotation=45)
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Time (m-d HH:MM)')
        ax[i].set_ylabel('Apparent Power (kVA)')
        ax[i].grid()

fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(16,10))

titles = ['GridLAB-D Transformer', 'GLD Load', 'OCHRE Load']

ax[0].plot(xfmr['# timestamp'], xfmr['real_polar_kva'])

ax[1].plot(gld_load['# timestamp'], gld_load['real_polar_kva'])

ax[2].plot(ochre['Time'], load_kw)



axes_processing(ax=ax, titles=titles)

ax[0].set_xlim(xfmr['# timestamp'].min(), xfmr['# timestamp'].max())
ax[1].set_xlim(gld_load['# timestamp'].min(), gld_load['# timestamp'].max())
ax[2].set_xlim(ochre['Time'].min(), ochre['Time'].max())

fig.tight_layout()

plt.show()