import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
import os
import numpy as np



ochre_dir = './cosimulation/bldg0112631/up00/'



def ts (df):
    return pd.to_datetime(df)

ochre = pd.read_csv(ochre_dir+'House_1.csv')
gld = pd.read_csv('./ochre_load.csv', header=8)

fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(16,10))
gld_kw = gld['constant_power_12'].apply(lambda x: (round(abs(complex(x)),2)/1000))
gld['time'] = gld['# timestamp'].apply(lambda x: x.replace('PDT',''))
gld['time'] = pd.to_datetime(gld['time']).dt.strftime('%H:%M')
# gld_time = pd.to_datetime(gld['# timestamp'])

ax[0].plot(gld['time'], gld_kw)
ax[0].set_title('OCHRE Load mirror in GLD')
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(20))
ax[0].grid()

ochre['Time'] = pd.to_datetime(ochre['Time']).dt.strftime('%H:%M')
ax[1].plot(ochre['Time'], ochre['Total Electric Power (kW)'])
ax[1].set_title('Actual OCHRE Load')
ax[1].xaxis.set_major_locator(ticker.MaxNLocator(20))
ax[1].grid()
# ax[1]
plt.show()
