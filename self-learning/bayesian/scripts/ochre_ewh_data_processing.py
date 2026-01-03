import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

root = Path (__file__).resolve ().parents[3]
dataset_dir = root / 'datasets' / 'resstock_2025' / 'load_profiles' / 'cosimulation' / '577' / 'up00' / 'ochre.csv'

df = pd.read_csv (dataset_dir, usecols = ['Time', 'Water Heating Electric Power (kW)'])

df = df.head(1440)

df.plot (x='Time', y='Water Heating Electric Power (kW)')
plt.show()