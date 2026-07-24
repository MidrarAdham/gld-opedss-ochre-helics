import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_days = 10

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for day in range(1, train_days + 1):
    filtered   = pd.read_csv(f'r2_vs_fleet_size_filtered_day{day}.csv')
    unfiltered = pd.read_csv(f'r2_vs_fleet_size_unfiltered_day{day}.csv')

    # plot on same axes, one line per day
    axes[0, 0].plot(filtered['N'],   filtered['r2_mean'],   label=f'day {day}')
    axes[0, 1].plot(unfiltered['N'], unfiltered['r2_mean'], label=f'day {day}')
    axes[1, 0].plot(filtered['N'],   filtered['mape_mean'], label=f'day {day}')
    axes[1, 1].plot(unfiltered['N'], unfiltered['mape_mean'], label=f'day {day}')

axes[0, 0].set_title('R² — Filtered (1-ton+)')
axes[0, 1].set_title('R² — Unfiltered')
axes[1, 0].set_title('MAPE — Filtered (1-ton+)')
axes[1, 1].set_title('MAPE — Unfiltered')

for ax in axes.flat:
    ax.set_xlabel('Number of HVAC Devices')
    ax.legend(fontsize=7)
    ax.grid(True)

axes[0, 0].set_ylabel('R²')
axes[0, 1].set_ylabel('R²')
axes[1, 0].set_ylabel('MAPE (%)')
axes[1, 1].set_ylabel('MAPE (%)')

plt.tight_layout()
plt.savefig('comparison_filtered_vs_unfiltered.png')
plt.show()
