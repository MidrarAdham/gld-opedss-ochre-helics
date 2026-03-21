'''
Author: MidrarAdham
Created: Fri Mar 20 2026
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def full_house_demand_and_WH_kw_estimation (wh_df : pd.DataFrame, feeder_df : pd.DataFrame):
    pass

def predicted_kw_each_der (df : pd.DataFrame, stats: dict, filename : str):
    '''
    This function plots the predicted WH profiles from the feeder demand.
    NOTE: 
        - Only WHs are used in houses, not all the equipment within a house.
        - So the feeder demand is essentially the aggregate WH demand.
    '''
    
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    time_col = pd.to_datetime(df ['# timestamp']).dt.strftime ('%H:%M')
    
    ax.fill_between(
            time_col, stats['y_low'], stats['y_up'],
            color='tab:blue',
            alpha=0.15,
            linewidth=0,
            label='95% CI'
        )

    ax.plot(
        time_col, stats['y_true'],
        color='black',
        linewidth=2.4,
        label='Ground truth'
    )

    ax.plot(
        time_col, stats['y_mean'],
        color='tab:blue',
        linewidth=2.0,
        label='Predicted mean'
    )



    ax.grid(True, which='major', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(
    f"{filename} | MAE={stats['mae']:.2f} kW | RMSE={stats['rmse']:.2f} kW | R²={stats['R_squared']:.2f}"
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Power [kW]')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.show()