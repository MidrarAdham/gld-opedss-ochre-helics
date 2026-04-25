'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import DataLoader
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class visualizer:
    
    def __init__(self, time_col : pd.Series):
        self.time_col = time_col
    
    def plot_wh_predicted_vs_ground_truth (
            self,
            wh_predicted : np.ndarray,
            wh_ground_truth : pd.Series
    ):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        # Plot predicted demand, converting from W to kW
        ax.plot(
            self.time_col,
            wh_predicted / 1e3,
            label='WH predicted',
            color='tab:blue',
            linewidth = 2
        )

        # Plot ground truth demand, converting from W to kW
        ax.plot(
            self.time_col,
            pd.to_numeric(wh_ground_truth, errors='coerce') / 1e3,
            label='WH ground truth',
            color='black',
            alpha=0.7,
            linewidth = 2
        )

        ax.set_ylabel('Demand [kW]')
        ax.set_title ('WH')
        ax.legend(frameon=False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

        plt.tight_layout()
        plt.savefig ('./water_heaters.png')

        plt.show()
    
    def plot_hvac_predicted_vs_ground_truth (
            self,
            hvac_predicted : np.ndarray,
            hvac_ground_truth : pd.Series
    ):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        # Plot predicted demand, converting from W to kW
        ax.plot(
            self.time_col,
            hvac_predicted / 1e3,
            label='HVAC predicted',
            color='tab:blue',
            linewidth = 2
        )

        # Plot ground truth demand, converting from W to kW
        ax.plot(
            self.time_col,
            pd.to_numeric(hvac_ground_truth, errors='coerce') / 1e3,
            label='WH ground truth',
            color='black',
            alpha=0.7,
            linewidth = 2
        )

        ax.set_ylabel('Demand [kW]')
        ax.set_title ('HVAC')
        ax.legend(frameon=False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

        plt.tight_layout()
        plt.savefig ('hvac.png')

        plt.show()
        