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

        # plt.show()
    
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
            label='HVAC ground truth',
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

        # plt.show()
    
    def plot_ols_residuals(
        self,
        feeder_minus_background: np.ndarray,
        combined_predicted: np.ndarray
        ):
        """
        Visualize the OLS fit quality by plotting the actual feeder signal,
        the predicted signal, and the residual errors as vertical red lines.

        The residuals are the vertical distances between the actual and
        predicted values at each chunk — these are exactly what OLS minimizes
        when finding the best fit. Smaller and more randomly scattered
        residuals indicate a better fit.

        Parameters
        ----------
        feeder_minus_background : np.ndarray
            The feeder signal with background demand subtracted.
            Comes from results['feeder_minus_background'] in main.py.

        combined_predicted : np.ndarray
            The sum of wh_predicted and hvac_predicted.
            Comes from results['combined_predicted'] in main.py.

        save_path : str, optional
            If provided, saves the figure to this path.
            If None, the figure is only displayed and not saved.
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        # Plot actual feeder signal minus background
        ax.plot(
            self.time_col,
            feeder_minus_background / 1e3,
            label='Actual (feeder - background)',
            color='black',
            alpha=0.7
        )

        # Plot OLS combined prediction
        ax.plot(
            self.time_col,
            combined_predicted / 1e3,
            label='OLS predicted (WH + HVAC)',
            color='tab:blue'
        )

        # Draw vertical red lines showing the residual at each chunk
        for i in range(len(self.time_col)):
            ax.vlines(
                x=self.time_col.iloc[i],
                ymin=min(feeder_minus_background[i], combined_predicted[i]) / 1e3,
                ymax=max(feeder_minus_background[i], combined_predicted[i]) / 1e3,
                color='red',
                alpha=0.4,
                linewidth=0.8
            )

        ax.set_ylabel('Demand [kW]')
        ax.set_xlabel('Time [HH:MM]')
        ax.legend(frameon=False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

        rmse = np.sqrt(np.mean((feeder_minus_background - combined_predicted) ** 2))
        r2   = 1 - np.sum((feeder_minus_background - combined_predicted) ** 2) / \
                np.sum((feeder_minus_background - feeder_minus_background.mean()) ** 2)

        ax.annotate(
            f'RMSE: {rmse/1e3:.2f} kW\nR²: {r2:.3f}',
            xy=(0.01, 0.95),
            xycoords='axes fraction',
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        plt.tight_layout()

        plt.savefig ('./residuals.png')

        # plt.show()

    def plot_posterior_variance(
        self,
        wh_histories: dict,
        hvac_histories: dict,
        save_path: str = None
        ):
        """
        Plot the posterior variance trajectory for each DER over time.

        The variance of the Beta posterior describes how uncertain the
        Bayesian estimator is about the ON-probability at each chunk.
        A high variance means the estimator is unsure — the device has
        been switching frequently or hasn't been observed long enough.
        A low variance means the estimator is confident.

        Comparing WH and HVAC variance trajectories can reveal differences
        in their dynamic behavior — devices with slow variance decay have
        longer thermal time constants, analogous to eigenvalues in
        Callaway's aggregate TCL model.

        Parameters
        ----------
        wh_histories : dict
            Output of BayesianEstimator.fit_many() for WH data.
            Keys are filenames, values are history dicts containing
            a 'variance' list of 144 values.

        hvac_histories : dict
            Same as wh_histories but for HVAC data.

        save_path : str, optional
            If provided, saves the figure to this path.
            If None, the figure is only displayed and not saved.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Plot WH variance trajectories — one line per DER
        for filename, history in wh_histories.items():
            ax1.plot(
                history['variance'],
                alpha=0.6,
                linewidth=0.9
            )

        ax1.set_ylabel('Posterior Variance')
        ax1.set_title('Water Heater Posterior Variance Trajectories')
        ax1.grid(True, linestyle='--', alpha=0.4)

        # Plot HVAC variance trajectories — one line per DER
        for filename, history in hvac_histories.items():
            ax2.plot(
                history['variance'],
                alpha=0.6,
                linewidth=0.9
            )

        ax2.set_ylabel('Posterior Variance')
        ax2.set_title('HVAC Posterior Variance Trajectories')
        ax2.set_xlabel('Chunk Index (10-min intervals)')
        ax2.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()

        plt.savefig ('./variance.png')

        plt.show()
        