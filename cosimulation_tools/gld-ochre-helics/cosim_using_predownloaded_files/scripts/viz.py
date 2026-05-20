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
    
    def plot_hvac_individual_predictions(
            self,
            results: dict,
            hvac_ground_truth: pd.Series,
            wh_ground_truth: pd.Series
            ):
        """
        Plot individual HVAC predictions (4 separate + 1 combined rest)
        alongside the ground truth feeder HVAC demand, plus a WH panel
        to verify WH estimation was not impacted by HVAC modifications.
 
        Parameters
        ----------
        results : dict
            Output of OrdinaryLeastSquare.run() containing kw_hvac_1..4,
            kw_hvac_rest, x_hvac_1..4, x_hvac_rest, hvac_predicted,
            kw_wh, x_wh, and wh_predicted.
        hvac_ground_truth : pd.Series
            Ground truth HVAC demand in Watts.
        wh_ground_truth : pd.Series
            Ground truth WH demand in Watts.
        """
        hvac_truth_clean = pd.to_numeric(hvac_ground_truth, errors='coerce').values
        wh_truth_clean   = pd.to_numeric(wh_ground_truth,   errors='coerce').values
 
        # Build individual HVAC predictions
        individual_preds = {
            'HVAC 1':    results['kw_hvac_1']    * results['x_hvac_1'],
            'HVAC 2':    results['kw_hvac_2']    * results['x_hvac_2'],
            'HVAC 3':    results['kw_hvac_3']    * results['x_hvac_3'],
            'HVAC 4':    results['kw_hvac_4']    * results['x_hvac_4'],
            'HVAC rest': results['kw_hvac_rest'] * results['x_hvac_rest'],
        }
 
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
 
        # ── Top panel: individual HVAC contributions ──────────────────
        for label, pred in individual_preds.items():
            ax1.plot(self.time_col, pred / 1e3, linewidth=1.2, label=label)
        ax1.set_ylabel('Demand [kW]')
        ax1.set_title('Individual HVAC Contributions')
        ax1.legend(frameon=False, ncol=5)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(20))
 
        # ── Middle panel: combined HVAC prediction vs ground truth ────
        ax2.plot(self.time_col,
                 hvac_truth_clean / 1e3,
                 color='black', linewidth=1.5,
                 label='Ground Truth', alpha=0.8)
        ax2.plot(self.time_col,
                 results['hvac_predicted'] / 1e3,
                 color='tab:blue', linewidth=1.5,
                 linestyle='--', label='Combined HVAC Predicted')
        ax2.set_ylabel('Demand [kW]')
        ax2.set_title('Combined HVAC Prediction vs Ground Truth')
        ax2.legend(frameon=False)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(20))
 
        hvac_rmse = np.sqrt(np.mean((results['hvac_predicted'] - hvac_truth_clean) ** 2))
        hvac_r2   = 1 - np.sum((results['hvac_predicted'] - hvac_truth_clean) ** 2) / \
                        np.sum((hvac_truth_clean - hvac_truth_clean.mean()) ** 2)
        ax2.annotate(
            f'RMSE: {hvac_rmse/1e3:.2f} kW\n$R^2$: {hvac_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.7, edgecolor='lightgray')
        )
 
        # ── Bottom panel: WH prediction vs ground truth ───────────────
        ax3.plot(self.time_col,
                 wh_truth_clean / 1e3,
                 color='black', linewidth=1.5,
                 label='Ground Truth', alpha=0.8)
        ax3.plot(self.time_col,
                 results['wh_predicted'] / 1e3,
                 color='tab:orange', linewidth=1.5,
                 linestyle='--', label='WH Predicted')
        ax3.set_ylabel('Demand [kW]')
        ax3.set_title('WH Prediction vs Ground Truth')
        ax3.set_xlabel('Time [HH:MM]')
        ax3.legend(frameon=False)
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(20))
 
        wh_rmse = np.sqrt(np.mean((results['wh_predicted'] - wh_truth_clean) ** 2))
        wh_r2   = 1 - np.sum((results['wh_predicted'] - wh_truth_clean) ** 2) / \
                      np.sum((wh_truth_clean - wh_truth_clean.mean()) ** 2)
        ax3.annotate(
            f'RMSE: {wh_rmse/1e3:.2f} kW\n$R^2$: {wh_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.7, edgecolor='lightgray')
        )
 
        plt.tight_layout()
        # plt.savefig('./hvac_individual_predictions.png')

        plt.savefig('./figures/hvac_individual_predictions_multi_regressor_method.png')
        # plt.show()

    def plot_for_collinearity(
            self,
            hvac_regressors: dict,
            hvac_ground_truth: pd.Series
            ):
        hvac_truth_clean = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        n = len(hvac_regressors)
        fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)

        for ax, (label, x) in zip(axes, hvac_regressors.items()):
            # print(x)
            # quit()
            ax.plot(self.time_col, x / 1e3,
                    linewidth=1.2, label=label, color='tab:blue')
            ax.plot(self.time_col, hvac_truth_clean / 1e3,
                    linewidth=1.2, label='Ground Truth',
                    color='black', alpha=0.6, linestyle='--')
            
            # Correlation coefficient
            corr = np.corrcoef(x, hvac_truth_clean)[0, 1]
            ax.set_ylabel('Demand [kW]')
            ax.set_title(f'{label}  |  corr with ground truth: {corr:.3f}')
            ax.legend(frameon=False, ncol=2)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

        axes[-1].set_xlabel('Time [HH:MM]')
        plt.tight_layout()
        plt.savefig('./collinearity_check.png')
        plt.show()
    

    def plot_hvac_variance(
        self,
        hvac_mean_matrix: pd.DataFrame
    ):
        """
        Plot the variance of each HVAC device's posterior mean trajectory
        across all 144 chunks. High variance = more switching activity =
        more informative regressor.
        """
        variances = hvac_mean_matrix.var(axis=0).sort_values(ascending=False)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # ── Top panel: bar chart of variance per device ───────────────
        ax1.bar(range(len(variances)), variances.values, color='tab:blue', alpha=0.7)
        ax1.set_xticks(range(len(variances)))
        ax1.set_xticklabels(variances.index, rotation=45, ha='right')
        ax1.set_ylabel('Variance')
        ax1.set_title('HVAC Posterior Mean Variance per Device (sorted)')
        ax1.axhline(y=variances.mean(), color='red', linestyle='--',
                    linewidth=1.2, label=f'Mean variance: {variances.mean():.4f}')
        ax1.legend(frameon=False)

        # ── Bottom panel: mean trajectory per device ──────────────────
        for col in hvac_mean_matrix.columns:
            ax2.plot(self.time_col, hvac_mean_matrix[col].values,
                    linewidth=0.9, alpha=0.7, label=col)
        ax2.set_ylabel('Posterior Mean P(ON)')
        ax2.set_title('HVAC Posterior Mean Trajectories')
        ax2.set_xlabel('Time [HH:MM]')
        ax2.legend(frameon=False, ncol=4)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(20))

        plt.tight_layout()
        plt.savefig('./hvac_variance.png')
        plt.show ()
