'''
Author: Midrar Adham
Created: Sun Apr 26 2026

Proposal Figures
----------------
    Section 1 — Bayesian State Estimation
        Fig 1. Raw binary states (WH vs HVAC)
        Fig 2. Posterior mean — discount=0.01 vs discount=0.3
        Fig 3. Posterior variance trajectories
        Fig 4. State matrix heatmap

    Section 2 — Mean Matrix
        Fig 5. Mean matrix heatmap (WH and HVAC)
        Fig 6. Aggregated x_wh and x_hvac time series

    Section 3 — Standard OLS
        Fig 7. Predicted vs ground truth (WH and HVAC)
        Fig 8. OLS residuals plot with RMSE and R²

    Section 4 — Delta OLS
        Fig 9. Δx_wh vs Δx_hvac — switching dynamics comparison
        Fig 10. Delta OLS predicted vs ground truth
        Fig 11. Comparison table across days (discount=0.01 vs 0.3)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Global style settings for formal proposal figures ─────────────────────────
# Uses a clean, minimal academic style suitable for IEEE/dissertation submission.
STYLE = {
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.alpha':        0.4,
    'legend.frameon':    False,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'lines.linewidth':   1.5,
}

# Color palette — grayscale-friendly for printing
COLORS = {
    'black':      '#1a1a1a',
    'dark_gray':  '#444444',
    'mid_gray':   '#888888',
    'light_gray': '#cccccc',
    'accent':     '#2166ac',   # blue — used sparingly for predicted lines
    'accent2':    '#d6604d',   # red — used for residuals / HVAC
}


class ProposalFigures:
    """
    Generates publication-quality figures for the dissertation proposal.

    All figures follow a consistent formal academic style:
    - Serif font (Times New Roman)
    - No top/right spines
    - Minimal grid lines
    - Grayscale-friendly color palette
    - 300 DPI output

    Parameters
    ----------
    time_col : pd.Series
        HH:MM formatted time labels for the x-axis, one per 10-minute chunk.
        Derived from the transformer DataFrame as:
        pd.to_datetime(xfmr_df['Time']).dt.strftime('%H:%M')

    save_dir : str
        Directory where figures are saved. Defaults to current directory.
    """

    def __init__(self, time_col: pd.Series, save_dir: str = './'):
        self.time_col = time_col
        self.save_dir = save_dir
        plt.rcParams.update(STYLE)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Save figure to save_dir and close it."""
        fig.savefig(f'{self.save_dir}{filename}')
        plt.close(fig)
        print(f"Saved: {self.save_dir}{filename}")

    # ── Section 1 — Bayesian State Estimation ────────────────────────────────

    def fig1_raw_binary_states(
            self,
            wh_df: pd.DataFrame,
            hvac_df: pd.DataFrame,
            wh_label: str = 'Water Heater',
            hvac_label: str = 'HVAC',
            filename: str = 'fig1_raw_binary_states.png'
            ) -> None:
        """
        Plot raw binary ON/OFF states for one WH unit and one HVAC unit
        side by side to visually show the difference in switching behavior.

        WH units switch abruptly and infrequently — sharp, well-defined
        pulses. HVAC units cycle more continuously — frequent, shorter
        bursts. This difference motivates the delta regression approach.

        Parameters
        ----------
        wh_df : pd.DataFrame
            Cleaned WH DataFrame from DataLoader — must contain 'state' column.

        hvac_df : pd.DataFrame
            Cleaned HVAC DataFrame from DataLoader — must contain 'state' column.

        wh_label : str
            Label for the WH subplot title.

        hvac_label : str
            Label for the HVAC subplot title.

        filename : str
            Output filename. Defaults to PDF for proposal submission.
        """
        # Resample to 10-minute resolution to match the OLS pipeline
        wh_states   = wh_df.set_index('time')['state'].resample('10min').mean()
        hvac_states = hvac_df.set_index('time')['state'].resample('10min').mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax1.step(range(len(wh_states)), wh_states.values,
                 where='post', color=COLORS['black'], linewidth=1.2)
        ax1.fill_between(range(len(wh_states)), wh_states.values,
                         step='post', alpha=0.15, color=COLORS['black'])
        ax1.set_ylabel('State (ON=1, OFF=0)')
        ax1.set_title(f'(a) {wh_label} Binary States')
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_yticks([0, 1])

        ax2.step(range(len(hvac_states)), hvac_states.values,
                 where='post', color=COLORS['dark_gray'], linewidth=1.2)
        ax2.fill_between(range(len(hvac_states)), hvac_states.values,
                         step='post', alpha=0.15, color=COLORS['dark_gray'])
        ax2.set_ylabel('State (ON=1, OFF=0)')
        ax2.set_title(f'(b) {hvac_label} Binary States')
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_yticks([0, 1])
        ax2.set_xlabel('Chunk Index (10-min intervals)')

        plt.tight_layout()
        self._save(fig, filename)

    def fig2_posterior_mean_comparison(
            self,
            histories_low: dict,
            histories_high: dict,
            der_label: str = 'Water Heater',
            discount_low: float = 0.01,
            discount_high: float = 0.3,
            filename: str = 'fig2_posterior_mean_comparison.png'
            ) -> None:
        """
        Compare posterior mean trajectories for one DER under two
        discount factors side by side.

        A low discount (0.01) produces a responsive signal that tracks
        switching events closely. A high discount (0.3) produces a smooth,
        slowly-varying signal that loses switching detail — making it
        unsuitable for delta regression.

        Parameters
        ----------
        histories_low : dict
            Output of BayesianEstimator.fit_many() with low discount.

        histories_high : dict
            Output of BayesianEstimator.fit_many() with high discount.

        der_label : str
            DER type label for plot titles.

        discount_low : float
            The low discount value used.

        discount_high : float
            The high discount value used.

        filename : str
            Output filename.
        """
        # Use the first DER in each history dict for illustration
        fname      = list(histories_low.keys())[0]
        mean_low   = histories_low[fname]['mean']
        mean_high  = histories_high[fname]['mean']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax1.plot(mean_low, color=COLORS['black'], linewidth=1.2)
        ax1.set_ylabel('Posterior Mean P(ON)')
        ax1.set_title(f'(a) {der_label} — Discount = {discount_low} (Responsive)')
        ax1.set_ylim(-0.05, 1.05)

        ax2.plot(mean_high, color=COLORS['dark_gray'], linewidth=1.2)
        ax2.set_ylabel('Posterior Mean P(ON)')
        ax2.set_title(f'(b) {der_label} — Discount = {discount_high} (Smooth)')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel('Chunk Index (10-min intervals)')

        plt.tight_layout()
        self._save(fig, filename)

    def fig3_posterior_variance(
            self,
            wh_histories: dict,
            hvac_histories: dict,
            filename: str = 'fig3_posterior_variance.png'
            ) -> None:
        """
        Plot posterior variance trajectories for all WH and HVAC units.

        WH variance spikes sharply at switching events then returns to
        near-zero — the estimator quickly becomes confident. HVAC variance
        remains persistently high — the estimator is rarely confident.
        This difference reflects the fundamentally different thermal
        dynamics of the two device types.

        Parameters
        ----------
        wh_histories : dict
            Output of BayesianEstimator.fit_many() for WH data.

        hvac_histories : dict
            Output of BayesianEstimator.fit_many() for HVAC data.

        filename : str
            Output filename.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for fname, history in wh_histories.items():
            ax1.plot(history['variance'], alpha=0.5,
                     color=COLORS['black'], linewidth=0.8)
        ax1.set_ylabel('Posterior Variance')
        ax1.set_title('(a) Water Heater Posterior Variance Trajectories')

        for fname, history in hvac_histories.items():
            ax2.plot(history['variance'], alpha=0.5,
                     color=COLORS['dark_gray'], linewidth=0.8)
        ax2.set_ylabel('Posterior Variance')
        ax2.set_title('(b) HVAC Posterior Variance Trajectories')
        ax2.set_xlabel('Chunk Index (10-min intervals)')

        plt.tight_layout()
        self._save(fig, filename)

    def fig4_state_matrix_heatmap(
            self,
            wh_state_matrix: pd.DataFrame,
            hvac_state_matrix: pd.DataFrame,
            filename: str = 'fig4_state_matrix_heatmap.png'
            ) -> None:
        """
        Visualize the WH and HVAC state matrices as heatmaps.

        Each row is a 10-minute chunk, each column is one DER unit.
        Color intensity represents the fraction of the window the device
        was ON (0=OFF, 1=fully ON). This shows at a glance how many
        devices are ON at any given time and how their patterns differ.

        Parameters
        ----------
        wh_state_matrix : pd.DataFrame
            Output of OLS._build_state_matrix() for WH data.
            Shape: (144 x n_wh_ders).

        hvac_state_matrix : pd.DataFrame
            Output of OLS._build_state_matrix() for HVAC data.
            Shape: (144 x n_hvac_ders).

        filename : str
            Output filename.
        """
        # Grayscale colormap — white=OFF, black=ON
        cmap = LinearSegmentedColormap.from_list(
            'proposal', ['white', COLORS['black']]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(wh_state_matrix.values, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1,
                         interpolation='nearest')
        ax1.set_title('(a) Water Heater State Matrix')
        ax1.set_xlabel('DER Unit Index')
        ax1.set_ylabel('Chunk Index (10-min intervals)')
        plt.colorbar(im1, ax=ax1, label='ON Fraction')

        im2 = ax2.imshow(hvac_state_matrix.values, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1,
                         interpolation='nearest')
        ax2.set_title('(b) HVAC State Matrix')
        ax2.set_xlabel('DER Unit Index')
        ax2.set_ylabel('Chunk Index (10-min intervals)')
        plt.colorbar(im2, ax=ax2, label='ON Fraction')

        plt.tight_layout()
        self._save(fig, filename)

    # ── Section 2 — Mean Matrix ───────────────────────────────────────────────

    def fig5_mean_matrix_heatmap(
            self,
            wh_mean_matrix: pd.DataFrame,
            hvac_mean_matrix: pd.DataFrame,
            filename: str = 'fig5_mean_matrix_heatmap.png'
            ) -> None:
        """
        Visualize the Bayesian posterior mean matrices for WH and HVAC.

        Unlike the binary state matrix, the mean matrix contains
        continuous values between 0 and 1 — the posterior probability
        that each device is ON at each chunk. Comparing this to the
        state matrix shows how the Bayesian estimator smooths the
        raw binary signal.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of OLS._build_mean_matrix() for WH histories.

        hvac_mean_matrix : pd.DataFrame
            Output of OLS._build_mean_matrix() for HVAC histories.

        filename : str
            Output filename.
        """
        cmap = LinearSegmentedColormap.from_list(
            'proposal', ['white', COLORS['black']]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(wh_mean_matrix.values, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1,
                         interpolation='nearest')
        ax1.set_title('(a) Water Heater Posterior Mean Matrix')
        ax1.set_xlabel('DER Unit Index')
        ax1.set_ylabel('Chunk Index (10-min intervals)')
        plt.colorbar(im1, ax=ax1, label='P(ON)')

        im2 = ax2.imshow(hvac_mean_matrix.values, aspect='auto',
                         cmap=cmap, vmin=0, vmax=1,
                         interpolation='nearest')
        ax2.set_title('(b) HVAC Posterior Mean Matrix')
        ax2.set_xlabel('DER Unit Index')
        ax2.set_ylabel('Chunk Index (10-min intervals)')
        plt.colorbar(im2, ax=ax2, label='P(ON)')

        plt.tight_layout()
        self._save(fig, filename)

    def fig6_aggregated_probability_vectors(
            self,
            x_wh: np.ndarray,
            x_hvac: np.ndarray,
            filename: str = 'fig6_aggregated_probability_vectors.png'
            ) -> None:
        """
        Plot the aggregated ON-probability vectors x_wh and x_hvac.

        These are the inputs to the OLS regression — the sum of
        posterior means across all DER units of each type. They
        represent the expected total number of ON devices at each
        10-minute chunk.

        Parameters
        ----------
        x_wh : np.ndarray
            Aggregated WH ON-probability vector. Shape: (144,).

        x_hvac : np.ndarray
            Aggregated HVAC ON-probability vector. Shape: (144,).

        filename : str
            Output filename.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax1.plot(x_wh, color=COLORS['black'], linewidth=1.2)
        ax1.set_ylabel('Expected Units ON')
        ax1.set_title('(a) Aggregated WH ON-Probability Vector ($x_{WH}$)')

        ax2.plot(x_hvac, color=COLORS['dark_gray'], linewidth=1.2)
        ax2.set_ylabel('Expected Units ON')
        ax2.set_title('(b) Aggregated HVAC ON-Probability Vector ($x_{HVAC}$)')
        ax2.set_xlabel('Chunk Index (10-min intervals)')

        plt.tight_layout()
        self._save(fig, filename)

    # ── Section 3 — Standard OLS ──────────────────────────────────────────────

    def fig7_ols_predicted_vs_truth(
            self,
            wh_predicted: np.ndarray,
            hvac_predicted: np.ndarray,
            wh_ground_truth: pd.Series,
            hvac_ground_truth: pd.Series,
            filename: str = 'fig7_ols_predicted_vs_truth.png'
            ) -> None:
        """
        Plot standard OLS predicted demand vs ground truth for both
        WH and HVAC on separate panels.

        Parameters
        ----------
        wh_predicted : np.ndarray
            Predicted WH demand from standard OLS. Shape: (144,).

        hvac_predicted : np.ndarray
            Predicted HVAC demand from standard OLS. Shape: (144,).

        wh_ground_truth : pd.Series
            Ground truth WH transformer demand.

        hvac_ground_truth : pd.Series
            Ground truth HVAC transformer demand.

        filename : str
            Output filename.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(self.time_col, pd.to_numeric(wh_ground_truth, errors='coerce') / 1e3,
                 color=COLORS['black'], linewidth=1.2, label='Ground Truth', alpha=0.8)
        ax1.plot(self.time_col, wh_predicted / 1e3,
                 color=COLORS['accent'], linewidth=1.2,
                 linestyle='--', label='OLS Predicted')
        ax1.set_ylabel('Demand [kW]')
        ax1.set_title('(a) Water Heater — Standard OLS')
        ax1.legend()
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))

        ax2.plot(self.time_col, pd.to_numeric(hvac_ground_truth, errors='coerce') / 1e3,
                 color=COLORS['black'], linewidth=1.2, label='Ground Truth', alpha=0.8)
        ax2.plot(self.time_col, hvac_predicted / 1e3,
                 color=COLORS['accent'], linewidth=1.2,
                 linestyle='--', label='OLS Predicted')
        ax2.set_ylabel('Demand [kW]')
        ax2.set_title('(b) HVAC — Standard OLS')
        ax2.set_xlabel('Time [HH:MM]')
        ax2.legend()
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(12))

        plt.tight_layout()
        self._save(fig, filename)

    def fig8_ols_residuals(
            self,
            feeder_minus_background: np.ndarray,
            combined_predicted: np.ndarray,
            filename: str = 'fig8_ols_residuals.png'
            ) -> None:
        """
        Plot the OLS fit quality with residuals shown as vertical lines.

        The residuals — the vertical distances between actual and
        predicted values — are exactly what OLS minimizes. This plot
        makes that minimization visible and allows visual assessment
        of where the model fits well and where it struggles.

        Parameters
        ----------
        feeder_minus_background : np.ndarray
            Feeder demand with background subtracted. Shape: (144,).

        combined_predicted : np.ndarray
            Sum of wh_predicted and hvac_predicted. Shape: (144,).

        filename : str
            Output filename.
        """
        rmse = np.sqrt(np.mean((feeder_minus_background - combined_predicted) ** 2))
        r2   = 1 - np.sum((feeder_minus_background - combined_predicted) ** 2) / \
                   np.sum((feeder_minus_background - feeder_minus_background.mean()) ** 2)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(self.time_col, feeder_minus_background / 1e3,
                color=COLORS['black'], linewidth=1.2,
                label='Actual (Feeder − Background)', alpha=0.8)
        ax.plot(self.time_col, combined_predicted / 1e3,
                color=COLORS['accent'], linewidth=1.2,
                linestyle='--', label='OLS Predicted (WH + HVAC)')

        for i in range(len(self.time_col)):
            ax.vlines(
                x=self.time_col.iloc[i],
                ymin=min(feeder_minus_background[i], combined_predicted[i]) / 1e3,
                ymax=max(feeder_minus_background[i], combined_predicted[i]) / 1e3,
                color=COLORS['accent2'], alpha=0.35, linewidth=0.8
            )

        ax.annotate(
            f'RMSE: {rmse/1e3:.2f} kW\n$R^2$: {r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                      edgecolor=COLORS['light_gray'])
        )

        ax.set_ylabel('Demand [kW]')
        ax.set_xlabel('Time [HH:MM]')
        ax.set_title('Standard OLS Fit Quality — Residuals')
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12))

        plt.tight_layout()
        self._save(fig, filename)

    # ── Section 4 — Delta OLS ─────────────────────────────────────────────────

    def fig9_delta_signals(
            self,
            x_wh: np.ndarray,
            x_hvac: np.ndarray,
            filename: str = 'fig9_delta_signals.png'
            ) -> None:
        """
        Plot Δx_wh and Δx_hvac side by side to show their different
        switching dynamics.

        WH delta signals have sharp, infrequent spikes corresponding
        to abrupt switching events. HVAC delta signals have more
        frequent, smaller fluctuations reflecting gradual cycling.
        This difference is what allows delta OLS to separate the two.

        Parameters
        ----------
        x_wh : np.ndarray
            Aggregated WH ON-probability vector. Shape: (144,).

        x_hvac : np.ndarray
            Aggregated HVAC ON-probability vector. Shape: (144,).

        filename : str
            Output filename.
        """
        delta_wh   = np.diff(x_wh)
        delta_hvac = np.diff(x_hvac)
        chunks     = np.arange(1, len(x_wh))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax1.bar(chunks, delta_wh, color=COLORS['black'],
                alpha=0.7, width=0.8)
        ax1.axhline(0, color=COLORS['mid_gray'], linewidth=0.8, linestyle='--')
        ax1.set_ylabel('$\Delta x_{WH}$')
        ax1.set_title('(a) Water Heater Delta Signal — Sharp, Infrequent Spikes')

        ax2.bar(chunks, delta_hvac, color=COLORS['dark_gray'],
                alpha=0.7, width=0.8)
        ax2.axhline(0, color=COLORS['mid_gray'], linewidth=0.8, linestyle='--')
        ax2.set_ylabel('$\Delta x_{HVAC}$')
        ax2.set_title('(b) HVAC Delta Signal — Frequent, Smaller Fluctuations')
        ax2.set_xlabel('Chunk Index (10-min intervals)')

        plt.tight_layout()
        self._save(fig, filename)

    def fig10_delta_ols_predicted_vs_truth(
            self,
            wh_predicted: np.ndarray,
            hvac_predicted: np.ndarray,
            wh_ground_truth: pd.Series,
            hvac_ground_truth: pd.Series,
            filename: str = 'fig10_delta_ols_predicted_vs_truth.png'
            ) -> None:
        """
        Plot Delta OLS predicted demand vs ground truth for WH and HVAC.

        Parameters
        ----------
        wh_predicted : np.ndarray
            Predicted WH demand from Delta OLS. Shape: (144,).

        hvac_predicted : np.ndarray
            Predicted HVAC demand from Delta OLS. Shape: (144,).

        wh_ground_truth : pd.Series
            Ground truth WH transformer demand.

        hvac_ground_truth : pd.Series
            Ground truth HVAC transformer demand.

        filename : str
            Output filename.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(self.time_col,
                 pd.to_numeric(wh_ground_truth, errors='coerce') / 1e3,
                 color=COLORS['black'], linewidth=1.2,
                 label='Ground Truth', alpha=0.8)
        ax1.plot(self.time_col, wh_predicted / 1e3,
                 color=COLORS['accent'], linewidth=1.2,
                 linestyle='--', label='Delta OLS Predicted')
        ax1.set_ylabel('Demand [kW]')
        ax1.set_title('(a) Water Heater — Delta OLS (discount = 0.01)')
        ax1.legend()
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))

        ax2.plot(self.time_col,
                 pd.to_numeric(hvac_ground_truth, errors='coerce') / 1e3,
                 color=COLORS['black'], linewidth=1.2,
                 label='Ground Truth', alpha=0.8)
        ax2.plot(self.time_col, hvac_predicted / 1e3,
                 color=COLORS['accent'], linewidth=1.2,
                 linestyle='--', label='Delta OLS Predicted')
        ax2.set_ylabel('Demand [kW]')
        ax2.set_title('(b) HVAC — Delta OLS (discount = 0.01)')
        ax2.set_xlabel('Time [HH:MM]')
        ax2.legend()
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(12))

        plt.tight_layout()
        self._save(fig, filename)

    def fig11_discount_comparison_table(
            self,
            results: dict,
            filename: str = 'fig11_discount_comparison_table.png'
            ) -> None:
        """
        Render a formatted comparison table showing WH and HVAC
        prediction errors across days for discount=0.01 vs discount=0.3.

        This table is the key evidence for the discount factor finding.

        Parameters
        ----------
        results : dict
            Structure:
            {
                'Day 2': {
                    'wh_truth': float, 'hvac_truth': float,
                    'low_wh':   float, 'low_hvac':   float,
                    'high_wh':  float, 'high_hvac':  float,
                },
                'Day 3': { ... },
                ...
            }

        filename : str
            Output filename.
        """
        days = list(results.keys())
        col_labels = [
            'Day',
            'WH Truth\n[W]', 'WH d=0.01\n[W]', 'WH Error\n[%]',
            'HVAC Truth\n[W]', 'HVAC d=0.01\n[W]', 'HVAC Error\n[%]',
            'WH d=0.3\n[W]', 'WH Err\n[%]',
            'HVAC d=0.3\n[W]', 'HVAC Err\n[%]',
        ]

        rows = []
        for day, vals in results.items():
            wh_err_low   = abs(vals['low_wh']  - vals['wh_truth'])  / vals['wh_truth']  * 100
            hvac_err_low = abs(vals['low_hvac'] - vals['hvac_truth']) / vals['hvac_truth'] * 100
            wh_err_high   = abs(vals['high_wh']  - vals['wh_truth'])  / vals['wh_truth']  * 100
            hvac_err_high = abs(vals['high_hvac'] - vals['hvac_truth']) / vals['hvac_truth'] * 100
            rows.append([
                day,
                f"{vals['wh_truth']:.0f}",
                f"{vals['low_wh']:.0f}",
                f"{wh_err_low:.1f}%",
                f"{vals['hvac_truth']:.0f}",
                f"{vals['low_hvac']:.0f}",
                f"{hvac_err_low:.1f}%",
                f"{vals['high_wh']:.0f}",
                f"{wh_err_high:.1f}%",
                f"{vals['high_hvac']:.0f}",
                f"{hvac_err_high:.1f}%",
            ])

        fig, ax = plt.subplots(figsize=(14, 2 + len(days) * 0.6))
        ax.axis('off')

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Style header row
        for j in range(len(col_labels)):
            table[0, j].set_facecolor(COLORS['dark_gray'])
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Alternate row shading
        for i in range(1, len(rows) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#f5f5f5')

        ax.set_title(
            'Table: Delta OLS Performance — discount=0.01 vs discount=0.3',
            fontsize=11, fontweight='bold', pad=20
        )

        plt.tight_layout()
        self._save(fig, filename)