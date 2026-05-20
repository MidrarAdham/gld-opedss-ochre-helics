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
        wh_all_dfs: dict,
        hvac_all_dfs: dict,
        filename: str = 'fig1_raw_binary_states.png'
        ) -> None:
        """
        Plot stacked bar charts showing the binary ON/OFF states of all
        WH and HVAC units across 144 ten-minute chunks.

        Each bar represents one chunk. Each colored segment within a bar
        represents one DER unit being ON during that chunk. The total
        bar height equals the number of units simultaneously ON.
        A solid line on top shows the aggregate count across all units.

        Parameters
        ----------
        wh_all_dfs : dict
            Output of DataLoader.all_dfs for WH data.

        hvac_all_dfs : dict
            Output of DataLoader.all_dfs for HVAC data.

        filename : str
            Output filename.
        """
        # ── Build resampled state matrices ────────────────────────────
        def build_resampled_states(all_dfs):
            state_dict = {}
            for fname, df in all_dfs.items():
                df = df.copy()
                df['time'] = pd.to_datetime(df['time'])
                states = df.set_index('time')['state']
                states = states.resample('10min').mean()
                label = fname.split('ochre_load_')[1].replace('.csv', '')
                state_dict[f'Unit {label}'] = states.values
            # Sort columns so Unit 0, 1, 2 ... appear in order
            df_out = pd.DataFrame(state_dict)
            df_out = df_out.reindex(sorted(df_out.columns,
                                    key=lambda x: int(x.split(' ')[1])), axis=1)
            return df_out

        wh_matrix   = build_resampled_states(wh_all_dfs)
        hvac_matrix = build_resampled_states(hvac_all_dfs)

        chunks = np.arange(len(wh_matrix))

        # ── Grayscale palette — one shade per unit, sorted ────────────
        def make_grays(n):
            return [str(v) for v in np.linspace(0.15, 0.80, n)]

        wh_colors   = make_grays(len(wh_matrix.columns))
        hvac_colors = make_grays(len(hvac_matrix.columns))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # ── WH stacked bar ─────────────────────────────────────────────
        bottoms = np.zeros(len(chunks))
        for col, color in zip(wh_matrix.columns, wh_colors):
            ax1.bar(chunks, wh_matrix[col].values,
                    bottom=bottoms, color=color,
                    width=0.9, edgecolor='none',
                    label=col)
            bottoms += wh_matrix[col].values

        # Aggregate line — total units ON at each chunk
        wh_total = wh_matrix.sum(axis=1).values
        ax1.step(chunks, wh_total, where='mid',
                color=COLORS['black'], linewidth=1.5,
                label='Total ON', zorder=5)

        ax1.set_ylabel('Units ON')
        ax1.set_title('Water Heater Binary States')
        ax1.set_ylim(0, len(wh_matrix.columns) + 0.5)
        ax1.set_yticks(range(len(wh_matrix.columns) + 1))
        ax1.legend(loc='upper right', ncol=5,
                fontsize=8, handlelength=1.0)

        # ── HVAC stacked bar ───────────────────────────────────────────
        bottoms = np.zeros(len(chunks))
        for col, color in zip(hvac_matrix.columns, hvac_colors):
            ax2.bar(chunks, hvac_matrix[col].values,
                    bottom=bottoms, color=color,
                    width=0.9, edgecolor='none',
                    label=col)
            bottoms += hvac_matrix[col].values

        # Aggregate line — total units ON at each chunk
        hvac_total = hvac_matrix.sum(axis=1).values
        ax2.step(chunks, hvac_total, where='mid',
                color=COLORS['black'], linewidth=1.5,
                label='Total ON', zorder=5)

        ax2.set_ylabel('Units ON')
        ax2.set_title('HVAC Binary States')
        ax2.set_ylim(0, len(hvac_matrix.columns) + 0.5)
        ax2.set_yticks(range(len(hvac_matrix.columns) + 1))
        ax2.set_xlabel('Chunk Index (10-min intervals)')
        ax2.legend(loc='upper right', ncol=5,
                fontsize=8, handlelength=1.0)

        plt.tight_layout()
        self._save(fig, filename)

    def fig2_posterior_mean_comparison(
        self,
        histories_low: dict,
        histories_high: dict,
        der_label: str,
        discount_low: float = 0.01,
        discount_high: float = 0.3,
        filename: str = 'fig2_posterior_mean_comparison.svg'
        ) -> None:
        """
        Compare population mean posterior trajectories for one DER type
        under two discount factors side by side.

        The population mean is the average of posterior mean values across
        all DER units of the given type at each chunk. Individual unit
        lines are omitted for clarity — the population mean captures the
        aggregate behavior that feeds into the OLS regression.

        A low discount (0.01) produces a responsive signal that tracks
        switching events closely. A high discount (0.3) produces a smooth,
        slowly-varying signal that loses switching detail.

        Parameters
        ----------
        histories_low : dict
            Output of BayesianEstimator.fit_many() with low discount.

        histories_high : dict
            Output of BayesianEstimator.fit_many() with high discount.

        der_label : str
            DER type label for plot titles. e.g. 'Water Heater' or 'HVAC'.

        discount_low : float
            The low discount value used.

        discount_high : float
            The high discount value used.

        filename : str
            Output filename.
        """
        def compute_population_mean(histories):
            """Average posterior mean across all DER units."""
            all_means = [h['mean'] for h in histories.values()]
            return np.mean(all_means, axis=0)

        pop_mean_low  = compute_population_mean(histories_low)
        pop_mean_high = compute_population_mean(histories_high)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        sharey=True, sharex=True)

        ax1.plot(pop_mean_low, color=COLORS['black'], linewidth=1.5)
        ax1.set_ylabel('Population Mean P(ON)')
        ax1.set_xlabel('Chunk Index (10-min intervals)')
        ax1.set_title(f'{der_label} — Discount = {discount_low}')
        ax1.set_ylim(-0.05, 1.05)

        ax2.plot(pop_mean_high, color=COLORS['black'], linewidth=1.5)
        ax2.set_xlabel('Chunk Index (10-min intervals)')
        ax2.set_title(f'{der_label} — Discount = {discount_high}')
        ax2.set_ylim(-0.05, 1.05)

        # Add a note explaining what the line represents
        fig.text(
            0.5, -0.02,
            f'Population mean computed as the average posterior mean across '
            f'all {der_label} units.',
            ha='center', fontsize=9, style='italic',
            color=COLORS['dark_gray']
        )

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

    def fig4_switching_event_zoom(
        self,
        wh_all_dfs: dict,
        wh_histories_low: dict,
        wh_histories_high: dict,
        unit_index: int = 0,
        center_chunk: int = None,
        window: int = 20,
        discount_low: float = 0.01,
        discount_high: float = 0.3,
        filename: str = 'fig4_switching_event_zoom.svg'
        ) -> None:
        """
        Zoom into a single WH switching event to show numerically how
        the posterior mean behaves differently under two discount factors.

        A switching event is a chunk where the device transitions from
        OFF to ON. The zoom window shows a few chunks before and after
        the event, making visible:
        - discount=0.01: posterior mean quickly reaches 1.0 and
            drops back sharply — responsive to the actual state
        - discount=0.3: posterior mean rises and falls slowly —
            lags behind the true state and never fully reaches 1.0
            for short ON events

        Parameters
        ----------
        wh_all_dfs : dict
            Output of DataLoader.all_dfs for WH data.

        wh_histories_low : dict
            Output of BayesianEstimator.fit_many() with low discount.

        wh_histories_high : dict
            Output of BayesianEstimator.fit_many() with high discount.

        unit_index : int
            Which WH unit to zoom into. Defaults to 0 (first unit).

        center_chunk : int
            The chunk index to center the zoom on. If None, the method
            automatically finds the first ON switching event for the
            selected unit.

        window : int
            Number of chunks to show on each side of the event.
            Default: 20 chunks (±20 = 40 chunks total = ~6.7 hours).

        discount_low : float
            The low discount value used.

        discount_high : float
            The high discount value used.

        filename : str
            Output filename.
        """
        # ── Get the selected unit ─────────────────────────────────────
        fname = sorted(
            wh_all_dfs.keys(),
            key=lambda x: int(x.split('ochre_load_')[1].replace('.csv', ''))
        )[unit_index]

        # ── Build resampled binary states ─────────────────────────────
        df = wh_all_dfs[fname].copy()
        df['time'] = pd.to_datetime(df['time'])
        states = df.set_index('time')['state'].resample('10min').mean().values

        # ── Get posterior means and CIs ───────────────────────────────
        mean_low  = np.array(wh_histories_low[fname]['mean'])
        mean_high = np.array(wh_histories_high[fname]['mean'])
        ci_lower  = np.array(wh_histories_low[fname]['ci_lower'])
        ci_upper  = np.array(wh_histories_low[fname]['ci_upper'])

        # ── Auto-detect switching event if not provided ───────────────
        if center_chunk is None:
            # Find first chunk where device switches from OFF to ON
            for i in range(1, len(states)):
                if states[i] > 0.5 and states[i-1] < 0.5:
                    center_chunk = i
                    break
            if center_chunk is None:
                center_chunk = window  # fallback

        # ── Define zoom window ────────────────────────────────────────
        start = max(0, center_chunk - window)
        end   = min(len(states), center_chunk + window)
        chunks = np.arange(start, end)

        # ── Plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))

        # Binary state — step function
        ax.step(chunks, states[start:end], where='post',
                color=COLORS['black'], linewidth=2.0,
                label='Binary State', zorder=4)

        # Posterior mean — discount=0.01
        ax.plot(chunks, mean_low[start:end],
                color=COLORS['accent'], linewidth=1.5,
                linestyle='--',
                label=f'Posterior Mean (discount={discount_low})',
                zorder=3)

        # CI shading for discount=0.01
        ax.fill_between(chunks,
                        ci_lower[start:end],
                        ci_upper[start:end],
                        color=COLORS['accent'], alpha=0.12,
                        label='95% CI (discount=0.01)')

        # Posterior mean — discount=0.3
        ax.plot(chunks, mean_high[start:end],
                color=COLORS['mid_gray'], linewidth=1.5,
                linestyle=':',
                label=f'Posterior Mean (discount={discount_high})',
                zorder=3)

        # Mark the switching event with a vertical line
        ax.axvline(x=center_chunk, color=COLORS['accent2'],
                linewidth=1.0, linestyle='--', alpha=0.6,
                label='Switching Event')

        ax.set_ylabel('State / P(ON)')
        ax.set_xlabel('Chunk Index (10-min intervals)')
        ax.set_title(
            f'Zoomed Switching Event — {fname.split("/")[-1]} '
            f'(Chunk {center_chunk})'
        )
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9, ncol=3)

        plt.tight_layout()
        self._save(fig, filename)

    # ── Section 2 — Mean Matrix ───────────────────────────────────────────────

    def fig5_switching_event_zoom_with_matrix(
        self,
        wh_all_dfs: dict,
        wh_histories_low: dict,
        wh_histories_high: dict,
        unit_index: int = 0,
        center_chunk: int = None,
        window: int = 20,
        discount_low: float = 0.01,
        discount_high: float = 0.3,
        filename: str = 'fig4_switching_event_zoom.svg'
        ) -> None:
            """
            Three-panel figure showing a zoomed switching event alongside the
            binary state and posterior mean matrices for the active window.
    
            Top panel:    Zoomed switching event plot showing binary state,
                        posterior mean under two discount factors, and 95% CI.
            Middle panel: Binary state matrix for the zoomed window — only
                        units that were active (ON at least once) are shown.
            Bottom panel: Posterior mean matrix for the same window and units,
                        with actual values annotated inside each cell.
    
            Parameters
            ----------
            wh_all_dfs : dict
                Output of DataLoader.all_dfs for WH data.
    
            wh_histories_low : dict
                Output of BayesianEstimator.fit_many() with low discount.
    
            wh_histories_high : dict
                Output of BayesianEstimator.fit_many() with high discount.
    
            unit_index : int
                Which WH unit to center the switching event on.
    
            center_chunk : int
                Chunk to center the zoom on. Auto-detected if None.
    
            window : int
                Number of chunks to show on each side of the event.
    
            discount_low : float
                The low discount value used.
    
            discount_high : float
                The high discount value used.
    
            filename : str
                Output filename.
            """
            # ── Sort filenames so units are ordered 0,1,2... ──────────────
            sorted_fnames = sorted(
                wh_all_dfs.keys(),
                key=lambda x: int(x.split('ochre_load_')[1].replace('.csv', ''))
            )
            fname = sorted_fnames[unit_index]
    
            # ── Build chunk-level binary states for selected unit ────────
            # Majority vote: chunk is ON (1) if >=50% of its minutes are ON
            df = wh_all_dfs[fname].copy()
            df['time'] = pd.to_datetime(df['time'])
            states_selected = df.set_index('time')['state']\
                                .resample('10min').mean()
            states_selected = (states_selected >= 0.5).astype(int).values
    
            # ── Get posterior means and CIs for selected unit ─────────────
            mean_low  = np.array(wh_histories_low[fname]['mean'])
            mean_high = np.array(wh_histories_high[fname]['mean'])
            ci_lower  = np.array(wh_histories_low[fname]['ci_lower'])
            ci_upper  = np.array(wh_histories_low[fname]['ci_upper'])
    
            # ── Auto-detect switching event ───────────────────────────────
            if center_chunk is None:
                for i in range(1, len(states_selected)):
                    if states_selected[i] > 0.5 and states_selected[i-1] < 0.5:
                        center_chunk = i
                        break
                if center_chunk is None:
                    center_chunk = window
    
            # ── Define zoom window ────────────────────────────────────────
            start  = max(0, center_chunk - window)
            end    = min(len(states_selected), center_chunk + window)
            chunks = np.arange(start, end)
    
            # ── Build state and mean matrices for ALL units in window ─────
            state_window = {}
            mean_window  = {}
            for fn in sorted_fnames:
                label = f"Unit {fn.split('ochre_load_')[1].replace('.csv', '')}"
                df_u  = wh_all_dfs[fn].copy()
                df_u['time'] = pd.to_datetime(df_u['time'])
                s = df_u.set_index('time')['state']\
                        .resample('10min').mean()
                s = (s >= 0.5).astype(int).values
                state_window[label] = s[start:end]
                mean_window[label]  = np.array(
                    wh_histories_low[fn]['mean']
                )[start:end]
    
            state_df = pd.DataFrame(state_window)
            mean_df  = pd.DataFrame(mean_window)
    
            # ── Keep only units active in this window ─────────────────────
            active_cols = [c for c in state_df.columns
                        if state_df[c].max() > 0]
            state_df = state_df[active_cols]
            mean_df  = mean_df[active_cols]
    
            # ── Colormaps ─────────────────────────────────────────────────
            cmap = LinearSegmentedColormap.from_list(
                'proposal', ['white', COLORS['black']]
            )
    
            # ── Figure layout — 3 panels, top taller ─────────────────────
            fig = plt.figure(figsize=(14, 10))
            gs  = gridspec.GridSpec(
                3, 1, height_ratios=[2.5, 1, 1], hspace=0.45
            )
            ax_plot   = fig.add_subplot(gs[0])
            ax_state  = fig.add_subplot(gs[1])
            ax_mean   = fig.add_subplot(gs[2])
    
            # ── Top panel — switching event plot ─────────────────────────
            ax_plot.step(chunks, states_selected[start:end],
                        where='post', color=COLORS['black'],
                        linewidth=2.0, label='Binary State', zorder=4)
            ax_plot.plot(chunks, mean_low[start:end],
                        color=COLORS['accent'], linewidth=1.5,
                        linestyle='--',
                        label=f'Posterior Mean (discount={discount_low})',
                        zorder=3)
            ax_plot.fill_between(chunks, ci_lower[start:end],
                                ci_upper[start:end],
                                color=COLORS['accent'], alpha=0.12,
                                label=f'95% CI (discount={discount_low})')
            ax_plot.plot(chunks, mean_high[start:end],
                        color=COLORS['mid_gray'], linewidth=1.5,
                        linestyle=':',
                        label=f'Posterior Mean (discount={discount_high})',
                        zorder=3)
            # ax_plot.axvline(x=center_chunk, color=COLORS['accent2'],
            #                 linewidth=1.0, linestyle='--', alpha=0.6)
            ax_plot.set_ylabel('State / P(ON)')
            ax_plot.set_title(
                f'Zoomed Switching Event- Window {center_chunk}',
                fontweight='bold'
            )
            ax_plot.set_ylim(-0.05, 1.15)
            ax_plot.legend(fontsize=9, ncol=2, loc='upper right', frameon=False)
            ax_plot.set_xlim(start, end - 1)
    
            # ── Middle panel — binary state matrix ───────────────────────
            im1 = ax_state.imshow(
                state_df.values.T,
                aspect='auto', cmap=cmap,
                vmin=0, vmax=1,
                interpolation='nearest',
                extent=[start - 0.5, end - 0.5,
                        len(active_cols) - 0.5, -0.5]
            )
            # Annotate each cell with its value
            for i, col in enumerate(active_cols):
                for j, chunk in enumerate(chunks):
                    val = state_df[col].values[j]
                    ax_state.text(
                        chunk, i, f'{int(val)}',
                        ha='center', va='center',
                        fontsize=7,
                        color='white' if val > 0.5 else COLORS['black']
                    )
            ax_state.set_yticks(range(len(active_cols)))
            ax_state.set_yticklabels(active_cols, fontsize=8)
            ax_state.set_title('Binary State Matrix',
                                fontweight='bold')
            ax_state.set_xlim(start - 0.5, end - 0.5)
            plt.colorbar(im1, ax=ax_state, fraction=0.02, pad=0.02,
                        label='State')
    
            # ── Bottom panel — posterior mean matrix ──────────────────────
            im2 = ax_mean.imshow(
                mean_df.values.T,
                aspect='auto', cmap=cmap,
                vmin=0, vmax=1,
                interpolation='nearest',
                extent=[start - 0.5, end - 0.5,
                        len(active_cols) - 0.5, -0.5]
            )
            # Annotate each cell with its value
            for i, col in enumerate(active_cols):
                for j, chunk in enumerate(chunks):
                    val = mean_df[col].values[j]
                    ax_mean.text(
                        chunk, i, f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=7,
                        color='white' if val > 0.5 else COLORS['black']
                    )
            ax_mean.set_yticks(range(len(active_cols)))
            ax_mean.set_yticklabels(active_cols, fontsize=8)
            ax_mean.set_title(
                f'Posterior Mean Matrix (discount={discount_low})',
                fontweight='bold'
            )
            ax_mean.set_xlabel('10-min intervals')
            ax_mean.set_xlim(start - 0.5, end - 0.5)
            plt.colorbar(im2, ax=ax_mean, fraction=0.02, pad=0.02,
                        label='P(ON)')
    
            plt.tight_layout()
            plt.show()
            # self._save(fig, filename)



    def fig6_aggregated_probability_vectors(
        self,
        wh_histories_low: dict,
        wh_histories_high: dict,
        hvac_histories_low: dict,
        hvac_histories_high: dict,
        discount_low: float = 0.01,
        discount_high: float = 0.3,
        filename: str = 'fig6_aggregated_probability_vectors.svg'
        ) -> None:
        """
        Plot the aggregated ON-probability vectors x_wh and x_hvac under
        two discount factors as time series.

        x_wh and x_hvac are the inputs to the OLS regression — computed
        by summing the posterior means across all DER units of each type
        at each chunk. They represent the expected total number of ON
        devices at each 10-minute interval.

        Comparing discount=0.01 vs discount=0.3 shows directly why the
        low discount produces better delta OLS results — the low discount
        vector has sharper, more informative transitions.

        Parameters
        ----------
        wh_histories_low : dict
            Output of BayesianEstimator.fit_many() for WH, discount=0.01.

        wh_histories_high : dict
            Output of BayesianEstimator.fit_many() for WH, discount=0.3.

        hvac_histories_low : dict
            Output of BayesianEstimator.fit_many() for HVAC, discount=0.01.

        hvac_histories_high : dict
            Output of BayesianEstimator.fit_many() for HVAC, discount=0.3.

        discount_low : float
            The low discount value used.

        discount_high : float
            The high discount value used.

        filename : str
            Output filename.
        """
        # ── Compute aggregated vectors ────────────────────────────────
        def aggregate(histories):
            """Sum posterior means across all units at each chunk."""
            all_means = np.array([h['mean'] for h in histories.values()])
            return all_means.sum(axis=0)

        x_wh_low    = aggregate(wh_histories_low)
        x_wh_high   = aggregate(wh_histories_high)
        x_hvac_low  = aggregate(hvac_histories_low)
        x_hvac_high = aggregate(hvac_histories_high)

        chunks = np.arange(len(x_wh_low))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # ── Top panel — WH aggregated vectors ────────────────────────
        ax1.plot(chunks, x_wh_low,
                color=COLORS['black'], linewidth=1.5,
                label=f'discount = {discount_low}')
        ax1.plot(chunks, x_wh_high,
                color=COLORS['mid_gray'], linewidth=1.5,
                linestyle='--',
                label=f'discount = {discount_high}')
        ax1.set_ylabel('Expected Units ON\n($x_{WH}$)')
        ax1.set_title('(a) Aggregated WH ON-Probability Vector')
        ax1.legend(fontsize=9)

        # ── Bottom panel — HVAC aggregated vectors ────────────────────
        ax2.plot(chunks, x_hvac_low,
                color=COLORS['black'], linewidth=1.5,
                label=f'discount = {discount_low}')
        ax2.plot(chunks, x_hvac_high,
                color=COLORS['mid_gray'], linewidth=1.5,
                linestyle='--',
                label=f'discount = {discount_high}')
        ax2.set_ylabel('Expected Units ON\n($x_{HVAC}$)')
        ax2.set_title('(b) Aggregated HVAC ON-Probability Vector')
        ax2.set_xlabel('Chunk Index (10-min intervals)')
        ax2.legend(fontsize=9)

        plt.tight_layout()
        self._save(fig, filename)

    # ── Section 3 — Standard OLS ──────────────────────────────────────────────

    def fig7_input_vectors(
        self,
        wh_histories_low: dict,
        wh_histories_high: dict,
        hvac_histories_low: dict,
        hvac_histories_high: dict,
        discount_low: float = 0.01,
        discount_high: float = 0.3,
        filename: str = 'fig7_input_vectors.svg'
        ) -> None:
        """
        Four-panel figure showing the OLS input vectors and their delta
        counterparts for both WH and HVAC under two discount factors.

        Top row:    x_wh and x_hvac — the aggregated ON-probability
                    vectors fed into standard OLS.
        Bottom row: Δx_wh and Δx_hvac — the first-order differences
                    fed into delta OLS.

        This figure bridges the Bayesian estimation and OLS sections —
        showing what the two methods receive as input and why the delta
        features are more separable than the raw vectors.

        Parameters
        ----------
        wh_histories_low : dict
            Output of BayesianEstimator.fit_many() for WH, discount=0.01.

        wh_histories_high : dict
            Output of BayesianEstimator.fit_many() for WH, discount=0.3.

        hvac_histories_low : dict
            Output of BayesianEstimator.fit_many() for HVAC, discount=0.01.

        hvac_histories_high : dict
            Output of BayesianEstimator.fit_many() for HVAC, discount=0.3.

        discount_low : float
            The low discount value used.

        discount_high : float
            The high discount value used.

        filename : str
            Output filename.
        """
        def aggregate(histories):
            return np.array([h['mean'] for h in histories.values()]).sum(axis=0)

        x_wh_low    = aggregate(wh_histories_low)
        x_wh_high   = aggregate(wh_histories_high)
        x_hvac_low  = aggregate(hvac_histories_low)
        x_hvac_high = aggregate(hvac_histories_high)

        delta_wh_low    = np.diff(x_wh_low)
        delta_wh_high   = np.diff(x_wh_high)
        delta_hvac_low  = np.diff(x_hvac_low)
        delta_hvac_high = np.diff(x_hvac_high)

        chunks       = np.arange(len(x_wh_low))
        delta_chunks = np.arange(1, len(x_wh_low))

        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)

        # ── Top left — x_wh ──────────────────────────────────────────
        axes[0, 0].plot(chunks, x_wh_low,
                        color=COLORS['black'], linewidth=1.5,
                        label=f'discount={discount_low}')
        axes[0, 0].plot(chunks, x_wh_high,
                        color=COLORS['mid_gray'], linewidth=1.5,
                        linestyle='--',
                        label=f'discount={discount_high}')
        axes[0, 0].set_ylabel('Expected Units ON')
        axes[0, 0].set_title('(a) $x_{WH}$ — WH Aggregated Vector')
        axes[0, 0].legend(fontsize=9)

        # ── Top right — x_hvac ───────────────────────────────────────
        axes[0, 1].plot(chunks, x_hvac_low,
                        color=COLORS['black'], linewidth=1.5,
                        label=f'discount={discount_low}')
        axes[0, 1].plot(chunks, x_hvac_high,
                        color=COLORS['mid_gray'], linewidth=1.5,
                        linestyle='--',
                        label=f'discount={discount_high}')
        axes[0, 1].set_ylabel('Expected Units ON')
        axes[0, 1].set_title('(b) $x_{HVAC}$ — HVAC Aggregated Vector')
        axes[0, 1].legend(fontsize=9)

        # ── Bottom left — Δx_wh ──────────────────────────────────────
        axes[1, 0].bar(delta_chunks, delta_wh_low,
                    color=COLORS['black'], alpha=0.7,
                    width=0.8, label=f'discount={discount_low}')
        axes[1, 0].bar(delta_chunks, delta_wh_high,
                    color=COLORS['mid_gray'], alpha=0.5,
                    width=0.8, label=f'discount={discount_high}')
        axes[1, 0].axhline(0, color=COLORS['mid_gray'],
                        linewidth=0.8, linestyle='--')
        axes[1, 0].set_ylabel(r'$\Delta$ Expected Units ON')
        axes[1, 0].set_xlabel('Chunk Index (10-min intervals)')
        axes[1, 0].set_title(r'(c) $\Delta x_{WH}$ — WH Delta Vector')
        axes[1, 0].legend(fontsize=9)

        # ── Bottom right — Δx_hvac ───────────────────────────────────
        axes[1, 1].bar(delta_chunks, delta_hvac_low,
                    color=COLORS['black'], alpha=0.7,
                    width=0.8, label=f'discount={discount_low}')
        axes[1, 1].bar(delta_chunks, delta_hvac_high,
                    color=COLORS['mid_gray'], alpha=0.5,
                    width=0.8, label=f'discount={discount_high}')
        axes[1, 1].axhline(0, color=COLORS['mid_gray'],
                        linewidth=0.8, linestyle='--')
        axes[1, 1].set_ylabel(r'$\Delta$ Expected Units ON')
        axes[1, 1].set_xlabel('Chunk Index (10-min intervals)')
        axes[1, 1].set_title(r'(d) $\Delta x_{HVAC}$ — HVAC Delta Vector')
        axes[1, 1].legend(fontsize=9)

        plt.tight_layout()
        self._save(fig, filename)


    def fig8_ols_comparison(
            self,
            std_wh_predicted: np.ndarray,
            std_hvac_predicted: np.ndarray,
            delta_wh_predicted: np.ndarray,
            delta_hvac_predicted: np.ndarray,
            wh_ground_truth: pd.Series,
            hvac_ground_truth: pd.Series,
            filename: str = 'fig8_ols_comparison.svg'
            ) -> None:
        """
        Compare standard OLS vs delta OLS predicted demand against
        ground truth for both WH and HVAC.

        Top panel:    WH — ground truth, standard OLS, delta OLS
        Bottom panel: HVAC — ground truth, standard OLS, delta OLS

        This figure answers whether delta OLS improves over standard
        OLS and by how much — the key performance comparison for the
        proposal.

        Parameters
        ----------
        std_wh_predicted : np.ndarray
            WH predicted demand from standard OLS. Shape: (144,).

        std_hvac_predicted : np.ndarray
            HVAC predicted demand from standard OLS. Shape: (144,).

        delta_wh_predicted : np.ndarray
            WH predicted demand from delta OLS. Shape: (144,).

        delta_hvac_predicted : np.ndarray
            HVAC predicted demand from delta OLS. Shape: (144,).

        wh_ground_truth : pd.Series
            Ground truth WH transformer demand.

        hvac_ground_truth : pd.Series
            Ground truth HVAC transformer demand.

        filename : str
            Output filename.
        """
        # ── Compute error metrics for annotation ─────────────────────
        wh_truth   = pd.to_numeric(wh_ground_truth,   errors='coerce').values
        hvac_truth = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        def clean_arrays(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            return y_true[mask], y_pred[mask]

        def rmse(y_true, y_pred):
            y_true, y_pred = clean_arrays(y_true, y_pred)
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        def r2_score(y_true, y_pred):
            y_true, y_pred = clean_arrays(y_true, y_pred)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            return np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

        std_wh_rmse     = rmse(wh_truth,   std_wh_predicted)
        delta_wh_rmse   = rmse(wh_truth,   delta_wh_predicted)
        std_hvac_rmse   = rmse(hvac_truth, std_hvac_predicted)
        delta_hvac_rmse = rmse(hvac_truth, delta_hvac_predicted)

        std_wh_r2     = r2_score(wh_truth,   std_wh_predicted)
        delta_wh_r2   = r2_score(wh_truth,   delta_wh_predicted)
        std_hvac_r2   = r2_score(hvac_truth, std_hvac_predicted)
        delta_hvac_r2 = r2_score(hvac_truth, delta_hvac_predicted)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # ── Top panel — WH ───────────────────────────────────────────
        ax1.plot(self.time_col, wh_truth / 1e3,
                color=COLORS['black'], linewidth=1.5,
                label='Ground Truth', alpha=0.85)
        ax1.plot(self.time_col, std_wh_predicted / 1e3,
                color=COLORS['mid_gray'], linewidth=1.2,
                linestyle='--', label='Standard OLS')
        ax1.plot(self.time_col, delta_wh_predicted / 1e3,
                color=COLORS['accent'], linewidth=1.2,
                linestyle=':', label='Delta OLS')
        ax1.set_ylabel('Demand [kW]')
        ax1.set_title('(a) Water Heater — Standard OLS vs Delta OLS')
        ax1.legend(fontsize=9, ncol=3)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax1.annotate(
            f'Standard OLS RMSE: {std_wh_rmse/1e3:.2f} kW\n'
            f'Standard OLS $R^2$: {std_wh_r2:.3f}\n'
            f'Delta OLS RMSE:    {delta_wh_rmse/1e3:.2f} kW\n'
            f'Delta OLS $R^2$:    {delta_wh_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.7, edgecolor=COLORS['light_gray'])
        )

        # ── Bottom panel — HVAC ──────────────────────────────────────
        ax2.plot(self.time_col, hvac_truth / 1e3,
                color=COLORS['black'], linewidth=1.5,
                label='Ground Truth', alpha=0.85)
        ax2.plot(self.time_col, std_hvac_predicted / 1e3,
                color=COLORS['mid_gray'], linewidth=1.2,
                linestyle='--', label='Standard OLS')
        ax2.plot(self.time_col, delta_hvac_predicted / 1e3,
                color=COLORS['accent'], linewidth=1.2,
                linestyle=':', label='Delta OLS')
        ax2.set_ylabel('Demand [kW]')
        ax2.set_title('(b) HVAC — Standard OLS vs Delta OLS')
        ax2.set_xlabel('Time [HH:MM]')
        ax2.legend(fontsize=9, ncol=3)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax2.annotate(
            f'Standard OLS RMSE: {std_hvac_rmse/1e3:.2f} kW\n'
            f'Standard OLS $R^2$: {std_hvac_r2:.3f}\n'
            f'Delta OLS RMSE:    {delta_hvac_rmse/1e3:.2f} kW\n'
            f'Delta OLS $R^2$:    {delta_hvac_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.7, edgecolor=COLORS['light_gray'])
        )

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
        ax1.set_ylabel(r'$\Delta$ x_{WH}')
        ax1.set_title('(a) Water Heater Delta Signal — Sharp, Infrequent Spikes')

        ax2.bar(chunks, delta_hvac, color=COLORS['dark_gray'],
                alpha=0.7, width=0.8)
        ax2.axhline(0, color=COLORS['mid_gray'], linewidth=0.8, linestyle='--')
        ax2.set_ylabel(r'$\Delta$ x_{HVAC}')
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
        wh_truth_clean   = pd.to_numeric(wh_ground_truth,   errors='coerce').values
        hvac_truth_clean = pd.to_numeric(hvac_ground_truth, errors='coerce').values

        wh_rmse   = np.sqrt(np.mean((wh_predicted   - wh_truth_clean)   ** 2))
        hvac_rmse = np.sqrt(np.mean((hvac_predicted - hvac_truth_clean) ** 2))

        wh_r2   = 1 - np.sum((wh_predicted   - wh_truth_clean)   ** 2) / \
                    np.sum((wh_truth_clean   - wh_truth_clean.mean())   ** 2)
        hvac_r2 = 1 - np.sum((hvac_predicted - hvac_truth_clean) ** 2) / \
                    np.sum((hvac_truth_clean - hvac_truth_clean.mean()) ** 2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(self.time_col,
                wh_truth_clean / 1e3,
                color=COLORS['black'], linewidth=1.2,
                label='Ground Truth', alpha=0.8)
        ax1.plot(self.time_col, wh_predicted / 1e3,
                color=COLORS['accent'], linewidth=1.2,
                linestyle='--', label='Delta OLS Predicted')
        ax1.set_ylabel('Demand [kW]')
        ax1.set_title('(a) Water Heater — Delta OLS (discount = 0.01)')
        ax1.legend()
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax1.annotate(
            f'RMSE: {wh_rmse/1e3:.2f} kW\n$R^2$: {wh_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.7, edgecolor=COLORS['light_gray'])
        )

        ax2.plot(self.time_col,
                hvac_truth_clean / 1e3,
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
        ax2.annotate(
            f'RMSE: {hvac_rmse/1e3:.2f} kW\n$R^2$: {hvac_r2:.3f}',
            xy=(0.01, 0.95), xycoords='axes fraction',
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.7, edgecolor=COLORS['light_gray'])
        )

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
    
    def fig12_bar_chart_for_each_device (
            self,
            estimated_data : dict,
            ground_truth_data : dict
            ) -> None:
        
        fig, axes = plt.subplots (figsize = (16, 10))
        gt_dict = {}
        print("estimated data\n")
        print(estimated_data)
        # for filename, df in ground_truth_data.items():
            

        # gt_series = pd.Series(gt_dict)