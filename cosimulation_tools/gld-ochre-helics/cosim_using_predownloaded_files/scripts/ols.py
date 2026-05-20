'''
Author: Midrar Adham
Created: Fri Apr 24 2026
'''
import numpy as np
import pandas as pd


class OrdinaryLeastSquare:

    def __init__(
        self,
        wh_histories:   dict,
        hvac_histories: dict,
        wh_all_dfs:     dict,
        hvac_all_dfs:   dict,
        feeder_demand:  pd.DataFrame,
    ):
        """
        Store everything OLS needs to run its analysis.

        Parameters
        ----------
        wh_histories : dict
            Output of BayesianEstimator.fit_many() for WH data.
            Keys are filenames, values are history dicts containing
            'mean', 'variance', etc. for each chunk.

        hvac_histories : dict
            Same as wh_histories but for HVAC data.

        wh_all_dfs : dict
            Output of DataLoader.all_dfs for WH data.

        hvac_all_dfs : dict
            Output of DataLoader.all_dfs for HVAC data.

        feeder_demand : pd.DataFrame
            Transformer-level feeder signal from
            DataLoader.load_transformer_data(). This is the signal
            we are trying to decompose.
        """
        self.wh_histories   = wh_histories
        self.hvac_histories = hvac_histories
        self.wh_all_dfs     = wh_all_dfs
        self.hvac_all_dfs   = hvac_all_dfs
        self.feeder_demand  = feeder_demand

    # ── Matrix builders ───────────────────────────────────────────────────────

    def _build_mean_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the posterior mean values from each DER's Bayesian history
        and organize them into a single matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'mean' list.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each column is one DER's posterior
            mean ON-probability sequence across all chunks.
        """
        return pd.DataFrame(
            {filename: history['mean']
             for filename, history in histories.items()}
        )

    def _build_variance_matrix(self, histories: dict) -> pd.DataFrame:
        """
        Extract the posterior variance values from each DER's Bayesian
        history and organize them into a matrix.

        Parameters
        ----------
        histories : dict
            Output of BayesianEstimator.fit_many(). Keys are filenames,
            values are history dicts each containing a 'variance' list.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each column is one DER's posterior
            variance sequence across all chunks.
        """
        return pd.DataFrame(
            {filename: history['variance']
             for filename, history in histories.items()}
        )

    def _build_state_matrix(self, all_dfs: dict) -> pd.DataFrame:
        """
        Build a binary state matrix from raw per-device dataframes,
        resampled to 10-minute chunks.

        Parameters
        ----------
        all_dfs : dict
            Output of DataLoader.all_dfs for WH or HVAC data.

        Returns
        -------
        pd.DataFrame
            Shape (n_chunks, n_ders). Each value is the mean state
            (fraction of minutes ON) within each 10-minute chunk.
        """
        state_matrix = {}
        for filename, df in all_dfs.items():
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            states = df.set_index('time')['state']
            states.index = pd.to_datetime(states.index)
            states = states.resample('10min').mean()
            state_matrix[filename] = states.values
        return pd.DataFrame(state_matrix)

    # ── OLS methods ───────────────────────────────────────────────────────────

    def _run_simultaneous_ols(
        self,
        wh_mean_matrix:   pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
    ) -> dict:
        """
        Stage 1 — Warm-up OLS.

        Estimates a single rated power per DER type using aggregated
        Bayesian posterior means as regressors.

        Solves:
            feeder(t) - background ≈ kw_wh × x_wh(t) + kw_hvac × x_hvac(t)

        This is the fallback method used when insufficient history has
        accumulated for per-device estimation.

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.

        Returns
        -------
        dict
            Contains kw_wh, kw_hvac, wh_predicted, hvac_predicted,
            x_wh, x_hvac, feeder_minus_background.
        """
        x_wh   = wh_mean_matrix.sum(axis=1).values
        x_hvac = hvac_mean_matrix.sum(axis=1).values

        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        A = np.column_stack([x_wh, x_hvac])
        kw_estimate, _, _, _ = np.linalg.lstsq(A, feeder_minus_background,
                                                rcond=None)
        kw_wh, kw_hvac = kw_estimate

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac':                 kw_hvac,
            'wh_predicted':            kw_wh   * x_wh,
            'hvac_predicted':          kw_hvac * x_hvac,
            'x_wh':                    x_wh,
            'x_hvac':                  x_hvac,
            'feeder_minus_background': feeder_minus_background,
        }

    def _run_per_device_hvac_ols(
        self,
        wh_mean_matrix:   pd.DataFrame,
        hvac_mean_matrix: pd.DataFrame,
        exclude:          list = None,
    ) -> dict:
        """
        Stage 2 — Per-device OLS.

        Estimates a separate rated power for each HVAC device using
        its individual Bayesian posterior mean as a regressor. Requires
        sufficient history (empirically ~10 days) for stable estimates.

        Solves:
            feeder(t) - background ≈ kw_wh × x_wh(t)
                                   + Σ_i [ kw_hvac_i × mean_i(t) ]

        Devices that were never ON (max posterior mean ≤ 0.01) are
        automatically excluded as they carry no information. Additional
        devices can be excluded via the `exclude` parameter (e.g.
        two-state devices that violate the ON/OFF power assumption).

        Parameters
        ----------
        wh_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for WH histories.
            Shape: (n_chunks, n_wh).

        hvac_mean_matrix : pd.DataFrame
            Output of _build_mean_matrix() for HVAC histories.
            Shape: (n_chunks, n_hvac).

        exclude : list, optional
            List of HVAC device filenames to exclude from estimation.
            Excluded devices receive a coefficient of 0.0.

        Returns
        -------
        dict
            Contains:
            - 'kw_wh'             : float, estimated W per WH unit
            - 'kw_hvac_per_device': pd.Series, estimated W per HVAC device
            - 'wh_predicted'      : np.ndarray, shape (n_chunks,)
            - 'hvac_predicted'    : np.ndarray, shape (n_chunks,)
            - 'x_wh'              : np.ndarray, shape (n_chunks,)
            - 'hvac_active'       : pd.DataFrame, active device mean matrix
            - 'feeder_minus_background': np.ndarray, shape (n_chunks,)
        """
        exclude = exclude or []

        # ── WH aggregated regressor ──────────────────────────────────
        x_wh = wh_mean_matrix.sum(axis=1).values

        # ── Drop never-ON and explicitly excluded HVAC devices ───────
        active_cols = [
            col for col in hvac_mean_matrix.columns
            if hvac_mean_matrix[col].max() > 0.01 and col not in exclude
        ]
        hvac_active = hvac_mean_matrix[active_cols]

        # ── Background subtraction ───────────────────────────────────
        background_constant     = self.feeder_demand['power_out'].values.min()
        feeder_minus_background = (
            self.feeder_demand['power_out'].values - background_constant
        )

        # ── Design matrix: [x_wh | mean_1 | mean_2 | ... | mean_N] ─
        A = np.column_stack([x_wh, hvac_active.values])

        # ── Solve OLS ────────────────────────────────────────────────
        coefs, _, _, _ = np.linalg.lstsq(A, feeder_minus_background,
                                         rcond=None)
        kw_wh = coefs[0]

        # Assign coefficients — zero for excluded/never-ON devices
        kw_hvac_per_device = pd.Series(0.0, index=hvac_mean_matrix.columns)
        kw_hvac_per_device[active_cols] = coefs[1:]

        return {
            'kw_wh':                   kw_wh,
            'kw_hvac_per_device':      kw_hvac_per_device,
            'wh_predicted':            kw_wh * x_wh,
            'hvac_predicted':          hvac_active.values @ coefs[1:],
            'x_wh':                    x_wh,
            'hvac_active':             hvac_active,
            'feeder_minus_background': feeder_minus_background,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        exclude_hvac: list = None,
    ) -> dict:
        """
        Run the full OLS pipeline.

        Builds the mean matrices from Bayesian histories, then runs
        both Stage 1 (simultaneous OLS) and Stage 2 (per-device OLS).

        Parameters
        ----------
        exclude_hvac : list, optional
            HVAC device filenames to exclude from per-device estimation.
            Used for two-state devices or other anomalous devices.

        Returns
        -------
        dict
            Combined results from both OLS stages. Keys are prefixed:
            - 'sim_*'    : Stage 1 simultaneous OLS results
            - 'per_d_*'  : Stage 2 per-device OLS results
        """
        # ── Build matrices ───────────────────────────────────────────
        wh_mean_matrix   = self._build_mean_matrix(histories=self.wh_histories)
        hvac_mean_matrix = self._build_mean_matrix(histories=self.hvac_histories)

        # ── Stage 1: Simultaneous OLS ────────────────────────────────
        sim_results = self._run_simultaneous_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
        )

        # ── Stage 2: Per-device OLS ──────────────────────────────────
        per_device_results = self._run_per_device_hvac_ols(
            wh_mean_matrix=wh_mean_matrix,
            hvac_mean_matrix=hvac_mean_matrix,
            exclude=exclude_hvac,
        )

        return {
            # Stage 1 — simultaneous OLS
            'sim_kw_wh':                sim_results['kw_wh'],
            'sim_kw_hvac':              sim_results['kw_hvac'],
            'sim_wh_predicted':         sim_results['wh_predicted'],
            'sim_hvac_predicted':       sim_results['hvac_predicted'],
            'sim_x_wh':                 sim_results['x_wh'],
            'sim_x_hvac':               sim_results['x_hvac'],
            'sim_feeder_minus_bg':      sim_results['feeder_minus_background'],

            # Stage 2 — per-device OLS
            'per_d_kw_wh':              per_device_results['kw_wh'],
            'per_d_kw_hvac':            per_device_results['kw_hvac_per_device'],
            'per_d_wh_predicted':       per_device_results['wh_predicted'],
            'per_d_hvac_predicted':     per_device_results['hvac_predicted'],
            'per_d_x_wh':               per_device_results['x_wh'],
            'per_d_hvac_active':        per_device_results['hvac_active'],
            'per_d_feeder_minus_bg':    per_device_results['feeder_minus_background'],
        }
